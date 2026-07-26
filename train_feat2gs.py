#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_gsplat, network_gui
import sys
from scene import Scene, Feat2GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.pose_utils import get_camera_from_tensor

import wandb
from wandb_utils import init_probe_run

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
from time import perf_counter

def save_pose(path, quat_pose, train_cams, llffhold=2):
    output_poses=[]
    index_colmap = [cam.colmap_id for cam in train_cams]
    for quat_t in quat_pose:
        w2c = get_camera_from_tensor(quat_t)
        output_poses.append(w2c)
    colmap_poses = []
    for i in range(len(index_colmap)):
        ind = index_colmap.index(i+1)
        bb=output_poses[ind]
        bb = bb
        colmap_poses.append(bb)
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def log_sh_magnitudes(gaussians, iteration):
    with torch.no_grad():
        f_dc = gaussians._features_dc.reshape(gaussians._features_dc.shape[0], -1)
        dc_norm = f_dc.norm(dim=1)
        log_dict = {
            "sh/f_dc_norm_mean": dc_norm.mean().item(),
            "sh/f_dc_norm_std": dc_norm.std().item(),
        }
        if gaussians._features_rest.numel() > 0:
            f_rest = gaussians._features_rest.reshape(gaussians._features_rest.shape[0], -1)
            rest_norm = f_rest.norm(dim=1)
            log_dict.update({
                "sh/f_rest_norm_mean": rest_norm.mean().item(),
                "sh/f_rest_norm_std": rest_norm.std().item(),
                "sh/f_rest_to_f_dc_ratio": (rest_norm.mean() / dc_norm.mean().clamp_min(1e-8)).item(),
            })
        wandb.log(log_dict, step=iteration)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt.iterations)
    feat_type = '-'.join(args.feat_type)
    feat_dim = args.feat_dim if feat_type not in ['iuv', 'iuvrgb'] else dataset.feat_default_dim[feat_type]
    gs_params_group = dataset.gs_params_group[args.model]
    gaussians = Feat2GaussianModel(dataset.sh_degree, feat_dim, gs_params_group)
    scene = Scene(dataset, gaussians, opt=args, shuffle=True)                                                                      
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    train_cams_init = scene.getTrainCameras().copy()
    os.makedirs(scene.model_path + 'pose', exist_ok=True)
    save_pose(scene.model_path + 'pose' + "/pose_org.npy", gaussians.P, train_cams_init)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    warm_iter = 1000 if len(gs_params_group.get('head', [])) > 0 else 0

    start = perf_counter()
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        if iteration > warm_iter:
            if iteration == warm_iter+1:
                gaussians.pc_feat.requires_grad_(False)
                gaussians.setup_rendering_learning_rate()
            gaussians.update_learning_rate(iteration - warm_iter)
        else:
            gaussians.update_warm_start_learning_rate(iteration)

        if args.optim_pose==False:
            gaussians.P.requires_grad_(False)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        pose = gaussians.get_RT(viewpoint_cam.uid)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        gaussians.inference()

        pretrained_loss_dict = {
            'xyz': l1_loss(gaussians._xyz, gaussians.param_init['xyz']),
            'f_dc': l1_loss(gaussians._features_dc, gaussians.param_init['f_dc']),
            'f_rest': l1_loss(gaussians._features_rest, gaussians.param_init['f_rest']),
            'opacity': l1_loss(gaussians._opacity, gaussians.param_init['opacity']),
            'scaling': l1_loss(gaussians._scaling, gaussians.param_init['scaling']),
            'rotation': l1_loss(gaussians._rotation, gaussians.param_init['rotation']),
            'pose': l1_loss(gaussians.P, gaussians.param_init['pose']),
            'pc_feat':l1_loss(gaussians.pc_feat, gaussians.param_init['pc_feat']),
            }

        if iteration <= warm_iter:
            loss = sum(loss for key, loss in pretrained_loss_dict.items() if key in gs_params_group['head'])
            Ll1 = torch.tensor(0)

        if iteration > warm_iter:
            render_pkg = render_gsplat(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 

            if feat_type in ['iuv', 'iuvrgb']:
                loss += l1_loss(gaussians._scaling, gaussians.param_init['scaling']) * 0.1

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                wandb.log({
                    "train/loss": loss.item(),
                    "train/ema_loss": ema_loss_for_log,
                    "train/l1": Ll1.item(),
                }, step=iteration)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 100 == 0:
                log_sh_magnitudes(gaussians, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_gsplat, (pipe, background), pretrained_loss_dict)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(scene.model_path + 'pose' + f"/pose_{iteration}.npy", gaussians.P, train_cams_init)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
        end = perf_counter()
        train_time = end - start


def prepare_output_and_logger(args, iteration=None):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(os.path.join(args.model_path, f"log_{iteration}"))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, prop_pred=None):
    if tb_writer:
        tb_writer.add_scalar('train_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for key, values in prop_pred.items():
            tb_writer.add_scalar(f'train_patches/delta_{key}', values.item(), iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, camera_pose=pose)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    if wandb.run is not None and idx < 3:
                        wandb.log({
                            f"images/{config['name']}_{viewpoint.image_name}_render": wandb.Image(
                                image.clamp(0, 1).contiguous().permute(1, 2, 0).cpu().numpy()),
                            f"images/{config['name']}_{viewpoint.image_name}_gt": wandb.Image(
                                gt_image.clamp(0, 1).contiguous().permute(1, 2, 0).cpu().numpy()),
                        }, step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb.run is not None:
                    wandb.log({
                        f"eval/{config['name']}_l1": l1_test.item(),
                        f"eval/{config['name']}_psnr": psnr_test.item(),
                    }, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, 
                        default=[500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7_000, \
                                 8_000, 9_000, 10_000, 11_000, 12_000, 13_000, 14_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--optim_pose", action="store_true")
    parser.add_argument("--feat_type", type=str, nargs='*', default=None, help="Feature type(s). Multiple types can be specified for combination.")
    parser.add_argument("--method", type=str, default='dust3r', help="Method of Initialization, e.g., 'dust3r' or 'mast3r'")
    parser.add_argument("--feat_dim", type=int, default=None, help="Feture dimension after PCA . If None, PCA is not applied.")
    parser.add_argument("--model", type=str, default='G', help="Model of Feat2gs, 'G'='geometry'/'T'='texture'/'A'='all'")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)

    init_probe_run(args.model_path, job_type="train", extra_config={
        "n_views": args.n_views, "iterations": args.iterations, "model": args.model,
        "feat_type": args.feat_type, "method": args.method,
    })

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    print("\nTraining complete.")
    wandb.finish()
