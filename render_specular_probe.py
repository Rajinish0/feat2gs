import matplotlib
matplotlib.use('Agg')

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_gsplat
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.pose_utils import get_tensor_from_camera
from PIL import Image
import torchvision.transforms.functional as tf

import wandb
from wandb_utils import init_probe_run


def render_set_specular_probe(model_path, name, iteration, views, gaussians, pipeline, background, msk_scr_path, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    ablated_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_ablated_specular")
    diff_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diff_specular")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    msks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")
    msk_suffix = os.path.basename(os.listdir(msk_scr_path)[0]).split('.')[-1]

    for p in (render_path, ablated_path, diff_path, gts_path, msks_path):
        makedirs(p, exist_ok=True)

    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress (specular probe)")):
        num_iter = args.optim_test_pose_iter
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))

        camera_tensor_T = camera_pose[-3:].requires_grad_()
        camera_tensor_q = camera_pose[:4].requires_grad_()
        pose_optimizer = torch.optim.Adam([
            {"params": [camera_tensor_T], "lr": 0.0003},
            {"params": [camera_tensor_q], "lr": 0.0001},
        ])

        candidate_q = camera_tensor_q.clone().detach()
        candidate_T = camera_tensor_T.clone().detach()
        current_min_loss = float(1e20)
        gt = view.original_image[0:3, :, :]

        mask = Image.open(os.path.join(msk_scr_path, view.image_name + '.' + msk_suffix))
        scale = view.image_width / 1600 if view.image_width > 1600 else 1.
        resolution = (int(view.image_height / scale), int(view.image_width / scale))
        mask = tf.resize(tf.to_tensor(mask), resolution)

        for iteration_ in range(num_iter):
            rendering = render_gsplat(
                view, gaussians, pipeline, background,
                camera_pose=torch.cat([camera_tensor_q, camera_tensor_T])
            )["render"]
            loss = torch.abs((gt - rendering) * mask.to(gt.device)).mean()
            loss.backward()
            with torch.no_grad():
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)
                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_q = camera_tensor_q.clone().detach()
                    candidate_T = camera_tensor_T.clone().detach()

        opt_pose = torch.cat([candidate_q, candidate_T])

        with torch.no_grad():
            rendering_full = render_gsplat(
                view, gaussians, pipeline, background, camera_pose=opt_pose
            )["render"].contiguous()

            saved_f_rest = gaussians._features_rest.clone()
            gaussians._features_rest.zero_()
            rendering_ablated = render_gsplat(
                view, gaussians, pipeline, background, camera_pose=opt_pose
            )["render"].contiguous()
            gaussians._features_rest.copy_(saved_f_rest)

            diff = torch.abs(rendering_full - rendering_ablated)

        torchvision.utils.save_image(rendering_full, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
        torchvision.utils.save_image(rendering_ablated, os.path.join(ablated_path, "{0:05d}".format(idx) + ".png"))
        torchvision.utils.save_image(diff, os.path.join(diff_path, "{0:05d}".format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))
        torchvision.utils.save_image(mask, os.path.join(msks_path, "{0:05d}".format(idx) + ".png"))

        if wandb.run is not None and idx < 3:
            wandb.log({
                f"specular_probe_imgs/{view.image_name}_full": wandb.Image(rendering_full.cpu().numpy().transpose(1, 2, 0)),
                f"specular_probe_imgs/{view.image_name}_ablated": wandb.Image(rendering_ablated.cpu().numpy().transpose(1, 2, 0)),
                f"specular_probe_imgs/{view.image_name}_diff": wandb.Image(diff.cpu().numpy().transpose(1, 2, 0)),
                f"specular_probe_imgs/{view.image_name}_gt": wandb.Image(gt.cpu().numpy().transpose(1, 2, 0)),
            })


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_test: bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    msk_path = os.path.join(args.source_path, "test_view/masks")

    if not skip_test:
        render_set_specular_probe(
            dataset.model_path, "test", scene.loaded_iter,
            scene.getTestCameras(), gaussians, pipeline, background, msk_path, args
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Specular ablation probe")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--n_views", default=None, type=int)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    parser.add_argument("--method", type=str, default='dust3r')
    parser.add_argument("--feat_type", type=str, nargs='*', default=None)

    args = get_combined_args(parser)
    print("Rendering (specular probe) " + args.model_path)
    init_probe_run(args.model_path, job_type="specular_probe_render")

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_test, args)
    wandb.finish()
