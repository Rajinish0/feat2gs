#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import matplotlib
matplotlib.use('Agg')

import math
import copy
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_gsplat
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.pose_utils import get_tensor_from_camera
import shutil
import numpy as np
import imageio.v3 as iio
from utils.graphics_utils import resize_render, make_video_divisble
from utils.camera_utils import visualizer

from utils.trajectories import (
    get_arc_w2cs,
    get_avg_w2c,
    get_lemniscate_w2cs,
    get_spiral_w2cs,
    get_wander_w2cs,
    get_lookat,
)

from utils.camera_utils import generate_interpolated_path, generate_ellipse_path
from utils.camera_traj_config import trajectory_configs


def save_interpolated_pose(model_path, iter, n_views):
    org_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    n_interp = int(10 * 30 / n_views)  # 10second, fps=30
    all_inter_pose = []
    for i in range(n_views-1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.array(all_inter_pose).reshape(-1, 3, 4)

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    return inter_pose


def save_ellipse_pose(model_path, iter, n_views):
    org_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    n_interp = int(10 * 30 / n_views) * (n_views-1)  # 10second, fps=30
    all_inter_pose = generate_ellipse_path(org_pose, n_interp)

    inter_pose_list = []
    for p in all_inter_pose:
        c2w = np.eye(4)
        c2w[:3, :4] = p
        inter_pose_list.append(np.linalg.inv(c2w))
    inter_pose = np.stack(inter_pose_list, 0)

    return inter_pose


def save_traj_pose(dataset, iter, args):
    traj_up = trajectory_configs.get(args.dataset, {}).get(args.scene, {}).get('up', [-1, 1])
    traj_params = trajectory_configs.get(args.dataset, {}).get(args.scene, {}).get(args.cam_traj, {})

    org_pose = np.load(dataset.model_path + f"pose/pose_{iter}.npy")
    train_w2cs = torch.from_numpy(org_pose).cuda()

    avg_w2c = get_avg_w2c(train_w2cs)
    train_c2ws = torch.linalg.inv(train_w2cs)
    lookat = get_lookat(train_c2ws[:, :3, -1], train_c2ws[:, :3, 2])
    avg_c2w = torch.linalg.inv(avg_w2c)
    up = traj_up[0] * (avg_c2w[:3, traj_up[1]])

    tmp_args = copy.deepcopy(args)
    tmp_args.get_video = False
    tmp_dataset = copy.deepcopy(dataset)
    tmp_dataset.eval = False
    with torch.no_grad():
        temp_gaussians = GaussianModel(dataset.sh_degree)
        temp_scene = Scene(tmp_dataset, temp_gaussians, load_iteration=iter, opt=tmp_args, shuffle=False)

    view = temp_scene.getTrainCameras()[0]
    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)
    focal_length_x = view.image_width / (2 * tanfovx)
    focal_length_y = view.image_height / (2 * tanfovy)

    K = torch.tensor([[focal_length_x, 0, view.image_width/2],
                     [0, focal_length_y, view.image_height/2],
                     [0, 0, 1]], device="cuda")
    img_wh = (view.image_width, view.image_height)

    del temp_scene
    del temp_gaussians

    rc_train_c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(avg_w2c), train_c2ws)
    rc_pos = rc_train_c2ws[:, :3, -1]
    rads = (rc_pos.amax(0) - rc_pos.amin(0)) * 1.25

    num_frames = int(10 * 30 / args.n_views) * (args.n_views-1)

    if args.cam_traj == 'arc':
        w2cs = get_arc_w2cs(ref_w2c=avg_w2c, lookat=lookat, up=up, focal_length=K[0, 0].item(),
                             rads=rads, num_frames=num_frames, degree=traj_params.get('degree', 180.0))
    elif args.cam_traj == 'spiral':
        w2cs = get_spiral_w2cs(ref_w2c=avg_w2c, lookat=lookat, up=up, focal_length=K[0, 0].item(),
                                rads=rads, num_frames=num_frames, zrate=traj_params.get('zrate', 0.5),
                                rots=traj_params.get('rots', 1))
    elif args.cam_traj == 'lemniscate':
        w2cs = get_lemniscate_w2cs(ref_w2c=avg_w2c, lookat=lookat, up=up, focal_length=K[0, 0].item(),
                                    rads=rads, num_frames=num_frames, degree=traj_params.get('degree', 45.0))
    elif args.cam_traj == 'wander':
        w2cs = get_wander_w2cs(ref_w2c=avg_w2c, focal_length=K[0, 0].item(), num_frames=num_frames,
                                max_disp=traj_params.get('max_disp', 48.0))
    else:
        raise ValueError(f"Unknown camera trajectory: {args.cam_traj}")

    return w2cs.cpu().numpy()


def render_frames(w2cs, scene, gaussians, pipeline, background, args):
    """Renders one trajectory pass over the current state of `gaussians`.
    Wrapped in no_grad since this is inference-only, not training."""
    video = []
    with torch.no_grad():
        for idx, w2c in enumerate(tqdm(w2cs, desc="Rendering progress")):
            camera_pose = get_tensor_from_camera(w2c.transpose(0, 1))
            view = scene.getTrainCameras()[0]  # template for intrinsics
            if args.resize:
                view = resize_render(view)

            rendering = render_gsplat(
                view, gaussians, pipeline, background, camera_pose=camera_pose
            )["render"]

            img = (torch.clamp(rendering, 0, 1).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
            video.append(img)

    video = np.stack(video, 0).transpose(0, 2, 3, 1)
    return video


def save_video(video, output_video_file):
    iio.imwrite(
        output_video_file,
        make_video_divisble(video),
        fps=30,
        codec='libx264',
        quality=None,
        output_params=[
            '-crf', '28',
            '-preset', 'veryslow',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart'
        ]
    )


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, args):
    if args.cam_traj in ['interpolated', 'ellipse']:
        w2cs = globals().get(f'save_{args.cam_traj}_pose')(dataset.model_path, iteration, args.n_views)
    else:
        w2cs = save_traj_pose(dataset, iteration, args)

    visualizer(w2cs, ["blue" for _ in w2cs], dataset.model_path + f"pose/poses_{args.cam_traj}.png")
    np.save(dataset.model_path + f"pose/pose_{args.cam_traj}.npy", w2cs)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    video_dir = os.path.join(dataset.model_path, 'videos')
    os.makedirs(video_dir, exist_ok=True)

    # --- Pass 1: full render, f_rest (view-dependent / specular) intact ---
    video_full = render_frames(w2cs, scene, gaussians, pipeline, background, args)
    if args.get_video:
        out_full = os.path.join(video_dir, f'{args.scene}_{args.n_views}_view_{args.cam_traj}.mp4')
        save_video(video_full, out_full)
        print(f"Saved: {out_full}")

    # --- Pass 2: ablated render, f_rest zeroed -> only f_dc (diffuse) contributes ---
    saved_f_rest = gaussians._features_rest.detach().clone()
    with torch.no_grad():
        gaussians._features_rest.zero_()

    video_fdc = render_frames(w2cs, scene, gaussians, pipeline, background, args)

    with torch.no_grad():
        gaussians._features_rest.copy_(saved_f_rest)  # restore, in case gaussians is reused after this call

    if args.get_video:
        out_fdc = os.path.join(video_dir, f'{args.scene}_{args.n_views}_view_{args.cam_traj}_fdc.mp4')
        save_video(video_fdc, out_fdc)
        print(f"Saved: {out_fdc}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--n_views", default=120, type=int)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--cam_traj", default='arc', type=str,
                       choices=['arc', 'spiral', 'lemniscate', 'wander', 'interpolated', 'ellipse'],
                       help="Camera trajectory type")
    parser.add_argument("--resize", action="store_true", default=True,
                       help="If True, resize rendering to square")
    parser.add_argument("--feat_type", type=str, nargs='*', default=None,
                       help="Feature type(s). Multiple types can be specified for combination.")
    parser.add_argument("--method", type=str, default='dust3r',
                       help="Method of Initialization, e.g., 'dust3r' or 'mast3r'")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args,
    )
