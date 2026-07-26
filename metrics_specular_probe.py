import os
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from argparse import ArgumentParser

import wandb
from wandb_utils import init_probe_run


def read_triplet(renders_dir, ablated_dir, gt_dir, msk_dir):
    renders, ablated, gts, masks, names = [], [], [], [], []
    for fname in sorted(os.listdir(gt_dir)):
        gt = tf.to_tensor(Image.open(gt_dir / fname)).unsqueeze(0)[:, :3].cuda().contiguous()
        full = tf.to_tensor(Image.open(renders_dir / fname)).unsqueeze(0)[:, :3].cuda().contiguous()
        abl = tf.to_tensor(Image.open(ablated_dir / fname)).unsqueeze(0)[:, :3].cuda().contiguous()
        mask = tf.to_tensor(Image.open(msk_dir / fname)).int().unsqueeze(0)[:, 0:1].cuda().contiguous()
        gts.append(gt); renders.append(full); ablated.append(abl); masks.append(mask); names.append(fname)
    return renders, ablated, gts, masks, names


def evaluate_specular_probe(run_dir, iteration=8000):
    run_dir = Path(run_dir) / "test" / f"ours_{iteration}"
    gt_dir = run_dir / "gt"
    full_dir = run_dir / "renders"
    ablated_dir = run_dir / "renders_ablated_specular"
    msk_dir = run_dir / "masks"
    diff_out_dir = run_dir / "diff_specular_colored"
    diff_out_dir.mkdir(exist_ok=True)

    renders, ablated, gts, masks, names = read_triplet(full_dir, ablated_dir, gt_dir, msk_dir)

    jet_cmap = plt.cm.jet
    rows = []
    for idx in range(len(renders)):
        gt, full, abl, mask = gts[idx], renders[idx], ablated[idx], masks[idx]
        gt_m, full_m, abl_m = gt * mask, full * mask, abl * mask

        full_psnr = psnr(full_m, gt_m).item()
        abl_psnr = psnr(abl_m, gt_m).item()
        full_ssim = ssim(full_m, gt_m).item()
        abl_ssim = ssim(abl_m, gt_m).item()
        full_lpips = lpips(full_m, gt_m, net_type='vgg').item()
        abl_lpips = lpips(abl_m, gt_m, net_type='vgg').item()

        spec_diff_map = (torch.abs(full_m - abl_m).mean(1) * mask.squeeze(1)).squeeze()
        spec_colored = jet_cmap((spec_diff_map.clamp(0, 0.2) * 5).cpu().numpy())
        Image.fromarray((spec_colored[:, :, :3] * 255).astype(np.uint8)).save(diff_out_dir / names[idx])

        # Global mean gets washed out by mostly-diffuse/background pixels --
        # specular highlights are spatially sparse, so also report the peak
        # of the diff distribution, which is where the real signal lives.
        flat = spec_diff_map[mask.squeeze() > 0]
        mean_diff = flat.mean().item() if flat.numel() else 0.0
        p99_diff = torch.quantile(flat, 0.99).item() if flat.numel() else 0.0
        max_diff = flat.max().item() if flat.numel() else 0.0

        rows.append(dict(
            name=names[idx],
            full_psnr=full_psnr, abl_psnr=abl_psnr,
            full_ssim=full_ssim, abl_ssim=abl_ssim,
            full_lpips=full_lpips, abl_lpips=abl_lpips,
            delta_psnr=full_psnr - abl_psnr,
            delta_ssim=full_ssim - abl_ssim,
            delta_lpips=abl_lpips - full_lpips,
            mean_specular_diff=mean_diff,
            p99_specular_diff=p99_diff,
            max_specular_diff=max_diff,
        ))

    avg = lambda k: sum(r[k] for r in rows) / len(rows)
    print(f"\n=== {run_dir} ===")
    print(f"{'':10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print(f"{'full':10} {avg('full_psnr'):8.3f} {avg('full_ssim'):8.4f} {avg('full_lpips'):8.4f}")
    print(f"{'ablated':10} {avg('abl_psnr'):8.3f} {avg('abl_ssim'):8.4f} {avg('abl_lpips'):8.4f}")
    print(f"{'delta':10} {avg('delta_psnr'):8.3f} {avg('delta_ssim'):8.4f} {avg('delta_lpips'):8.4f}")
    print(f"\nspecular diff: mean={avg('mean_specular_diff'):.5f}  p99={avg('p99_specular_diff'):.5f}  max={avg('max_specular_diff'):.5f}")

    print(f"\nper-frame delta_psnr, sorted:")
    for r in sorted(rows, key=lambda r: -r['delta_psnr']):
        print(f"  {r['name']:14s} delta_psnr={r['delta_psnr']:+.3f}  p99_diff={r['p99_specular_diff']:.5f}  max_diff={r['max_specular_diff']:.5f}")

    with open(run_dir / "specular_probe_results.json", 'w') as f:
        json.dump(rows, f, indent=2)

    return rows

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--iteration", type=int, default=8000)
    args = p.parse_args()

    init_probe_run(args.run_dir, job_type="specular_probe_metrics")
    rows = evaluate_specular_probe(args.run_dir, args.iteration)

    avg = lambda k: sum(r[k] for r in rows) / len(rows)
    wandb.log({
        "specular_probe/full_psnr": avg('full_psnr'), "specular_probe/ablated_psnr": avg('abl_psnr'),
        "specular_probe/delta_psnr": avg('delta_psnr'),
        "specular_probe/full_ssim": avg('full_ssim'), "specular_probe/ablated_ssim": avg('abl_ssim'),
        "specular_probe/delta_ssim": avg('delta_ssim'),
        "specular_probe/full_lpips": avg('full_lpips'), "specular_probe/ablated_lpips": avg('abl_lpips'),
        "specular_probe/delta_lpips": avg('delta_lpips'),
        "specular_probe/mean_specular_diff": avg('mean_specular_diff'),
        "specular_probe/p99_specular_diff": avg('p99_specular_diff'),
        "specular_probe/max_specular_diff": avg('max_specular_diff'),
    })
    wandb.finish()
