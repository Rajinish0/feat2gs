import re
import json
import glob
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

OUTPUT_ROOT = Path("/home/stud124/output/Feat2gs/output/eval")
ORACLE_ROOT = Path.home() / "gaussian-splatting/output"
STAGING_ROOT = Path.home() / "dataset/eval_tmp_data/refnerf"

ORACLE_RUN_IDS = {
    "toaster": "2f2d3295-6",
    "ball": "4f91fe01-e",
    "car": "94290c1c-b",
    "helmet": "943decee-8",
    "teapot": "0172f272-3",
    "coffee": "e6973763-a"
    # add remaining scenes here as their dense oracles finish
}

TOP_PCT = 0.20


def get_idx_to_rN(scene):
    files = sorted(glob.glob(str(STAGING_ROOT / scene / "test_*.png")))
    mapping = {}
    for i, f in enumerate(files):
        m = re.search(r"test_\d+_(r_\d+)\.png$", Path(f).name)
        if m:
            mapping[i] = m.group(1)
    return mapping


def load_gray(path, size=None):
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


_diagnosed_scenes = set()

def load_oracle_diff(scene, r_name, target_size):
    run_id = ORACLE_RUN_IDS.get(scene)
    if run_id is None:
        return None
    npy_path = ORACLE_ROOT / run_id / "ablation_probe" / f"{r_name}_diff_raw.npy"
    if not npy_path.exists():
        return None
    diff = np.load(npy_path)

    if scene not in _diagnosed_scenes:
        _diagnosed_scenes.add(scene)
        print(f"  [diagnostic] {scene} oracle diff ({r_name}): "
              f"std={diff.std():.6f}  max={diff.max():.6f}  mean={diff.mean():.6f}")

    diff_norm = diff / (diff.max() + 1e-8)
    diff_resized = cv2.resize(diff_norm, target_size, interpolation=cv2.INTER_AREA)
    return diff_resized


def compute_sparse_diff(run_dir, idx):
    full_p = run_dir / "renders" / f"{idx:05d}.png"
    abl_p = run_dir / "renders_ablated_specular" / f"{idx:05d}.png"
    if not (full_p.exists() and abl_p.exists()):
        return None, None, None
    full = load_rgb(full_p)
    abl = load_rgb(abl_p)
    diff = np.abs(full - abl).mean(axis=-1)
    return diff, full, abl


def compute_iou(oracle_diff, sparse_diff, mask, top_pct=TOP_PCT, min_std=1e-4):
    """Returns None (rather than a misleading number) when the oracle diff is
    near-constant in the foreground -- quantile thresholding on a flat array
    ties everything at the threshold, making o_bin swallow the whole
    foreground and IoU collapse to exactly top_pct regardless of the sparse
    map. That's what was happening for every teapot row."""
    fg = mask > 0.5
    if fg.sum() == 0:
        return None
    o_vals = oracle_diff[fg]
    s_vals = sparse_diff[fg]
    if o_vals.std() < min_std:
        return None
    o_thresh = np.quantile(o_vals, 1 - top_pct)
    s_thresh = np.quantile(s_vals, 1 - top_pct)
    o_bin = (oracle_diff >= o_thresh) & fg
    s_bin = (sparse_diff >= s_thresh) & fg
    inter = np.logical_and(o_bin, s_bin).sum()
    union = np.logical_or(o_bin, s_bin).sum()
    return inter / union if union > 0 else None


def weighted_psnr(pred_rgb, gt_rgb, weight, mask):
    w = (weight * mask)[..., None]
    sq_err = (pred_rgb - gt_rgb) ** 2
    weighted_mse = (sq_err * w).sum() / (w.sum() * 3 + 1e-8)
    if weighted_mse <= 0:
        return None
    return 20 * np.log10(1.0 / np.sqrt(weighted_mse))


PATH_RE = re.compile(
    r"eval/(?P<dataset>[^/]+)/(?P<scene>[^/]+)/(?P<n_views>\d+)_views/"
    r"feat2gs-(?P<model>[^/]+)/(?P<pointmap>[^/]+)/(?P<feature>[^/]+)/?$"
)


def find_sparse_runs(root, models=("S", "T")):
    runs = []
    for p in root.glob("*/*/*_views/feat2gs-*/*/*"):
        m = PATH_RE.search(str(p) + "/")
        if not m:
            continue
        d = m.groupdict()
        if d["model"] not in models:
            continue
        test_dir = p / "test"
        if not test_dir.exists():
            continue
        for iter_dir in test_dir.glob("ours_*"):
            if (iter_dir / "renders").exists():
                d["run_dir"] = iter_dir
                d["n_views"] = int(d["n_views"])
                runs.append(d)
    return runs


def aggregate_by_model_feature(rows):
    """Averages IoU and weighted_delta_psnr across scenes, separately per
    (model, feature). Reports iou_n_scenes separately from delta_n_scenes,
    since a degenerate-oracle scene (like teapot currently) contributes to
    the delta average but gets excluded from the IoU average."""
    from collections import defaultdict
    grouped = defaultdict(lambda: {"iou": [], "delta": [], "scenes_delta": set(), "scenes_iou": set()})

    for r in rows:
        key = (r["model"], r["feature"])
        if r.get("mean_iou") is not None:
            grouped[key]["iou"].append(r["mean_iou"])
            grouped[key]["scenes_iou"].add(r["scene"])
        if r.get("weighted_delta_psnr") is not None:
            grouped[key]["delta"].append(r["weighted_delta_psnr"])
            grouped[key]["scenes_delta"].add(r["scene"])

    out = []
    for (model, feature), v in grouped.items():
        out.append(dict(
            model=model, feature=feature,
            mean_iou=float(np.mean(v["iou"])) if v["iou"] else None,
            iou_n_scenes=len(v["scenes_iou"]),
            mean_weighted_delta_psnr=float(np.mean(v["delta"])) if v["delta"] else None,
            delta_n_scenes=len(v["scenes_delta"]),
        ))
    return out


def main():
    runs = find_sparse_runs(OUTPUT_ROOT)
    print(f"Found {len(runs)} S/T-mode runs")

    rows = []
    for run in runs:
        scene = run["scene"]
        idx_to_rN = get_idx_to_rN(scene)
        if not idx_to_rN:
            print(f"  [skip] no staging files found for scene {scene}")
            continue

        run_dir = run["run_dir"]
        mask_dir = run_dir / "masks"

        ious, wpsnr_full, wpsnr_abl = [], [], []
        for idx, r_name in idx_to_rN.items():
            sparse_diff, sparse_full, sparse_abl = compute_sparse_diff(run_dir, idx)
            if sparse_diff is None:
                continue

            mask_path = mask_dir / f"{idx:05d}.png"
            if not mask_path.exists():
                continue
            mask = load_gray(mask_path, size=(sparse_diff.shape[1], sparse_diff.shape[0]))

            oracle_diff = load_oracle_diff(scene, r_name, target_size=(sparse_diff.shape[1], sparse_diff.shape[0]))
            if oracle_diff is None:
                continue

            iou = compute_iou(oracle_diff, sparse_diff, mask)
            if iou is not None:
                ious.append(iou)

            gt_path = run_dir / "gt" / f"{idx:05d}.png"
            if gt_path.exists():
                gt = load_rgb(gt_path, size=(sparse_diff.shape[1], sparse_diff.shape[0]))
                fp = weighted_psnr(sparse_full, gt, oracle_diff, mask)
                ap = weighted_psnr(sparse_abl, gt, oracle_diff, mask)
                if fp is not None:
                    wpsnr_full.append(fp)
                if ap is not None:
                    wpsnr_abl.append(ap)

        if not ious and not wpsnr_full:
            continue

        row = dict(scene=scene, n_views=run["n_views"], model=run["model"], feature=run["feature"])
        row["mean_iou"] = float(np.mean(ious)) if ious else None
        row["n_views_matched"] = len(ious)
        if wpsnr_full and wpsnr_abl:
            row["weighted_full_psnr"] = float(np.mean(wpsnr_full))
            row["weighted_ablated_psnr"] = float(np.mean(wpsnr_abl))
            row["weighted_delta_psnr"] = row["weighted_full_psnr"] - row["weighted_ablated_psnr"]
        rows.append(row)

    out_dir = Path.home() / "feat2gs_analysis"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "specular_mask_analysis.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {len(rows)} rows -> {out_dir / 'specular_mask_analysis.json'}")

    for r in sorted(rows, key=lambda r: -(r.get("weighted_delta_psnr") or -999)):
        iou_str = f"{r['mean_iou']:.3f}" if r.get("mean_iou") is not None else "  N/A"
        print(f"  {r['scene']:10s} {r['model']:3s} {r['feature']:12s} "
              f"IoU={iou_str}  weighted_delta_psnr={r.get('weighted_delta_psnr', float('nan')):.3f}  "
              f"(n={r['n_views_matched']})")

    agg = aggregate_by_model_feature(rows)
    agg_sorted = sorted(agg, key=lambda a: (a["model"], -(a["mean_weighted_delta_psnr"] or -999)))

    with open(out_dir / "specular_mask_analysis_by_model.json", "w") as f:
        json.dump(agg_sorted, f, indent=2)

    print(f"\n=== Averaged across scenes, per model x feature ===")
    for model in sorted(set(a["model"] for a in agg_sorted)):
        print(f"\n--- model={model} ---")
        for a in [x for x in agg_sorted if x["model"] == model]:
            iou_str = f"{a['mean_iou']:.3f} (n={a['iou_n_scenes']})" if a["mean_iou"] is not None else "N/A"
            delta_str = f"{a['mean_weighted_delta_psnr']:.3f} (n={a['delta_n_scenes']})" if a["mean_weighted_delta_psnr"] is not None else "N/A"
            print(f"  {a['feature']:12s} IoU={iou_str:20s} weighted_delta_psnr={delta_str}")

    print(f"\nWrote per-model averages -> {out_dir / 'specular_mask_analysis_by_model.json'}")


if __name__ == "__main__":
    main()
