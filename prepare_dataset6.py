import json
import re
from pathlib import Path
import numpy as np
from PIL import Image

SRC_ROOT = Path.home() / "datasets/refnerf"
STAGING_ROOT = Path.home() / "dataset/eval_tmp_data/refnerf"
SCENES = ["ball", "car", "coffee", "helmet", "teapot", "toaster"]
N_TRAIN_VIEWS = 5
N_TEST_VIEWS = 4


def composite_rgb(src_png: Path, dst_png: Path):
    img = Image.open(src_png).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    Image.alpha_composite(bg, img).convert("RGB").save(dst_png)


def load_cam_positions(transforms_json: Path):
    with open(transforms_json) as f:
        meta = json.load(f)
    positions = {}
    for frame in meta["frames"]:
        name = Path(frame["file_path"]).name
        mat = np.array(frame["transform_matrix"])
        positions[name] = mat[:3, 3]
    return positions


def azimuth_sorted(names, positions):
    """Sorts cameras by angular position around the hemisphere's dominant
    plane (found via PCA, so it works regardless of which axis is 'up'),
    instead of filename order (random) or max-spread FPS (breaks DUSt3R
    pairwise overlap when only a few views are picked)."""
    pts = np.stack([positions[nm] for nm in names])
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T  # coords in the plane of maximum angular spread
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    order = np.argsort(angles)
    return [names[i] for i in order]


def evenly_spaced(ordered_names, n):
    """Picks n uniformly-spaced entries from an already angle-ordered list --
    moderate spacing, guaranteed coverage, guaranteed overlap between
    consecutive picks, and already in a continuous walking order for free."""
    if n >= len(ordered_names) or n <= 0:
        return ordered_names
    if n == 1:
        return [ordered_names[0]]
    idxs = [round(i * (len(ordered_names) - 1) / (n - 1)) for i in range(n)]
    return [ordered_names[i] for i in idxs]


def prep_scene(scene):
    train_src = SRC_ROOT / scene / "train"
    test_src = SRC_ROOT / scene / "test"
    scene_dst = STAGING_ROOT / scene
    train_dst = scene_dst / "train"
    train_dst.mkdir(parents=True, exist_ok=True)

    train_pos = load_cam_positions(SRC_ROOT / scene / "transforms_train.json")
    test_pos = load_cam_positions(SRC_ROOT / scene / "transforms_test.json")

    all_train_names = sorted(train_pos, key=lambda s: int(re.search(r"\d+", s).group()))
    all_test_names = sorted(test_pos, key=lambda s: int(re.search(r"\d+", s).group()))

    ## MAKE SURE stirde for train and test are co primes

    stride = 13
    start_idx = 0
    chosen_train = all_test_names[start_idx : start_idx + (N_TRAIN_VIEWS * stride) : stride]

    stride = 7
    start_idx = 1
    chosen_test = all_test_names[start_idx : start_idx + (N_TEST_VIEWS * stride) : stride]

    for i, nm in enumerate(chosen_train):
        composite_rgb(test_src / f"{nm}.png", train_dst / f"train_{i:02d}_{nm}.png")
    for i, nm in enumerate(chosen_test):
        composite_rgb(test_src / f"{nm}.png", scene_dst / f"test_{i:02d}_{nm}.png")

    print(f"[{scene}] train (angle-ordered): {chosen_train}")
    print(f"[{scene}] test  (index-ordered):  {chosen_test}")


if __name__ == "__main__":
    for s in SCENES:
        prep_scene(s)
