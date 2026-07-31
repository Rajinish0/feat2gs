import re
import json
from pathlib import Path
import pandas as pd

OUTPUT_ROOT = Path("/home/stud124/output/Feat2gs/output/eval")

PATH_RE = re.compile(
    r"eval/(?P<dataset>[^/]+)/(?P<scene>[^/]+)/(?P<n_views>\d+)_views/"
    r"feat2gs-(?P<model>[^/]+)/(?P<pointmap>[^/]+)/(?P<feature>[^/]+)/"
    r"test/ours_(?P<iteration>\d+)/specular_probe_results\.json$"
)


def parse_run_path(json_path: Path):
    m = PATH_RE.search(str(json_path))
    if not m:
        return None
    d = m.groupdict()
    d["n_views"] = int(d["n_views"])
    d["iteration"] = int(d["iteration"])
    return d


def load_all_runs(root: Path):
    per_frame_rows = []
    summary_rows = []
    json_files = list(root.glob("**/specular_probe_results.json"))
    print(f"Found {len(json_files)} specular_probe_results.json files")

    for jf in json_files:
        meta = parse_run_path(jf)
        if meta is None:
            print(f"  [skip] couldn't parse path: {jf}")
            continue
        try:
            with open(jf) as f:
                frames = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] couldn't read {jf}: {e}")
            continue
        if not frames:
            continue

        for fr in frames:
            row = dict(meta)
            row.update(fr)
            per_frame_rows.append(row)

        df_run = pd.DataFrame(frames)
        numeric_cols = df_run.select_dtypes(include="number").columns
        summary = df_run[numeric_cols].mean().to_dict()
        summary.update(meta)
        summary["n_frames"] = len(frames)
        summary_rows.append(summary)

    return pd.DataFrame(per_frame_rows), pd.DataFrame(summary_rows)


def add_recovery_ratios(summary_df, baseline_model="F"):
    if summary_df.empty:
        return summary_df
    group_cols = ["dataset", "scene", "n_views", "pointmap"]
    ratio_metrics = ["delta_psnr", "mean_specular_diff", "p99_specular_diff", "max_specular_diff"]

    baselines = (
        summary_df[summary_df["model"] == baseline_model]
        .groupby(group_cols)[ratio_metrics]
        .mean()
        .rename(columns={m: f"{m}_baseline" for m in ratio_metrics})
    )
    merged = summary_df.merge(baselines, on=group_cols, how="left")
    for m in ratio_metrics:
        merged[f"{m}_recovery_ratio"] = merged[m] / merged[f"{m}_baseline"].replace(0, pd.NA)
    return merged


def aggregate_by_feature(summary_df, metrics=None):
    """Averages each metric across scenes, per (model, feature).
    n_scenes/scenes columns let you see coverage at a glance -- a feature
    tested on 2 scenes isn't directly comparable to one tested on 6."""
    if metrics is None:
        metrics = ["delta_psnr", "delta_ssim", "delta_lpips",
                   "mean_specular_diff", "p99_specular_diff", "max_specular_diff"]
    metrics = [m for m in metrics if m in summary_df.columns]

    grouped = summary_df.groupby(["model", "feature"]).agg(
        n_scenes=("scene", "nunique"),
        scenes=("scene", lambda s: ", ".join(sorted(set(s)))),
        **{m: (m, "mean") for m in metrics},
    ).reset_index()

    return grouped.sort_values(["model", "delta_psnr"], ascending=[True, False])


def common_scene_ranking(summary_df, model, metrics=None):
    """Fair comparison: restricts to the intersection of scenes every feature
    in this mode was actually tested on, so no feature gets an advantage
    just from favorable scene coverage."""
    if metrics is None:
        metrics = ["delta_psnr", "delta_ssim", "delta_lpips",
                   "mean_specular_diff", "p99_specular_diff", "max_specular_diff"]
    metrics = [m for m in metrics if m in summary_df.columns]

    df = summary_df[summary_df["model"] == model]
    if df.empty:
        return pd.DataFrame()

    scene_sets = df.groupby("feature")["scene"].apply(set)
    common = set.intersection(*scene_sets.tolist()) if len(scene_sets) else set()

    if not common:
        print(f"  [{model}] no scene is shared by every feature yet -- skipping common-scene ranking")
        return pd.DataFrame()

    df_common = df[df["scene"].isin(common)]
    grouped = df_common.groupby("feature").agg(
        n_scenes=("scene", "nunique"),
        **{m: (m, "mean") for m in metrics},
    ).reset_index()
    grouped["common_scenes"] = ", ".join(sorted(common))
    return grouped.sort_values("delta_psnr", ascending=False)


def paper_reproduction_table(summary_df, models=("G", "T", "A", "F")):
    """Full (non-ablated) test-view PSNR/SSIM/LPIPS per model x feature,
    averaged over scenes. This is the same quantity metrics.py would report --
    the specular probe's 'full' render is a normal test-view render, nothing
    ablated -- so no separate metrics.py run is needed for this check."""
    df = summary_df[summary_df["model"].isin(models)]
    if df.empty:
        print("No G/T/A rows found yet.")
        return pd.DataFrame()

    grouped = df.groupby(["model", "feature"]).agg(
        n_scenes=("scene", "nunique"),
        scenes=("scene", lambda s: ", ".join(sorted(set(s)))),
        full_psnr=("full_psnr", "mean"),
        full_ssim=("full_ssim", "mean"),
        full_lpips=("full_lpips", "mean"),
    ).reset_index()

    return grouped.sort_values(["model", "full_psnr"], ascending=[True, False])


def paper_reproduction_by_model(summary_df, models=("G", "T", "A", "F")):
    """Collapses features too -- one row per model, averaged over every
    feature and scene available. This is the number that directly answers
    'does G beat T on held-out views', independent of which feature was used."""
    df = summary_df[summary_df["model"].isin(models)]
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("model").agg(
        n_rows=("feature", "count"),
        n_scenes=("scene", "nunique"),
        n_features=("feature", "nunique"),
        full_psnr=("full_psnr", "mean"),
        full_ssim=("full_ssim", "mean"),
        full_lpips=("full_lpips", "mean"),
    ).reset_index()

    return grouped.sort_values("full_psnr", ascending=False)


def main():
    per_frame_df, summary_df = load_all_runs(OUTPUT_ROOT)
    if summary_df.empty:
        print("No results found yet -- runs may still be in progress.")
        return

    summary_df = add_recovery_ratios(summary_df, baseline_model="F")

    id_cols = ["dataset", "scene", "n_views", "model", "pointmap", "feature", "n_frames"]
    metric_cols = [c for c in summary_df.columns if c not in id_cols]
    summary_df = summary_df[id_cols + sorted(metric_cols)]
    summary_df = summary_df.sort_values(["dataset", "scene", "n_views", "feature", "model"])
    per_frame_df = per_frame_df.sort_values(["dataset", "scene", "n_views", "feature", "model", "name"])

    out_dir = Path.home() / "feat2gs_analysis"
    out_dir.mkdir(exist_ok=True)

    summary_df.to_csv(out_dir / "specular_probe_summary.csv", index=False)
    per_frame_df.to_csv(out_dir / "specular_probe_per_frame.csv", index=False)
    print(f"\nWrote {len(summary_df)} summary rows -> {out_dir / 'specular_probe_summary.csv'}")
    print(f"Wrote {len(per_frame_df)} per-frame rows -> {out_dir / 'specular_probe_per_frame.csv'}")

    feature_ranking = aggregate_by_feature(summary_df)
    feature_ranking.to_csv(out_dir / "feature_ranking_raw.csv", index=False)
    print(f"\n=== Per-feature averages (raw -- watch n_scenes for coverage differences) ===")
    print(feature_ranking[["model", "feature", "n_scenes", "delta_psnr", "p99_specular_diff", "scenes"]].to_string(index=False))

    print(f"\n=== Common-scene rankings (fair comparison, per model) ===")
    common_frames = []
    for model in sorted(summary_df["model"].unique()):
        cr = common_scene_ranking(summary_df, model)
        if not cr.empty:
            cr.insert(0, "model", model)
            common_frames.append(cr)
            print(f"\n--- model={model} (common scenes: {cr['common_scenes'].iloc[0]}) ---")
            print(cr[["feature", "n_scenes", "delta_psnr", "p99_specular_diff"]].to_string(index=False))

    if common_frames:
        pd.concat(common_frames, ignore_index=True).to_csv(out_dir / "feature_ranking_common_scenes.csv", index=False)
        print(f"\nWrote common-scene rankings -> {out_dir / 'feature_ranking_common_scenes.csv'}")

    print(f"\n=== Paper reproduction: full test-view PSNR/SSIM/LPIPS, per model x feature ===")
    paper_table = paper_reproduction_table(summary_df)
    if not paper_table.empty:
        paper_table.to_csv(out_dir / "paper_reproduction_by_feature.csv", index=False)
        print(paper_table.to_string(index=False))

    print(f"\n=== Paper reproduction: G vs T vs A, collapsed across features/scenes ===")
    paper_by_model = paper_reproduction_by_model(summary_df)
    if not paper_by_model.empty:
        paper_by_model.to_csv(out_dir / "paper_reproduction_by_model.csv", index=False)
        print(paper_by_model.to_string(index=False))


if __name__ == "__main__":
    main()
