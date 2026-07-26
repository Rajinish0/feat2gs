import os
import re
import hashlib
import wandb

WANDB_PROJECT = "feat2gs-probes"
WANDB_ENTITY = "3dvis"
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".wandb_token")


def _login():
    if os.environ.get("WANDB_API_KEY"):
        return
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            token = f.read().strip()
        if token:
            wandb.login(key=token, relogin=False)


def run_id_from_model_path(model_path: str) -> str:
    """Deterministic id from model_path -> every script touching the same
    experiment cell (train + all probes) resumes the SAME run instead of
    spawning duplicates."""
    clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_path.strip('/'))
    if len(clean) > 120:
        h = hashlib.sha1(model_path.encode()).hexdigest()[:10]
        clean = clean[:100] + "_" + h
    return clean


def parse_experiment_tags(model_path: str) -> dict:
    """Parses .../eval/<dataset>/<scene>/<n_views>_views/feat2gs-<model>/<method>/<feature>/"""
    parts = [p for p in model_path.strip('/').split('/') if p]
    tags = {}
    try:
        idx = parts.index('eval')
        tags['dataset'] = parts[idx + 1]
        tags['scene'] = parts[idx + 2]
        tags['n_views'] = parts[idx + 3].replace('_views', '')
        tags['model'] = parts[idx + 4].replace('feat2gs-', '')
        tags['pointmap'] = parts[idx + 5]
        tags['feature'] = parts[idx + 6]
    except (ValueError, IndexError):
        pass
    return tags


def init_probe_run(model_path: str, job_type: str, extra_config: dict = None):
    _login()
    tags = parse_experiment_tags(model_path)
    run_id = run_id_from_model_path(model_path)
    group = f"{tags.get('dataset','?')}_{tags.get('scene','?')}_{tags.get('n_views','?')}views"
    name = f"{tags.get('scene', '?')}_{tags.get('model','?')}_{tags.get('feature','?')}"

    config = dict(tags)
    if extra_config:
        config.update(extra_config)

    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        id=run_id,
        resume="allow",
        group=group,
        name=name,
        job_type=job_type,
        tags=[t for t in (tags.get('model'), tags.get('feature'), tags.get('pointmap')) if t],
        config=config,
    )
