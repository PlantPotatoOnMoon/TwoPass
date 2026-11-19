"""Test-set PSNR / SSIM / LPIPS / FVD from the per-window results of twopass/evaluate.py."""
import argparse
import json
from pathlib import Path

import numpy as np

from twopass.evaluate import config_settings, list_test_windows


def frechet_distance(gt, pred):
    """Frechet distance between Gaussians fitted to two feature sets [N, D]."""
    a = gt - gt.mean(0)
    b = pred - pred.mean(0)
    return float(np.square(gt.mean(0) - pred.mean(0)).sum()
                 + (np.square(a).sum() + np.square(b).sum()
                    - 2 * np.linalg.svd(a @ b.T, compute_uv=False).sum()) / (len(gt) - 1))


def main():
    parser = argparse.ArgumentParser(description='Aggregate TwoPass evaluation results')
    parser.add_argument('--config', required=True, help='JSON file of the dataset (configs/)')
    parser.add_argument('--data_root', required=True, help='dataset directory containing annotation/test')
    parser.add_argument('--eval_dir', required=True, help='--out_dir of twopass/evaluate.py')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    eval_dir = Path(args.eval_dir)
    rows = list_test_windows(args.data_root, config['dataset'])
    paths = [eval_dir / 'windows' / row['id'] / 'record.json' for row in rows]
    missing = [row['id'] for row, path in zip(rows, paths) if not path.exists()]
    records = [json.loads(path.read_text()) for path in paths if path.exists()]

    # All windows must be finished, with one common set of settings that agrees with --config.
    groups = {}
    for record in records:
        groups.setdefault(json.dumps(record.get('settings'), sort_keys=True), []).append(record['id'])
    problems = []
    if missing:
        problems.append(f'{len(missing)} windows have no finished record: {missing}')
    if len(groups) > 1:
        problems.append('the records were written with different settings:\n' + '\n'.join(
            f'  {len(ids)} windows with {settings}: {ids}' for settings, ids in groups.items()))
    for settings in groups:
        settings = json.loads(settings)
        if settings is None or any(settings.get(key) != value for key, value in config_settings(config).items()):
            problems.append(f'records with settings {settings} do not match {args.config}')
    if problems:
        raise SystemExit(f'Cannot aggregate {eval_dir}:\n' + '\n'.join(problems))

    gt_features, pred_features = [], []
    for path in paths:
        with np.load(path.with_name('features.npz')) as features:
            gt_features.append(features['gt'].copy())
            pred_features.append(features['pred'].copy())
    summary = dict(settings=records[0]['settings'], windows=len(records),
                   episodes=len({record['episode_id'] for record in records}),
                   **{key: float(np.mean([record[key] for record in records])) for key in ('psnr', 'ssim', 'lpips')},
                   fvd=frechet_distance(np.array(gt_features), np.array(pred_features)))
    (eval_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
