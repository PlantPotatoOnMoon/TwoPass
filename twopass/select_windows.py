"""Select the training windows of a dataset and write the training manifest.

Bridge: every training episode with valid actions contributes one window with a
uniformly random start; the remaining windows are a second window from as many
episodes as needed, placed as far as possible from the first one.

Language Table: the training episodes are shuffled and each of the first
`train_windows` valid episodes contributes one window with a random start.

  python -m twopass.select_windows --config configs/bridge.json \
      --data_root /path/to/bridge --out_dir /path/to/train_data/bridge
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def inspect_bridge_episode(path, data_root, n_frames):
    """Episode description, or None when its actions or its video cannot supply a window."""
    annotation = json.loads(path.read_text())
    actions = np.asarray(annotation.get('action', []), np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < n_frames - 1 or not np.isfinite(actions).all():
        return None
    video = data_root / annotation['videos'][0]['video_path']
    capture = cv2.VideoCapture(str(video))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if frames < n_frames:
        return None
    return dict(episode_id=str(annotation['episode_id']), annotation=str(path), video=str(video),
                text=(annotation.get('texts') or [''])[0],
                max_start=min(frames - n_frames, len(actions) - (n_frames - 1)))


def select_bridge_windows(data_root, n_frames, count, seed):
    paths = sorted((data_root / 'annotation/train').glob('*.json'))
    with ThreadPoolExecutor(8) as pool:
        episodes = [episode for episode in pool.map(lambda p: inspect_bridge_episode(p, data_root, n_frames), paths)
                    if episode is not None]
    assert sum(episode['max_start'] + 1 for episode in episodes) >= count
    rng = random.Random(seed)
    rng.shuffle(episodes)
    used = [set() for _ in episodes]
    rows = []
    # Round robin over the episodes: a random start first, then the unused
    # start farthest from the ones already taken.
    while len(rows) < count:
        for episode, starts in zip(episodes, used):
            maximum = episode['max_start']
            if len(starts) == maximum + 1:
                continue
            if not starts:
                start = rng.randrange(maximum + 1)
            else:
                candidates = [x for x in range(maximum + 1) if x not in starts]
                rng.shuffle(candidates)
                start = max(candidates, key=lambda x: min(abs(x - y) for y in starts))
            starts.add(start)
            rows.append(dict(id=f"{episode['episode_id']}_0_{start}", episode_id=episode['episode_id'],
                             annotation=episode['annotation'], video=episode['video'], text=episode['text'],
                             start=start))
            if len(rows) == count:
                break
    return rows


def select_languagetable_windows(data_root, n_frames, count, seed):
    paths = sorted((data_root / 'annotation/train').glob('*.json'))
    random.Random(seed).shuffle(paths)
    rows = []
    for path in paths:
        annotation = json.loads(path.read_text())
        actions = np.asarray(annotation['actions'], np.float32)
        if len(actions) < n_frames - 1 or not np.isfinite(actions).all():
            continue
        start = random.Random(seed + int(path.stem)).randint(0, len(actions) - (n_frames - 1))
        episode = str(annotation['episode_id'])
        rows.append(dict(id=f'lt_{episode}_{start}', episode_id=episode, annotation=str(path),
                         video=str(data_root / annotation['video_path']), text='', start=start))
        if len(rows) == count:
            break
    return rows


def main():
    parser = argparse.ArgumentParser(description='Select the TwoPass training windows')
    parser.add_argument('--config', required=True, help='JSON file of the dataset (configs/)')
    parser.add_argument('--data_root', required=True, help='dataset directory containing annotation/train')
    parser.add_argument('--out_dir', required=True, help='receives train_manifest.json; caches go to cache/')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    select = select_bridge_windows if config['dataset'] == 'bridge' else select_languagetable_windows
    rows = select(Path(args.data_root), config['n_frames'], config['train_windows'], config['selection_seed'])
    out_dir = Path(args.out_dir).resolve()
    for row in rows:
        row['cache_npz'] = str(out_dir / 'cache' / f"{row['id']}.npz")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train_manifest.json').write_text(json.dumps(dict(dataset=config['dataset'], rows=rows), indent=2))
    print(f"{len(rows)} windows from {len({row['episode_id'] for row in rows})} episodes")


if __name__ == '__main__':
    main()
