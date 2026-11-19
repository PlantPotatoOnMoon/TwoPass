"""Encode the selected training windows into the cached model inputs.

For every window of the manifest: read 17 consecutive frames, encode the clip
and, separately, its first frame with the Wan2.2 VAE, encode the task text with
the umT5 text encoder, and store them with the raw actions in one `.npz` file.

  python -m twopass.cache_windows --config configs/bridge.json \
      --manifest /path/to/train_data/bridge/train_manifest.json --wan_dir /path/to/Wan2.2-TI2V-5B
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_2 import Wan2_2_VAE
from twopass.windows import read_frames, window_actions


def main():
    parser = argparse.ArgumentParser(description='Cache the TwoPass training windows')
    parser.add_argument('--config', required=True, help='JSON file of the dataset (configs/)')
    parser.add_argument('--manifest', required=True, help='train_manifest.json written by twopass.select_windows')
    parser.add_argument('--wan_dir', required=True, help='directory with the Wan2.2-TI2V-5B weights')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshards', type=int, default=1)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    dataset, n_frames, height, width = (config[key] for key in ('dataset', 'n_frames', 'height', 'width'))
    rows = json.loads(Path(args.manifest).read_text())['rows'][args.shard::args.nshards]
    pending = [row for row in rows if not Path(row['cache_npz']).exists()]
    if not pending:
        return

    device = 'cuda'
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    wan_dir = Path(args.wan_dir)
    vae = Wan2_2_VAE(vae_pth=str(wan_dir / 'Wan2.2_VAE.pth'), device=device)
    text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=device,
                                  checkpoint_path=str(wan_dir / 'models_t5_umt5-xxl-enc-bf16.pth'),
                                  tokenizer_path=str(wan_dir / 'google/umt5-xxl'), shard_fn=None)
    contexts = {}
    for done, row in enumerate(pending, 1):
        frames = read_frames(row['video'], row['start'], n_frames, height, width)
        annotation = json.loads(Path(row['annotation']).read_text())
        actions = window_actions(annotation, dataset, row['start'], n_frames)
        clip = torch.from_numpy(frames).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
        with torch.no_grad():
            latents = vae.encode([clip])[0]
            boundary = vae.encode([clip[:, :1]])[0]
            if row['text'] not in contexts:
                contexts[row['text']] = text_encoder([row['text']], device)[0].float().cpu().numpy()
        path = Path(row['cache_npz'])
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix('.partial')
        with partial.open('wb') as f:
            np.savez(f, gt_latents=latents.cpu().numpy(), boundary_latent=boundary.cpu().numpy(),
                     context=contexts[row['text']], actions=actions)
        partial.replace(path)
        if done % 25 == 0 or done == len(pending):
            print('CACHED', args.shard, done, len(pending), row['id'], flush=True)


if __name__ == '__main__':
    main()
