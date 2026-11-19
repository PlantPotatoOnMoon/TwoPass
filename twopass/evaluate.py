"""Evaluate a TwoPass checkpoint on the Bridge or Language Table test split.

Every test episode is cut into non-overlapping 16-frame windows. For each
window the model receives the first frame and the recorded actions and
generates a 17-frame clip with 4 or 6 function evaluations. The first 16
frames are scored against the ground truth: PSNR, SSIM and LPIPS on the 15
future frames, plus I3D features of the 16-frame clip for FVD. Windows can be
split over several GPUs with --shard/--nshards; twopass/aggregate.py then
reports the test-set metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_2 import Wan2_2_VAE
from twopass.data import ACTION_SCALE
from twopass.model import TwoPassWanModel
from twopass.sampler import generate
from twopass.windows import read_frames, window_actions

SCORED_FRAMES = 16
I3D_KWARGS = dict(rescale=True, resize=True, return_features=True)


def list_test_windows(data_root, dataset):
    """All non-overlapping 16-frame windows of the test split, in a fixed order."""
    data_root = Path(data_root)
    rows = []
    for path in sorted((data_root / 'annotation/test').glob('*.json')):
        annotation = json.loads(path.read_text())
        episode = str(annotation['episode_id'])
        if dataset == 'bridge':
            length = len(annotation['state'])
            video = annotation['videos'][0]['video_path']
            text = (annotation.get('texts') or [''])[0]
        else:
            length = len(annotation['actions'])
            video = annotation['video_path']
            text = ''
        for start in range(0, length - 15, SCORED_FRAMES):
            window_id = f'{episode}_0_{start}' if dataset == 'bridge' else f'lt_{episode}_{start}'
            rows.append(dict(id=window_id, episode_id=episode, start=start, annotation=str(path),
                             video=str(data_root / video), text=text))
    return rows


def config_settings(config):
    """The configuration entries that determine the result of a window."""
    return dict(dataset=config['dataset'], seed=config['eval_seed'], shift=config['shift'],
                **{key: config[key] for key in ('n_frames', 'height', 'width', 'action_dim', 'adapter_after')},
                scored_frames=SCORED_FRAMES)


def window_seed(window_id, seed):
    digest = hashlib.sha256(f'{int(seed)}:{window_id}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % (2**63)


def pixel_metrics(pred, gt):
    """Per-frame PSNR and SSIM on grayscale uint8 frames."""
    pred_gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in pred]
    gt_gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in gt]
    psnr = np.array([peak_signal_noise_ratio(y, x, data_range=255) for x, y in zip(pred_gray, gt_gray)])
    ssim = np.array([structural_similarity(y, x, data_range=255) for x, y in zip(pred_gray, gt_gray)])
    return psnr, ssim


def main():
    parser = argparse.ArgumentParser(description='TwoPass test-set evaluation')
    parser.add_argument('--config', required=True, help='JSON file of the dataset (configs/)')
    parser.add_argument('--checkpoint', required=True, help='trained model, stepXXXXXXX.pt')
    parser.add_argument('--wan_dir', required=True, help='directory with the Wan2.2-TI2V-5B weights')
    parser.add_argument('--data_root', required=True, help='dataset directory containing annotation/test')
    parser.add_argument('--i3d_path', required=True, help='TorchScript I3D detector used for FVD')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--nfe', type=int, choices=[4, 6], default=4)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshards', type=int, default=1)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    dataset, n_frames, height, width = (config[key] for key in ('dataset', 'n_frames', 'height', 'width'))
    out_dir = Path(args.out_dir)
    rows = list_test_windows(args.data_root, dataset)[args.shard::args.nshards]
    # A window is finished only if its record was written with exactly these settings;
    # otherwise it is generated again and its outputs are overwritten.
    settings = dict(config_settings(config), checkpoint=str(Path(args.checkpoint).resolve()), nfe=args.nfe)
    records = [out_dir / 'windows' / row['id'] / 'record.json' for row in rows]
    pending = [row for row, record in zip(rows, records)
               if not (record.exists() and json.loads(record.read_text()).get('settings') == settings)]
    if not pending:
        return

    device = 'cuda'
    torch.set_num_threads(2)
    wan_dir = Path(args.wan_dir)
    vae = Wan2_2_VAE(vae_pth=str(wan_dir / 'Wan2.2_VAE.pth'), device=device)
    text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=device,
                                  checkpoint_path=str(wan_dir / 'models_t5_umt5-xxl-enc-bf16.pth'),
                                  tokenizer_path=str(wan_dir / 'google/umt5-xxl'), shard_fn=None)
    model = TwoPassWanModel.from_pretrained_wan(
        str(wan_dir), action_dim=config['action_dim'], adapter_after=config['adapter_after'])
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    del payload
    model.to(device).eval().requires_grad_(False)
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    i3d = torch.jit.load(args.i3d_path, map_location=device).eval()
    i3d_warm = False

    for done, row in enumerate(pending, 1):
        folder = out_dir / 'windows' / row['id']
        folder.mkdir(parents=True, exist_ok=True)
        gt = read_frames(row['video'], row['start'], SCORED_FRAMES, height, width)
        annotation = json.loads(Path(row['annotation']).read_text())
        raw_actions = window_actions(annotation, dataset, row['start'], n_frames)

        with torch.inference_mode():
            context = text_encoder([row['text']], device)
            first = torch.from_numpy(gt[0].copy()).permute(2, 0, 1).float().to(device) / 127.5 - 1.
            actions = (torch.from_numpy(raw_actions.copy()).to(device) * ACTION_SCALE.to(device)).unsqueeze(0)
            torch.manual_seed(window_seed(row['id'], config['eval_seed']))
            video = generate(model, vae, context, first, actions, n_frames, height, width, device,
                             nfe=args.nfe, shift=config['shift'])
            prediction = ((video.permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
        np.savez_compressed(folder / 'prediction.npz', frames=prediction)

        # The observed first frame is shared; pixel metrics cover the 15 future frames.
        scored = prediction[:SCORED_FRAMES].copy()
        scored[0] = gt[0]
        psnr, ssim = pixel_metrics(scored[1:], gt[1:])
        with torch.inference_mode():
            pair = torch.from_numpy(np.stack([scored, gt])).to(device).permute(0, 1, 4, 2, 3).float()
            clips = pair.permute(0, 2, 1, 3, 4).contiguous()
            if not i3d_warm:
                # One warm-up call, so every scored window uses the same TorchScript execution plan.
                i3d(clips, **I3D_KWARGS)
                i3d_warm = True
            features = i3d(clips, **I3D_KWARGS).float().cpu().numpy()
            perceptual = lpips_model(pair[0, 1:] / 127.5 - 1, pair[1, 1:] / 127.5 - 1).flatten().cpu().numpy()
        np.savez_compressed(folder / 'features.npz', pred=features[0], gt=features[1])
        record = dict(id=row['id'], episode_id=row['episode_id'], settings=settings, psnr=float(psnr.mean()),
                      ssim=float(ssim.mean()), lpips=float(perceptual.mean()))
        (folder / 'record.json').write_text(json.dumps(record, indent=2))
        print('DONE', dataset, args.shard, done, len(pending), json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
