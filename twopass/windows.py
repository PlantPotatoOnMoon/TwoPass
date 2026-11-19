"""Reading one window of an episode: video frames and the aligned actions."""
import cv2
import numpy as np


def read_frames(video, start, count, height, width):
    """Frames [count, H, W, 3] (uint8 RGB, bilinear resize) starting at frame `start`."""
    capture = cv2.VideoCapture(video)
    frames = []
    try:
        for index in range(start + count):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f'Missing frame {index}: {video}')
            if index >= start:
                frames.append(cv2.cvtColor(
                    cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(frames)


def window_actions(annotation, dataset, start, n_frames):
    """Raw actions [n_frames, 7]: a zero first row, then the transitions of the window.

    Language Table's planar actions fill the first two columns.
    """
    raw = np.asarray(annotation['action' if dataset == 'bridge' else 'actions'], dtype=np.float32)
    chunk = raw[start:start + n_frames - 1]
    actions = np.zeros((n_frames, 7), np.float32)
    actions[1:1 + len(chunk), :chunk.shape[1]] = chunk
    return actions
