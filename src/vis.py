"""
Tiny visualisation helpers for debugging.
"""
from typing import Tuple

import cv2
import numpy as np
import torch


def events_to_image(
    events: torch.Tensor,
    hw: Tuple[int, int],
    *,
    on_color=(0, 255, 0),
    off_color=(0, 0, 255),
) -> np.ndarray:
    """
    Quick & dirty rasteriser: draws every event as one coloured pixel.
    """
    H, W = hw
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # vectorised indexing
    xs, ys, _, pol = events.T
    xs = xs.numpy()
    ys = ys.numpy()
    pol = events[:, 3].numpy().astype(bool)

    frame[ys[pol], xs[pol]] = on_color
    frame[ys[~pol], xs[~pol]] = off_color
    return frame
