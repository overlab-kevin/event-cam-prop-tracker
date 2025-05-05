#!/usr/bin/env python
"""
run_aedat.py – stream an *.aedat4* recording through a trained sparse‑UNet and
visualise a confidence heat‑map next to the raw polarity view. Optionally
record the two views side‑by‑side into an MP4.

Reduction modes
---------------
Per‑pixel confidence when more than one event lands on the same pixel within
the 10 ms window can be collapsed in three ways:
  • last (default) – newest event overwrites older ones (fastest)
  • max            – keep the highest confidence
  • mean           – average all confidences (smoothest)

Usage
-----
python scripts/run_aedat.py RUN_DIR RECORDING.aedat4 [--batch_us 10000] \
                                 [--reduce {last,max,mean}] \
                                 [--record out.mp4]

Arguments
---------
RUN_DIR            Folder with *latest.pt* and *config.yaml*.
RECORDING.aedat4   Path to an AEDAT‑4 recording.

Optional arguments
------------------
--batch_us         Temporal window per inference (default: 10 000 µs).
--reduce           How to collapse multiple confidences per pixel.
--record out.mp4   Save an MP4 (H.264) at 1e6 / batch_us FPS with the events
                   view on the left and the heat‑map on the right.

Hot‑keys during playback
------------------------
␣ (space)          Pause / resume.
q  or  ESC          Quit.

The visualiser colours **only the event pixels**; background remains black.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch

# repo imports -------------------------------------------------------------
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root
from config.config import Cfg
from src.model import SpMiniUNetWrapper, combine_voxel_grids_to_sparse_tensor
from src.io import stream_aedat
from src.vis import events_to_image


# -------------------------------------------------------------------------
@torch.no_grad()
def voxelise_events(events: torch.Tensor, bin_us: int):
    """Convert (x,y,t,pol) events → coords, feats ready for the network."""
    xs, ys, ts, ps = events.T
    z = ((ts - ts.min()) // bin_us).int()
    coords = torch.stack((z, ys, xs), 1).cpu().numpy().astype(np.int32)
    feats  = np.stack((ps.numpy(), 1 - ps.numpy()), 1).astype(np.float32)
    return coords, feats


# -------------------------------------------------------------------------
@torch.no_grad()
def colourise_events(coords: np.ndarray, conf: np.ndarray, hw, mode="last"):
    """Return an RGB frame where only event pixels are coloured.

    Parameters
    ----------
    coords : (N,3) int32   voxel coords (z,y,x)
    conf   : (N,) float    sigmoid confidences 0‑1 per event voxel
    hw     : (H,W) int     sensor resolution
    mode   : str           'last' | 'max' | 'mean'
    """
    H, W = hw
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    ys, xs = coords[:, 1], coords[:, 2]

    if mode == "last":
        gray = (conf * 255).clip(0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(gray, cv2.COLORMAP_JET)[:, 0, :]
        frame[ys, xs] = colors
        return frame

    # allocate helper images once per call (could reuse across frames)
    if mode == "max":
        acc = np.zeros((H, W), dtype=np.float32)
        np.maximum.at(acc, (ys, xs), conf)
        mask = acc > 0
        if not mask.any():
            return frame  # nothing to draw
        gray = (acc[mask] * 255).clip(0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(gray, cv2.COLORMAP_JET)[:, 0, :]
        frame[mask] = colors
        return frame

    if mode == "mean":
        sum_img = np.zeros((H, W), dtype=np.float32)
        cnt_img = np.zeros((H, W), dtype=np.uint16)
        np.add.at(sum_img, (ys, xs), conf)
        np.add.at(cnt_img, (ys, xs), 1)
        mask = cnt_img > 0
        if not mask.any():
            return frame
        mean = sum_img[mask] / cnt_img[mask]
        gray = (mean * 255).clip(0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(gray, cv2.COLORMAP_JET)[:, 0, :]
        frame[mask] = colors
        return frame

    raise ValueError(f"Unknown reduction mode: {mode}")


# -------------------------------------------------------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="directory with latest.pt + config.yaml")
    ap.add_argument("aedat",   type=Path, help="input *.aedat4 recording")
    ap.add_argument("--batch_us", type=int, default=10_000, help="temporal window (µs)")
    ap.add_argument("--reduce", choices=["last", "max", "mean"], default="max",
                    help="confidence reduction per pixel")
    ap.add_argument("--record", type=Path, help="output MP4 path (optional)")
    args = ap.parse_args()

    # ---------- load network ------------------------------------------------
    cfg = Cfg.load(args.run_dir / "config.yaml")
    bin_us = getattr(cfg.dataset, "time_bin_us", 400)  # fallback to constant
    model = SpMiniUNetWrapper(cfg).cuda().eval()

    ckpt = torch.load(args.run_dir / "latest.pt", map_location="cuda")
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"Weights loaded from {args.run_dir/'latest.pt'}  (epoch {ckpt['epoch']})")

    # ---------- windows -----------------------------------------------------
    cv2.namedWindow("events", cv2.WINDOW_NORMAL)
    cv2.namedWindow("heat",   cv2.WINDOW_NORMAL)

    # ---------- optional recorder ------------------------------------------
    writer = None
    fps = int(1_000_000 / args.batch_us) if args.record else None

    # ---------- main loop ---------------------------------------------------
    stream = stream_aedat(args.aedat, batch_us=args.batch_us)
    paused = False
    last_events = last_heat = None

    while True:
        if not paused:
            try:
                events, hw = next(stream)
            except StopIteration:
                break

            if len(events) == 0:
                continue

            # 1) raw polarity view ------------------------------------------
            last_events = events_to_image(events, hw)

            # 2) network inference ------------------------------------------
            coords, feats = voxelise_events(events, bin_us)
            batch = [{"voxel_grids": [coords], "voxel_feats": [feats]}]
            feats_out = model.forward_batch(batch)[0][0]  # (M, out_ch)
            logits    = feats_out[:, 0]
            conf      = torch.sigmoid(logits).cpu().numpy()  # (M,)

            # 3) render sparse heat‑map --------------------------------------
            last_heat = colourise_events(coords, conf, hw, mode=args.reduce)

            # 4) recording ----------------------------------------------------
            if args.record and last_events is not None and last_heat is not None:
                if writer is None:
                    H, W, _ = last_events.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(args.record), fourcc, fps, (W*2, H))
                    if not writer.isOpened():
                        print("[warn] Could not open VideoWriter. Disabling recording.")
                        writer = None
                if writer is not None:
                    side_by_side = np.hstack((last_events, last_heat))
                    writer.write(side_by_side)

        # ---------- show current frames -------------------------------------
        if last_events is not None:
            cv2.imshow("events", last_events)
        if last_heat is not None:
            cv2.imshow("heat", last_heat)

        # ---------- handle key presses --------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    # ---------- clean‑up ----------------------------------------------------
    if writer is not None:
        writer.release()
    print("Finished.")


if __name__ == "__main__":
    main()
