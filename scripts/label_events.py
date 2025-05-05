# scripts/label_events.py
"""
Interactive labeller for AEDAT-4 recordings.
Plays event batches, lets you mark one drone-propeller bounding box,
and exports a tensor  (x, y, t, polarity, label)  per accepted batch.
"""
from pathlib import Path
import argparse
import re
import cv2
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root
from src.io import stream_aedat
from src.vis import events_to_image

# ─────────────────────────── utils ──────────────────────────────
_ZFILERE = re.compile(r"^(\d{8})\.pt$")


def next_index(out_dir: Path) -> int:
    """Find the next free 8-digit index in *out_dir*."""
    nums = [
        int(m.group(1)) for m in (_ZFILERE.match(p.name) for p in out_dir.iterdir())
        if m
    ]
    return max(nums) + 1 if nums else 0


def label_one_batch(events: torch.Tensor, rect) -> torch.Tensor:
    """Return events with a new label column (0/1) given bbox (x, y, w, h)."""
    x, y, w, h = rect
    xs, ys = events[:, 0], events[:, 1]
    inside = (xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h)
    labels = inside.to(torch.uint8).unsqueeze(1)
    return torch.cat((events, labels), dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aedat", type=Path, help="input .aedat4 file")
    ap.add_argument("-o", "--out", type=Path, required=True,
                    help="directory to save labelled tensors")
    ap.add_argument("--batch_us", type=int, default=10_000,
                    help="target batch length (µs)")
    ap.add_argument("--min_us", type=int, default=9_000,
                    help="minimum accepted batch span (µs)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    idx = next_index(args.out)

    stream = stream_aedat(args.aedat, batch_us=args.batch_us)
    cv2.namedWindow("events", cv2.WINDOW_NORMAL)

    playing = True
    current = None          # (events, frame, hw)

    while True:
        if playing:
            try:
                events, hw = next(stream)
            except StopIteration:
                break

            span = int(events[:, 2].max() - events[:, 2].min())
            if span < args.min_us:
                continue

            frame = events_to_image(events, hw)
            current = (events, frame, hw)

            cv2.imshow("events", frame)
            key = cv2.waitKey(10) & 0xFF

            if key == ord(" "):       # space → pause
                playing = False
                continue              # redraw happens in next loop
        else:
            # paused – wait for ROI selection
            events, frame, hw = current
            rect = cv2.selectROI("events", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("ROI selector")          # return focus to main window

            if rect == (0, 0, 0, 0):
                playing = True
                continue

            # create colour-coded preview: background = white, propeller = green
            H, W = hw
            preview = np.zeros((H, W, 3), dtype=np.uint8)
            xs, ys = events[:, 0].numpy(), events[:, 1].numpy()
            preview[ys, xs] = (255, 255, 255)          # all events white

            x, y, w, h = rect
            inside = (xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h)
            preview[ys[inside], xs[inside]] = (0, 255, 0)  # propellers green
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.imshow("events", preview)
            cv2.waitKey(1)                             # force redraw

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in (13, 10, ord(" ")):          # Enter or space → accept
                    labelled_events = label_one_batch(events, rect)
                    out_path = args.out / f"{idx:08d}.pt"
                    torch.save(labelled_events, out_path)
                    print(f"saved {out_path.name}  (N={len(events)})")
                    idx += 1
                    playing = True
                    break
                if key in (ord("c"),):                 # c → cancel / discard
                    playing = True
                    break
                if key in (ord("q"), 27):              # q or ESC → quit
                    return

        # immediate quit hot-keys while playing
        if key in (ord("q"), 27):
            break

    print("Finished.")


if __name__ == "__main__":
    main()
