#!/usr/bin/env python3
"""
count_event_labels.py – Summarise segmentation-label statistics
for a folder of labelled event-batch tensors (*.pt).

Each .pt file is expected to contain an (N, 5) torch.tensor with
columns [x, y, t, polarity, label]:
  • label == 1 → propeller voxel (drone blade)
  • label == 0 → background voxel

Usage
-----
python count_event_labels.py /path/to/folder/with/pt/files
"""
from pathlib import Path
import argparse
import torch

def main():
    ap = argparse.ArgumentParser(
        description="Count propeller vs background events in labelled .pt files"
    )
    ap.add_argument("root", type=Path, help="Directory containing *.pt tensors")
    args = ap.parse_args()

    if not args.root.is_dir():
        ap.error(f"{args.root} is not a directory")

    pt_files = sorted(args.root.glob("*.pt"))
    if not pt_files:
        ap.error(f"No .pt files found in {args.root}")

    total_prop, total_back = 0, 0

    for p in pt_files:
        try:
            ev = torch.load(p, map_location="cpu")
            if ev.ndim != 2 or ev.shape[1] < 5:
                print(f"[warn] {p.name}: unexpected tensor shape {tuple(ev.shape)}, skipped")
                continue
            labels = ev[:, 4]
            n_prop = int((labels == 1).sum())
            n_back = labels.numel() - n_prop
            total_prop += n_prop
            total_back += n_back
        except Exception as e:
            print(f"[warn] Could not load {p.name}: {e}")

    total_events = total_prop + total_back
    if total_events == 0:
        print("No events counted – are the files empty?")
        return

    print(f"Scanned  {len(pt_files):>6d}  files in {args.root}")
    print(f"Total events:         {total_events:,}")
    print(f"  • Propeller events:  {total_prop:,}  ({100*total_prop/total_events:.2f} %)")
    print(f"  • Background events: {total_back:,}  ({100*total_back/total_events:.2f} %)")

if __name__ == "__main__":
    main()
