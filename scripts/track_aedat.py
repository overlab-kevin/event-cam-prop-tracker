#!/usr/bin/env python
"""
track_aedat.py – stream an *.aedat4* recording through a trained sparse‑UNet,
visualise a confidence heat‑map next to the raw polarity view, perform a
lightweight detector → tracker loop, AND evaluate performance against labels.

Workflow
========
1. **Detection mode** – process the *full frame* until the max‑logit crosses
   `--det_thresh`. If that happens, estimate the drone centre and switch to
   tracking.
2. **Tracking mode** – crop a user‑defined ROI (`--crop_w`, `--crop_h`) around
   the previous centre; voxelise only events in that window; run the network
   and localise again (centroid of voxels with `logit ≥ centre_thresh`). If
   localisation fails for `--max_misses` consecutive batches we fall back to
   detection mode.
3. **Evaluation** - If a corresponding '.txt' label file exists (same name as
   the .aedat4 file), load it and calculate accuracy and relative error
   metrics against the tracker's output whenever timestamps align.

A blue cross marks the current centre on the events view (if displayed). The script is fully
self‑contained (no imports from *run_aedat.py*).

Examples
--------
```bash
# vanilla – defaults match run_aedat.py, with display
python scripts/track_aedat.py runs/propdet flight.aedat4

# stricter detection threshold, 64×64 tracking window, save video (with display)
python scripts/track_aedat.py runs/propdet flight.aedat4 \
    --det_thresh 1.5 --crop_w 64 --crop_h 64 --record out.mp4

# Run without display windows, only evaluation
python scripts/track_aedat.py runs/propdet flight.aedat4 --no-display
```

Hot‑keys (only when display is enabled)
--------
␣ (space)  Pause / resume
q or ESC    Quit
"""
from pathlib import Path
import argparse
# Import cv2 conditionally later if needed for display or recording
import numpy as np
import torch
import sys
import math
import time
from typing import Dict, List, Tuple, Optional
import spconv.pytorch as spconv

# ─────────── repo imports (add repo root first) ──────────────────────────
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
from config.config import Cfg
from src.model import SpMiniUNetWrapper
from src.io import stream_aedat
# Import events_to_image conditionally if needed
# from src.vis import events_to_image

# --- Global flags and variables for conditional imports/operations ---
_DISPLAY_ENABLED = True
_RECORDING_ENABLED = False
_CV2_IMPORTED = False
_O3D_IMPORTED = False
_CM_IMPORTED = False
cv2 = None # Placeholder for potential cv2 import
o3d = None # Placeholder for potential open3d import
cm = None # Placeholder for potential matplotlib.cm import
events_to_image_func = None # Placeholder for events_to_image import

# --- Helper functions for conditional imports ---
def ensure_cv2_import():
    """Attempts to import cv2 if not already imported."""
    global _CV2_IMPORTED, cv2
    if not _CV2_IMPORTED:
        try:
            import cv2
            _CV2_IMPORTED = True
            print("Successfully imported OpenCV (cv2).")
        except ImportError:
            print("Warning: OpenCV (cv2) not found. Display and recording will be disabled.")
            _CV2_IMPORTED = False # Ensure we don't try again
    return _CV2_IMPORTED

def ensure_events_to_image_import():
    """Attempts to import events_to_image from src.vis."""
    global events_to_image_func
    if events_to_image_func is None:
        try:
            from src.vis import events_to_image
            events_to_image_func = events_to_image
            print("Successfully imported events_to_image.")
        except ImportError:
             print("Warning: Could not import events_to_image from src.vis. Visualization/Recording might fail.")
             # events_to_image_func remains None
    return events_to_image_func is not None


def ensure_o3d_import():
    """Attempts to import open3d if not already imported."""
    global _O3D_IMPORTED, o3d
    if not _O3D_IMPORTED and o3d is None: # Check o3d too in case import failed before
        try:
            import open3d as o3d_lib
            o3d = o3d_lib # Assign to global placeholder
            _O3D_IMPORTED = True
            print("Successfully imported Open3D.")
        except ImportError:
            print("Warning: Open3D not found. 3D visualization will be skipped.")
            _O3D_IMPORTED = False # Ensure we don't try again
    return _O3D_IMPORTED

def ensure_cm_import():
    """Attempts to import matplotlib.cm if not already imported."""
    global _CM_IMPORTED, cm
    if not _CM_IMPORTED and cm is None: # Check cm too
        try:
            import matplotlib.cm as cm_lib
            cm = cm_lib # Assign to global placeholder
            _CM_IMPORTED = True
            print("Successfully imported Matplotlib colormaps.")
        except ImportError:
            print("Warning: Matplotlib not found. 3D color mapping might be affected.")
            _CM_IMPORTED = False # Ensure we don't try again
    return _CM_IMPORTED


def visualize_input_voxels_3d(coords: np.ndarray, feats: np.ndarray, hw: Tuple[int, int]):
    """Visualizes input voxels as a 3D point cloud colored by polarity."""
    # Skip if display off, or O3D import failed
    if not _DISPLAY_ENABLED or not ensure_o3d_import(): return

    if coords is None or len(coords) == 0:
        print("No input voxel data to visualize.")
        return

    H, W = hw
    # Coords are (z_bin, y_voxel, x_voxel) -> Vis: (x_voxel, y_voxel_inv, z_bin)
    points = coords[:, [2, 1, 0]].astype(np.float32)
    points[:, 1] = H - 1 - points[:, 1] # Invert y-axis

    # Derive polarity from feats (channel 0 is ON, channel 1 is OFF)
    polarity_on = feats[:, 0] > 0.5
    colors = np.zeros((len(points), 3))
    colors[polarity_on] = [0, 0, 1]  # Blue for ON
    colors[~polarity_on] = [1, 0, 0] # Red for OFF

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Add coordinate axes
    axes_pos = np.min(points, axis=0) # Place axes near minimum point
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(10, H//10, W//10), origin=axes_pos) # Adjust size based on resolution

    print("Showing input voxels (x, y_inverted, time_bin). Close window to continue.")
    o3d.visualization.draw_geometries([pcd, axes], window_name="Input Voxels (Polarity)")

def visualize_output_voxels_3d(coords: np.ndarray, conf: np.ndarray, hw: Tuple[int, int]):
    """Visualizes output voxels as a 3D point cloud colored by confidence."""
    # Skip if display off, or O3D/CM import failed
    if not _DISPLAY_ENABLED or not ensure_o3d_import() or not ensure_cm_import(): return

    if coords is None or len(coords) == 0:
        print("No output voxel data to visualize.")
        return

    H, W = hw
    # Coords are (z_bin, y_voxel, x_voxel) -> Vis: (x_voxel, y_voxel_inv, z_bin)
    points = coords[:, [2, 1, 0]].astype(np.float32)
    points[:, 1] = H - 1 - points[:, 1] # Invert y-axis

    # Map confidence (0-1) to colors using a heatmap
    conf_np = np.clip(conf, 0.0, 1.0)
    colors = cm.jet(conf_np)[:, :3] # Use JET colormap, take RGB channels

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Add coordinate axes
    axes_pos = np.min(points, axis=0)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(10, H//10, W//10), origin=axes_pos)

    print("Showing output voxels (x, y_inverted, time_bin). Close window to continue.")
    o3d.visualization.draw_geometries([pcd, axes], window_name="Output Voxels (Confidence)")


# ───────────────────────── Label Loading (from test.py) ──────────────────
def load_labels(label_path: Path) -> Dict[int, Tuple[float, float, float, float]]:
    """Loads labels from a text file. Returns empty dict if file not found or empty."""
    labels: Dict[int, Tuple[float, float, float, float]] = {}
    if not label_path.is_file():
        print(f"Info: Optional label file not found at {label_path}. Running without evaluation.")
        return labels
    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split(',')
                if len(parts) == 5:
                    try:
                        ts = int(parts[0])
                        xmin, ymin, xmax, ymax = map(float, parts[1:])
                        if xmin >= xmax or ymin >= ymax:
                            print(f"Warning: Skipping label with invalid coordinates (min >= max) at line {line_num} in {label_path}: {line}")
                            continue
                        labels[ts] = (xmin, ymin, xmax, ymax)
                    except ValueError:
                         print(f"Warning: Skipping label with non-numeric value at line {line_num} in {label_path}: {line}")
                else:
                    print(f"Warning: Skipping malformed line {line_num} in {label_path}: {line}")
        if labels:
             print(f"Loaded {len(labels)} labels from {label_path}")
        else:
             print(f"Warning: Label file {label_path} was empty or contained no valid labels.")
    except Exception as e:
        print(f"Error loading labels from {label_path}: {e}")
        return {} # Return empty dict on error
    return labels

# ───────────────────────── colour‑map helper ─────────────────────────────
@torch.no_grad()
def colourise_events(coords: np.ndarray, conf: np.ndarray, hw: Tuple[int, int], mode: str ="last") -> np.ndarray:
    """Colourise *only event pixels* using a JET map. Requires cv2."""
    H, W = hw
    frame = np.zeros((H, W, 3), dtype=np.uint8) # Start with black frame
    if len(coords) == 0:
        return frame

    # Only proceed with coloring if cv2 is available
    if not _CV2_IMPORTED:
        # Fallback: maybe return grayscale if no cv2? Or just black. Let's return black.
        # print("Debug: cv2 not imported, returning black frame from colourise_events")
        return frame

    # Coords assumed valid indices [0, H-1], [0, W-1]
    ys = coords[:, 1]
    xs = coords[:, 2]

    try:
        if mode == "last":
            gray = (conf * 255).clip(0, 255).astype(np.uint8)
            # applyColorMap expects single channel uint8, returns (N, 1, 3) BGR
            colors = cv2.applyColorMap(gray[:, np.newaxis], cv2.COLORMAP_JET)[:, 0, :]
            frame[ys, xs] = colors
        elif mode == "max":
            acc = np.zeros((H, W), dtype=np.float32)
            np.maximum.at(acc, (ys, xs), conf) # Efficient reduction
            mask = acc > 0
            if mask.any():
                gray = (acc[mask] * 255).clip(0, 255).astype(np.uint8)
                colors = cv2.applyColorMap(gray[:, np.newaxis], cv2.COLORMAP_JET)[:, 0, :]
                frame[mask] = colors
        elif mode == "mean":
            sum_img = np.zeros((H, W), dtype=np.float32)
            cnt_img = np.zeros((H, W), dtype=np.uint16)
            np.add.at(sum_img, (ys, xs), conf) # Efficient reduction
            np.add.at(cnt_img, (ys, xs), 1)
            mask = cnt_img > 0
            if mask.any():
                mean_conf = sum_img[mask] / cnt_img[mask]
                gray = (mean_conf * 255).clip(0, 255).astype(np.uint8)
                colors = cv2.applyColorMap(gray[:, np.newaxis], cv2.COLORMAP_JET)[:, 0, :]
                frame[mask] = colors
        else:
             raise ValueError(f"Unknown reduction mode: {mode}")
    except Exception as e:
        print(f"Error during cv2.applyColorMap (mode={mode}): {e}. Returning black frame.")
        # Return black frame on colormap error
        frame = np.zeros((H, W, 3), dtype=np.uint8)

    return frame


# ───────────────────────── tracker class ────────────────────────────────
class SimpleTracker:
    """Tiny FSM: detection → tracking → detection."""
    def __init__(self, crop_w: int, crop_h: int, det_thresh_logit: float, centre_thresh_logit: float, max_misses: int):
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.det_thresh_logit = det_thresh_logit
        self.centre_thresh_logit = centre_thresh_logit
        self.max_misses = max_misses
        self.last_center: Optional[Tuple[int, int]] = None # (x, y) in full‑frame coords or None
        self.misses: int = 0

    def roi(self, hw: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Return (x0, y0, x1, y1) ROI in *full frame* coords or None if not tracking."""
        if self.last_center is None: return None
        H, W = hw
        cx, cy = self.last_center
        half_w, half_h = self.crop_w // 2, self.crop_h // 2
        x0, y0 = max(0, cx - half_w), max(0, cy - half_h)
        x1, y1 = min(W, cx + half_w), min(H, cy + half_h)
        if x0 >= x1 or y0 >= y1:
             # print(f"Warning: Calculated invalid ROI ({x0},{y0},{x1},{y1}) from center ({cx},{cy}). Resetting.")
             self.reset()
             return None
        return x0, y0, x1, y1

    def update(self, logits: torch.Tensor, coords: np.ndarray, offset_xy: Tuple[int, int], hw: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Update internal state with current ROI logits/coords and return new center if found."""
        H, W = hw
        if len(logits) == 0: # No voxels in ROI/frame -> miss
            self._register_miss()
            return None

        # Check detection threshold
        max_logit = float(logits.max())
        if max_logit < self.det_thresh_logit:
             if self.last_center is not None: # Only count miss if already tracking
                 self._register_miss()
             return None # Not detected or lost track

        # Calculate weighted centroid using voxels above centre_thresh
        logits_np = logits.cpu().numpy()
        center_mask = logits_np >= self.centre_thresh_logit
        if not center_mask.any(): # Fallback if no voxels meet centre_thresh
            center_mask = logits_np >= self.det_thresh_logit # Use det_thresh as looser criterion
            if not center_mask.any(): # Still nothing? -> miss
                 self._register_miss()
                 return None

        # Coords are relative to ROI (or full frame if offset is 0,0)
        ys_roi = coords[center_mask, 1].astype(np.float32)
        xs_roi = coords[center_mask, 2].astype(np.float32)
        # Use logits as weights (shift slightly to ensure positive weights)
        ws = logits_np[center_mask] - self.centre_thresh_logit + 1e-6

        # Weighted average within ROI
        cx_roi = np.average(xs_roi, weights=ws)
        cy_roi = np.average(ys_roi, weights=ws)

        # Convert back to full-frame coordinates
        cx = int(round(cx_roi + offset_xy[0]))
        cy = int(round(cy_roi + offset_xy[1]))

        # Clip to bounds and update state
        self.last_center = (np.clip(cx, 0, W - 1), np.clip(cy, 0, H - 1))
        self.misses = 0
        return self.last_center

    def reset(self) -> None:
        """Resets the tracker to detection mode."""
        self.last_center = None
        self.misses = 0

    def _register_miss(self) -> None:
        """Increments miss counter and resets if max_misses is reached."""
        if self.last_center is not None: # Only count misses if we were tracking
            self.misses += 1
            if self.misses >= self.max_misses:
                # print(f"Tracker lost lock after {self.max_misses} misses. Resetting.")
                self.reset()

# ────────────────────── voxelisation helper ────────────────────────────
@torch.no_grad()
def voxelise_events(events: torch.Tensor, bin_us: int, offset_xy: Tuple[int, int]=(0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    """Convert events → (coords, feats) with optional XY offset for ROI processing."""
    if len(events) == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 2), dtype=np.float32)

    ox, oy = offset_xy
    xs = events[:, 0].int() - ox # Coords relative to ROI origin
    ys = events[:, 1].int() - oy
    ts = events[:, 2].long()
    ps = events[:, 3].bool() # Polarity (0 or 1)

    if bin_us <= 0: raise ValueError("time_bin_us must be positive")
    t_start = ts.min()
    z = ((ts - t_start) // bin_us).int() # Time bin index relative to batch start

    # Coords: (time_bin, y_roi, x_roi)
    coords = torch.stack((z, ys, xs), 1).cpu().numpy().astype(np.int32)
    # Feats: [is_ON, is_OFF]
    feats = np.stack((ps.cpu().numpy(), ~ps.cpu().numpy()), axis=1).astype(np.float32)

    return coords, feats

# ───────────────── colour‑map for ROI → full frame ─────────────────────
@torch.no_grad()
def colourise_events_global(coords_roi: np.ndarray, conf: np.ndarray, hw: Tuple[int, int], offset_xy: Tuple[int, int], mode: str ="last") -> np.ndarray:
    """Applies colourise_events using global coordinates. Requires cv2."""
    H, W = hw
    if len(coords_roi) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8) # Return black if no coords

    # Convert ROI coordinates back to global frame coordinates
    coords_global = coords_roi.copy()
    coords_global[:, 1] += offset_xy[1] # Add y offset
    coords_global[:, 2] += offset_xy[0] # Add x offset

    # Clip global coordinates BEFORE passing to colourise_events
    coords_global[:, 1] = np.clip(coords_global[:, 1], 0, H - 1)
    coords_global[:, 2] = np.clip(coords_global[:, 2], 0, W - 1)

    # Call standard colourise function (checks _CV2_IMPORTED internally)
    return colourise_events(coords_global, conf, hw, mode)


# ─────────────────────────── main loop ────────────────────────────────
@torch.no_grad()
def main():
    # --- Argument Parsing ---
    ap = argparse.ArgumentParser(
        description="Track objects in AEDAT4 files using a Sparse UNet and optionally evaluate against labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="Folder containing latest.pt and config.yaml")
    ap.add_argument("aedat", type=Path, help="Input *.aedat4 recording file")
    ap.add_argument("--batch_us", type=int, default=10_000, help="Temporal window size (microseconds)")
    ap.add_argument("--det_thresh", type=float, default=1.0, help="Logit threshold to initiate tracking")
    ap.add_argument("--centre_thresh", type=float, default=0.0, help="Logit threshold for centroid calculation")
    ap.add_argument("--crop_w", type=int, default=150, help="Width of the ROI when tracking")
    ap.add_argument("--crop_h", type=int, default=150, help="Height of the ROI when tracking")
    ap.add_argument("--max_misses", type=int, default=3, help="Number of misses before tracker resets")
    ap.add_argument("--reduce", choices=["last", "max", "mean"], default="max", help="Method to reduce confidence values per pixel")
    ap.add_argument("--record", type=Path, help="Optional path to save output video (MP4)")
    ap.add_argument("--no-display", action="store_true", help="Run in headless mode without displaying OpenCV windows")
    args = ap.parse_args()

    # --- Set global flags based on args ---
    global _DISPLAY_ENABLED, _RECORDING_ENABLED
    _DISPLAY_ENABLED = not args.no_display
    # Recording requires display to be enabled (as frames are generated conditionally)
    _RECORDING_ENABLED = args.record is not None and _DISPLAY_ENABLED
    if args.record is not None and not _DISPLAY_ENABLED:
        print("Info: Recording specified but --no-display is set. Recording will be disabled.")

    # --- Attempt required imports based on flags ---
    if _DISPLAY_ENABLED or _RECORDING_ENABLED:
        ensure_cv2_import()
        ensure_events_to_image_import()
    if _DISPLAY_ENABLED:
        ensure_o3d_import()
        ensure_cm_import()

    # --- Path Handling (Symlink Aware) ---
    run_dir = Path(args.run_dir).expanduser().resolve()
    # Store the original path for label lookup
    original_aedat_path = Path(args.aedat).expanduser()
    # Resolve the path for reading the actual file
    resolved_aedat_path = original_aedat_path.resolve()

    # --- Validate Input Paths ---
    if not run_dir.is_dir(): raise NotADirectoryError(f"Run directory not found: {run_dir}")
    if not resolved_aedat_path.is_file(): raise FileNotFoundError(f"AEDAT file not found at resolved path: {resolved_aedat_path}")
    print(f"Input AEDAT path: {original_aedat_path}")
    if original_aedat_path != resolved_aedat_path:
        print(f"Resolved AEDAT path: {resolved_aedat_path}")

    # --- Load Network ---
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "latest.pt"
    if not cfg_path.exists(): raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not ckpt_path.exists(): raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    print(f"Loading config from: {cfg_path}")
    cfg = Cfg.load(cfg_path)
    bin_us = getattr(cfg.dataset, "time_bin_us", 1000)
    if bin_us <= 0: raise ValueError("--time_bin_us must be positive")
    print(f"Using time bin size (z-axis): {bin_us} us")

    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_wrapper = SpMiniUNetWrapper(cfg).to(device).eval()

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_wrapper.load_state_dict(ckpt["model"], strict=False)
    model = model_wrapper.net # Get the core network
    print(f"Weights loaded successfully (trained up to epoch {ckpt.get('epoch', 'N/A')}).")

    # --- Initialize Tracker ---
    tracker = SimpleTracker(
        crop_w=args.crop_w, crop_h=args.crop_h,
        det_thresh_logit=args.det_thresh, centre_thresh_logit=args.centre_thresh,
        max_misses=args.max_misses,
    )
    print(f"Tracker initialized: ROI={args.crop_w}x{args.crop_h}, DetThresh={args.det_thresh}, CentreThresh={args.centre_thresh}, MaxMiss={args.max_misses}")

    # --- Load Labels for Evaluation (Using ORIGINAL path) ---
    label_file_path = original_aedat_path.with_suffix('.txt')
    print(f"Looking for labels at: {label_file_path}")
    labels = load_labels(label_file_path)
    evaluate = bool(labels)
    sorted_label_items = sorted(labels.items()) if evaluate else []
    label_idx = 0
    num_labels = len(sorted_label_items)

    # --- Evaluation Result Storage ---
    accuracies: List[bool] = []
    relative_errors: List[float] = []
    pixel_errors: List[float] = []
    processed_event_count = 0
    start_time = time.perf_counter()
    last_print_time = start_time

    # --- UI Setup (Conditional) ---
    if _DISPLAY_ENABLED and _CV2_IMPORTED:
        print("Display enabled. Creating windows...")
        cv2.namedWindow("events", cv2.WINDOW_NORMAL)
        cv2.namedWindow("heat", cv2.WINDOW_NORMAL)
        # Optional: Set initial size
        # cv2.resizeWindow("events", 640, 480)
        # cv2.resizeWindow("heat", 640, 480)
    elif _DISPLAY_ENABLED and not _CV2_IMPORTED:
        print("Display was requested, but OpenCV import failed. Running headless.")
        _DISPLAY_ENABLED = False # Force disable
        _RECORDING_ENABLED = False # Also disable recording
    else:
        print("Display disabled.")


    # --- Video Writer Setup (Conditional) ---
    writer = None
    fps = None
    if _RECORDING_ENABLED and _CV2_IMPORTED:
        # Ensure the output directory exists
        args.record.parent.mkdir(parents=True, exist_ok=True)
        fps = 1_000_000 / args.batch_us
        print(f"Recording enabled: {args.record} at {fps:.2f} FPS")
        # Writer initialized later when resolution is known
    elif _RECORDING_ENABLED and not _CV2_IMPORTED:
         print("Recording specified, but OpenCV import failed. Recording disabled.")
         _RECORDING_ENABLED = False # Force disable


    # --- Main Processing Loop ---
    print(f"\nStarting processing of {original_aedat_path}...")
    stream = stream_aedat(original_aedat_path, batch_us=args.batch_us)
    paused = False
    last_events_vis = None # Store last valid visualization frames for pause/display
    last_heat_vis = None
    hw: Optional[Tuple[int, int]] = None # Sensor resolution (H, W)

    try:
        while True:
            key = -1 # Default key value if no waitKey is called

            # --- Handle Paused State ---
            if paused:
                if _DISPLAY_ENABLED and _CV2_IMPORTED:
                    # Refresh display with last frames
                    if last_events_vis is not None: cv2.imshow("events", last_events_vis)
                    if last_heat_vis is not None: cv2.imshow("heat", last_heat_vis)
                    key = cv2.waitKey(30) & 0xFF # Check keys periodically
                else:
                    time.sleep(0.03) # Prevent busy-waiting when headless paused
            else:
                # --- Get Next Event Batch ---
                try:
                    events, current_hw = next(stream)
                    if hw is None: # First batch initialization
                        hw = current_hw
                        H, W = hw
                        print(f"Detected stream resolution: {W}x{H}")
                        # Initialize video writer now that resolution is known
                        if _RECORDING_ENABLED and writer is None and _CV2_IMPORTED:
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(str(args.record.resolve()), fourcc, fps, (W * 2, H))
                            if not writer.isOpened():
                                print(f"\n[ERROR] Could not open VideoWriter for {args.record}. Disabling recording.")
                                writer = None
                                _RECORDING_ENABLED = False

                    # --- Filter events with out-of-bounds coordinates ---
                    if hw is not None:
                        H, W = hw
                        valid_mask = (events[:, 0] >= 0) & (events[:, 0] < W) & \
                                     (events[:, 1] >= 0) & (events[:, 1] < H)
                        events = events[valid_mask]

                except StopIteration:
                    print("\nEnd of event stream reached.")
                    break # Exit main loop
                except Exception as e:
                    print(f"\nError reading from event stream: {e}")
                    import traceback
                    traceback.print_exc()
                    break

                # --- Basic Stats and Progress ---
                batch_event_count = len(events)
                processed_event_count += batch_event_count
                current_loop_time = time.perf_counter()
                if current_loop_time - last_print_time > 1.0:
                    elapsed_time = current_loop_time - start_time
                    rate_kevps = (processed_event_count / 1000.0) / elapsed_time if elapsed_time > 0 else 0
                    progress = f"Time: {elapsed_time:.1f}s, Rate: {rate_kevps:.1f} kEv/s"
                    if evaluate: progress += f", Labels: {label_idx}/{num_labels}"
                    print(f"\rProcessing... {progress}   ", end="")
                    last_print_time = current_loop_time

                # Skip if batch is empty AFTER filtering or resolution not set yet
                if batch_event_count == 0 or hw is None:
                    if _DISPLAY_ENABLED and _CV2_IMPORTED:
                         # Keep showing last frame if available
                         if last_events_vis is not None: cv2.imshow("events", last_events_vis)
                         if last_heat_vis is not None: cv2.imshow("heat", last_heat_vis)
                         key = cv2.waitKey(1) & 0xFF # Still process keys minimally
                    else:
                         time.sleep(0.001) # Small sleep if headless and no events
                    continue

                # --- Prepare Visualizations (only if needed) ---
                current_events_vis = None
                if (_DISPLAY_ENABLED or _RECORDING_ENABLED) and events_to_image_func:
                    current_events_vis = events_to_image_func(events, hw)

                # --- Determine ROI for Processing ---
                roi = tracker.roi(hw)
                offset_xy = (0, 0)
                events_roi = events
                roi_vis = None # For drawing later
                if roi is not None:
                    x0, y0, x1, y1 = roi
                    offset_xy = (x0, y0)
                    mask = ((events[:, 0] >= x0) & (events[:, 0] < x1) &
                            (events[:, 1] >= y0) & (events[:, 1] < y1))
                    events_roi = events[mask]
                    roi_vis = roi # Store for drawing

                # --- Network Inference ---
                logits_out = torch.empty(0, device=device) # Defaults
                out_coords = np.empty((0,3), dtype=np.int32)
                conf_out = np.empty(0)
                out_feats = np.empty((0,2), dtype=np.float32) # Store input feats for 3D vis

                if len(events_roi) > 0:
                    coords_roi, feats_roi = voxelise_events(events_roi, bin_us, offset_xy=offset_xy)
                    out_feats = feats_roi # Store for potential 3D vis

                    if len(coords_roi) > 0:
                        # --- Calculate required sparse tensor shape ---
                        num_time_bins = int(math.ceil(args.batch_us / bin_us + 1e-9))
                        if roi is None: shape_y, shape_x = hw
                        else:
                            x0, y0, x1, y1 = roi
                            shape_y, shape_x = max(1, y1 - y0), max(1, x1 - x0)
                        # Ensure min time dim is >= 2 for stride-2 convs
                        required_shape = [max(2, num_time_bins), shape_y, shape_x]

                        # --- Create SparseConvTensor ---
                        coords_tensor = torch.from_numpy(coords_roi).to(device)
                        feats_tensor = torch.from_numpy(feats_roi).to(device)
                        batch_indices = torch.zeros((len(coords_tensor), 1), dtype=torch.int32, device=device)
                        st_indices = torch.cat([batch_indices, coords_tensor], dim=1)
                        st = spconv.SparseConvTensor(
                            features=feats_tensor, indices=st_indices,
                            spatial_shape=required_shape, batch_size=1
                        )

                        # --- Run inference ---
                        out_st = model(st)
                        out_coords = out_st.indices[:, 1:].cpu().numpy() # Z,Y,X coords
                        logits_out = out_st.features[:, 0] # Assuming channel 0 is logit
                        conf_out = torch.sigmoid(logits_out).cpu().numpy()

                        # --- Update Tracker State ---
                        tracker.update(logits_out, out_coords, offset_xy, hw)
                    else: # Voxelization resulted in no voxels
                        tracker._register_miss()
                else: # No events in ROI
                    tracker._register_miss()

                # --- Create Heatmap Visualization (only if needed) ---
                current_heat_vis = None
                if _DISPLAY_ENABLED or _RECORDING_ENABLED:
                    if len(out_coords) > 0 :
                        # Requires cv2, checks internally
                        current_heat_vis = colourise_events_global(out_coords, conf_out, hw, offset_xy, mode=args.reduce)
                    elif hw is not None: # Need a black frame if no output coords
                        H, W = hw
                        current_heat_vis = np.zeros((H, W, 3), dtype=np.uint8)

                # --- Draw ROI and Tracker Center (only if needed) ---
                if (_DISPLAY_ENABLED or _RECORDING_ENABLED) and _CV2_IMPORTED:
                    if current_events_vis is not None: # Draw on event image if available
                        if roi_vis:
                            x0, y0, x1, y1 = roi_vis
                            cv2.rectangle(current_events_vis, (x0, y0), (x1, y1), (255, 255, 0), 1) # Cyan ROI
                        if tracker.last_center:
                            cx, cy = tracker.last_center
                            cv2.drawMarker(current_events_vis, (cx, cy), (0, 0, 255), # Red cross
                                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                # --- Evaluation Step (runs regardless of display) ---
                if evaluate and tracker.last_center is not None:
                    current_ts_us = int(events[-1, 2]) if len(events) > 0 else 0
                    tx, ty = tracker.last_center
                    # Process all labels up to the current time
                    while label_idx < num_labels and current_ts_us >= sorted_label_items[label_idx][0]:
                        label_ts, (xmin, ymin, xmax, ymax) = sorted_label_items[label_idx]
                        gt_cx, gt_cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                        gt_w, gt_h = xmax - xmin, ymax - ymin
                        gt_size = min(gt_w, gt_h) if gt_w > 0 and gt_h > 0 else 1.0 # Avoid size 0

                        is_accurate = (tx >= xmin) and (tx < xmax) and (ty >= ymin) and (ty < ymax)
                        accuracies.append(is_accurate)
                        pixel_error = math.sqrt((tx - gt_cx)**2 + (ty - gt_cy)**2)
                        pixel_errors.append(pixel_error)
                        if gt_size > 1e-3: # Avoid division by near-zero
                            relative_errors.append(pixel_error / gt_size)

                        label_idx += 1

                # --- Update Displays (Conditional) ---
                if _DISPLAY_ENABLED and _CV2_IMPORTED:
                    if current_events_vis is not None:
                        cv2.imshow("events", current_events_vis)
                        last_events_vis = current_events_vis # Store for pause
                    if current_heat_vis is not None:
                         cv2.imshow("heat", current_heat_vis)
                         last_heat_vis = current_heat_vis # Store for pause

                # --- Recording (Conditional) ---
                if _RECORDING_ENABLED and writer is not None:
                    if current_events_vis is not None and current_heat_vis is not None:
                        combined_frame = np.hstack((current_events_vis, current_heat_vis))
                        writer.write(combined_frame)
                    # else: Skip frame if vis data missing

                # --- Handle Key Presses ---
                if _DISPLAY_ENABLED and _CV2_IMPORTED:
                    key = cv2.waitKey(1) & 0xFF # Crucial for display updates & key checks
                else:
                    # No waitKey if headless, loop relies on stream blocking
                    pass # key remains -1

            # --- Key Handling (Common for paused and running) ---
            if key == ord('q') or key == 27: # q or ESC
                print("\nQuit key pressed.")
                break
            if key == ord(' '): # Space bar
                paused = not paused
                status = "Paused" if paused else "Running"
                print(f"\n{status}")
                # Optional: Trigger 3D visualization on pause if display is enabled
                if paused and _DISPLAY_ENABLED:
                     visualize_input_voxels_3d(out_coords, out_feats, hw) # Use out_coords and INPUT feats
                     visualize_output_voxels_3d(out_coords, conf_out, hw) # Use out_coords and OUTPUT conf

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C).")
    except Exception as e:
         print(f"\n--- An unexpected error occurred during processing ---")
         import traceback
         traceback.print_exc()
         print(f"--- Error: {e} ---")
    finally:
        # --- Cleanup ---
        print("\nCleaning up resources...")
        if writer is not None:
            writer.release()
            print(f"Recording saved to {args.record.resolve()}")
        if _DISPLAY_ENABLED and _CV2_IMPORTED:
            cv2.destroyAllWindows()
            print("Closed OpenCV windows.")
        elif not _DISPLAY_ENABLED:
            print("Display was disabled.")
        elif not _CV2_IMPORTED:
             print("OpenCV not available, no windows to close.")


        # --- Final Performance Summary ---
        end_time = time.perf_counter()
        total_time = end_time - start_time
        final_rate_kevps = (processed_event_count / 1000.0) / total_time if total_time > 0 else 0
        print(f"\nFinished processing {processed_event_count:,} events in {total_time:.2f}s ({final_rate_kevps:.1f} kEv/s)")

        # --- Print Evaluation Results ---
        if evaluate:
            num_evals = len(accuracies)
            if num_evals == 0:
                print("\n--- Evaluation Results ---")
                print("No labels were evaluated (no overlap with tracker activity or labels missing/empty).")
            else:
                overall_accuracy = sum(accuracies) / num_evals
                mean_pixel_error = sum(pixel_errors) / num_evals
                mean_relative_error = sum(relative_errors) / len(relative_errors) if relative_errors else float('nan')

                print("\n--- Evaluation Results ---")
                print(f"Labels evaluated: {num_evals}")
                print(f"Overall Accuracy (tracker center inside GT bbox): {overall_accuracy:.3f}")
                print(f"Mean Pixel Error (Euclidean distance):          {mean_pixel_error:.2f} pixels")
                print(f"Mean Relative Error (norm. by min GT dim):      {mean_relative_error:.3f} (from {len(relative_errors)} valid labels)")
            print("-------------------------")
        else:
            print("\nEvaluation skipped (no label file found or loaded).")

        print("Exiting.")


if __name__ == "__main__":
    main()
