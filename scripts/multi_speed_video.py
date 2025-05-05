# File: scripts/multi_speed_video.py
"""
Creates an MP4 video from an AEDAT4 file, playing back events at varying speeds
within a specified time range.

The video playback speed follows a 5-phase profile:
1. Real-time
2. Decelerate to 100x slower
3. 100x slower
4. Accelerate back to real-time
5. Real-time

Each phase occupies roughly 1/5th of the total video duration.
"""

import argparse
from pathlib import Path
import time
import math

import cv2
import numpy as np
import torch
import dv_processing as dv # Assuming dv_processing is installed

# Add repository root to path to allow importing from src
import sys
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# Assuming src.vis.events_to_image exists and works as in the provided repo content
try:
    from src.vis import events_to_image
except ImportError:
    print("Error: Could not import 'events_to_image' from 'src.vis'.")
    print("Please ensure src/vis.py exists and is importable.")
    sys.exit(1)

from typing import Tuple, Optional

# --- Constants ---
VIDEO_FPS = 60.0
REALTIME_SPEED = 1.0
MAX_SLOWDOWN = 100.0
REALTIME_US_PER_FRAME = 1_000_000.0 / VIDEO_FPS
SLOWEST_US_PER_FRAME = REALTIME_US_PER_FRAME / MAX_SLOWDOWN

# --- Helper Functions ---

def get_us_per_frame(frame_idx: int, total_frames: int):
    """
    Calculates the duration of events (in microseconds) to render for a given
    video frame index based on the 5-phase speed profile.
    """
    phase_len = total_frames / 5.0
    current_phase = math.floor(frame_idx / phase_len)
    progress_in_phase = (frame_idx % phase_len) / phase_len

    if current_phase == 0: # Phase 1: Real-time
        speed_factor = REALTIME_SPEED
    elif current_phase == 1: # Phase 2: Decelerate
        # Linear interpolation from REALTIME_SPEED to MAX_SLOWDOWN
        speed_factor = REALTIME_SPEED + (MAX_SLOWDOWN - REALTIME_SPEED) * progress_in_phase
    elif current_phase == 2: # Phase 3: Slowest
        speed_factor = MAX_SLOWDOWN
    elif current_phase == 3: # Phase 4: Accelerate
        # Linear interpolation from MAX_SLOWDOWN to REALTIME_SPEED
        speed_factor = MAX_SLOWDOWN + (REALTIME_SPEED - MAX_SLOWDOWN) * progress_in_phase
    else: # Phase 5: Real-time (and any frames beyond due to floor/rounding)
        speed_factor = REALTIME_SPEED

    # Inverse relationship: slower speed means less event time per frame
    return REALTIME_US_PER_FRAME / speed_factor


class EventReader:
    """Reads events from an AEDAT4 file within a time range, handling absolute timestamps."""
    def __init__(self, aedat_path: Path, relative_start_us: int, relative_end_us: int):
        if not aedat_path.exists():
            raise FileNotFoundError(f"AEDAT file not found: {aedat_path}")

        # Temporary reader to find the first timestamp
        try:
            temp_reader = dv.io.MonoCameraRecording(str(aedat_path))
            if not temp_reader.isEventStreamAvailable():
                raise RuntimeError(f"No event stream found in {aedat_path}")

            first_batch = temp_reader.getNextEventBatch()
            if first_batch is None or first_batch.size() == 0:
                # Try reading a bit more in case the first batch was empty
                for _ in range(10): # Try a few times
                    first_batch = temp_reader.getNextEventBatch()
                    if first_batch is not None and first_batch.size() > 0:
                        break
                if first_batch is None or first_batch.size() == 0:
                    raise RuntimeError(f"Could not read first event batch from {aedat_path}")

            # Assuming the first event in the first non-empty batch is the start
            self._recording_start_ts = first_batch[0].timestamp()
            print(f"Detected recording start timestamp (absolute): {self._recording_start_ts}")
            del temp_reader # Close the temporary reader

        except Exception as e:
             print(f"Error determining recording start time: {e}")
             # Fallback or re-raise
             raise RuntimeError("Failed to determine recording start time.") from e

        # Calculate absolute time range needed
        self._abs_start_time_us = self._recording_start_ts + relative_start_us
        self._abs_end_time_us = self._recording_start_ts + relative_end_us
        print(f"Absolute time range requested: [{self._abs_start_time_us}, {self._abs_end_time_us}]")


        # Initialize the main reader
        self._reader = dv.io.MonoCameraRecording(str(aedat_path))
        if not self._reader.isEventStreamAvailable():
            # This check should be redundant now, but keep for safety
            raise RuntimeError(f"No event stream found in {aedat_path}")

        # Get resolution
        resolution = self._reader.getEventResolution()
        if resolution is None:
             raise RuntimeError(f"Could not get resolution from {aedat_path}")
        width, height = resolution
        self._hw = (height, width)

        self._buffer = None # Initialize buffer

        # --- Manual Seeking Logic (using absolute timestamps) ---
        print(f"Manually seeking to absolute time {self._abs_start_time_us} µs by reading...")
        found_start = False
        seek_batches_read = 0
        seek_start_time = time.monotonic()
        while not found_start and self._reader.isRunning():
            batch_dv = self._reader.getNextEventBatch()
            if batch_dv is None:
                if not self._reader.isRunning():
                    print(f"\nWarning: Reached end of file after reading {seek_batches_read} batches while seeking.")
                    break
                else: continue

            seek_batches_read += 1
            if seek_batches_read % 100 == 0:
                elapsed_seek = time.monotonic() - seek_start_time
                print(f"\rSeek progress: Read {seek_batches_read} batches... ({elapsed_seek:.1f}s)", end="")

            batch_size = batch_dv.size()
            if batch_size > 0:
                 last_event_ts = batch_dv[batch_size - 1].timestamp()
                 # Compare with ABSOLUTE start time
                 if last_event_ts >= self._abs_start_time_us:
                    seek_end_time = time.monotonic()
                    print(f"\nFound batch potentially containing start time (last ts: {last_event_ts}) after reading {seek_batches_read} batches in {seek_end_time - seek_start_time:.2f}s.")
                    # Convert and store the whole batch initially
                    np_events = batch_dv.numpy()
                    xs = torch.from_numpy(np_events["x"]).int().unsqueeze(1)
                    ys = torch.from_numpy(np_events["y"]).int().unsqueeze(1)
                    ts = torch.from_numpy(np_events["timestamp"]).long().unsqueeze(1)
                    ps = torch.from_numpy(np_events["polarity"]).bool().unsqueeze(1).to(torch.uint8)
                    self._buffer = torch.cat((xs, ys, ts, ps), dim=1)
                    found_start = True

            if not self._reader.isRunning():
                if not found_start:
                     print(f"\nWarning: Reached end of file after reading {seek_batches_read} batches while seeking.")
                break
        # --- End Manual Seeking Logic ---

        if not found_start:
             print(f"\nWarning: Could not find absolute start time {self._abs_start_time_us} (EOF reached or file empty?).")

        self._eof = not self._reader.isRunning() if self._reader else True

        # Filter the initial buffer using absolute time
        if self._buffer is not None and len(self._buffer) > 0:
             self._buffer = self._buffer[self._buffer[:, 2] >= self._abs_start_time_us]
             if len(self._buffer) == 0:
                 self._buffer = None
                 # Try filling buffer immediately in case first batch ended exactly before abs_start_time_us
                 if not self._eof: self._fill_buffer()


    def get_recording_start_ts(self):
        """Returns the detected absolute timestamp of the first event."""
        return self._recording_start_ts

    def get_resolution(self):
        """Returns (Height, Width)"""
        return self._hw

    def _fill_buffer(self):
        # Uses self._abs_end_time_us internally for filtering
        if self._eof: return

        next_batch_dv = self._reader.getNextEventBatch()

        if next_batch_dv is None or next_batch_dv.size() == 0:
            if not self._reader.isRunning(): self._eof = True
            return

        batch_size = next_batch_dv.size()
        if batch_size == 0:
             if not self._reader.isRunning(): self._eof = True
             return

        np_events = next_batch_dv.numpy()
        xs = torch.from_numpy(np_events["x"]).int().unsqueeze(1)
        ys = torch.from_numpy(np_events["y"]).int().unsqueeze(1)
        ts = torch.from_numpy(np_events["timestamp"]).long().unsqueeze(1)
        ps = torch.from_numpy(np_events["polarity"]).bool().unsqueeze(1).to(torch.uint8)
        new_events = torch.cat((xs, ys, ts, ps), dim=1)

        # Filter using ABSOLUTE end time
        new_events = new_events[new_events[:, 2] <= self._abs_end_time_us]
        if len(new_events) == 0:
             # Check original batch timestamp against ABSOLUTE end time
             if next_batch_dv[batch_size - 1].timestamp() >= self._abs_end_time_us:
                  self._eof = True
             return

        if self._buffer is None or len(self._buffer) == 0:
            self._buffer = new_events
        else:
            self._buffer = torch.cat((self._buffer, new_events), dim=0)

        # Check combined buffer timestamp against ABSOLUTE end time
        if self._buffer[-1, 2] >= self._abs_end_time_us:
             self._eof = True


    def get_events_for_duration(self, frame_abs_start_ts: int, duration_us: float):
        """
        Gets a slice of events covering [frame_abs_start_ts, frame_abs_start_ts + duration_us).
        Timestamps are absolute. Returns None if end of relevant stream reached.
        """
        if duration_us <= 0:
            return torch.empty((0, 4), dtype=torch.int64)

        # Calculate absolute target end timestamp
        target_abs_end_ts = frame_abs_start_ts + int(duration_us)

        fill_attempts = 0
        MAX_FILL_ATTEMPTS = 10
        # Ensure buffer covers up to target_abs_end_ts
        while not self._eof and (self._buffer is None or len(self._buffer) == 0 or self._buffer[-1, 2] < target_abs_end_ts):
            self._fill_buffer()
            fill_attempts += 1
            if fill_attempts > MAX_FILL_ATTEMPTS and not self._eof:
                 print(f"\nWarning: Exceeded {MAX_FILL_ATTEMPTS} fill attempts. Target Ts: {target_abs_end_ts}")
                 break # Prevent potential hang
            # Check if EOF reached and buffer still doesn't cover the *start* of this frame
            if self._eof and (self._buffer is None or len(self._buffer) == 0 or self._buffer[-1, 2] < frame_abs_start_ts):
                 return None # No more relevant events

        if self._buffer is None or len(self._buffer) == 0:
             # If EOF reached or fill attempts exceeded, return empty tensor
             # Otherwise, something unexpected happened
             if self._eof or fill_attempts > MAX_FILL_ATTEMPTS:
                  return torch.empty((0, 4), dtype=torch.int64)
             else:
                  # This case should ideally not be reached if logic is sound
                  print("\nWarning: Buffer empty unexpectedly in get_events.")
                  return torch.empty((0, 4), dtype=torch.int64)


        # Select events using absolute timestamps
        frame_events_mask = (self._buffer[:, 2] >= frame_abs_start_ts) & (self._buffer[:, 2] < target_abs_end_ts)
        frame_events = self._buffer[frame_events_mask]

        # Remove used events from buffer using absolute timestamp
        self._buffer = self._buffer[self._buffer[:, 2] >= target_abs_end_ts]

        # No need to check for final None return based on EOF here,
        # just return the selected events (which might be empty)
        return frame_events


# --- Main Function ---
def main():
    ap = argparse.ArgumentParser(
        description="Generate a multi-speed video from AEDAT4 events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("aedat_file", type=Path, help="Input AEDAT4 file path.")
    ap.add_argument("output_mp4", type=Path, help="Output MP4 video file path.")
    ap.add_argument("start_time", type=float, help="Start time in seconds (relative to data).")
    ap.add_argument("end_time", type=float, help="End time in seconds (relative to data).")
    ap.add_argument("--total_frames", type=int, default=int(VIDEO_FPS * 10), # Default 10 sec video
                    help="Total number of frames for the output video.")
    ap.add_argument("--on_color", nargs=3, type=int, default=[0, 255, 0], metavar=('B', 'G', 'R'),
                    help="BGR color for ON events.")
    ap.add_argument("--off_color", nargs=3, type=int, default=[0, 0, 255], metavar=('B', 'G', 'R'),
                    help="BGR color for OFF events.")

    args = ap.parse_args()

    # --- Store relative times from args ---
    relative_start_s = args.start_time
    relative_end_s = args.end_time
    relative_start_us = int(relative_start_s * 1_000_000)
    relative_end_us = int(relative_end_s * 1_000_000)
    # --- End storing relative times ---

    on_color_tuple = tuple(args.on_color)
    off_color_tuple = tuple(args.off_color)

    print(f"Input AEDAT: {args.aedat_file}")
    print(f"Output MP4: {args.output_mp4}")
    # --- Print relative time range ---
    print(f"Data Time Range (Relative): {relative_start_s:.3f} s to {relative_end_s:.3f} s")
    # --- End print relative ---
    print(f"Output Video: {args.total_frames} frames @ {VIDEO_FPS:.1f} FPS ({args.total_frames / VIDEO_FPS:.2f} s)")
    print(f"ON Color: {on_color_tuple}, OFF Color: {off_color_tuple}")

    writer = None
    frames_with_events = 0

    try:
        # --- Pass relative times to reader ---
        reader = EventReader(args.aedat_file, relative_start_us, relative_end_us)
        hw = reader.get_resolution()
        H, W = hw
        print(f"Event resolution: {W}x{H}")

        # --- Get absolute start/end times from reader ---
        recording_start_ts = reader.get_recording_start_ts()
        abs_start_us = recording_start_ts + relative_start_us
        abs_end_us = recording_start_ts + relative_end_us
        # --- End get absolute ---


        # Ensure output directory exists
        args.output_mp4.parent.mkdir(parents=True, exist_ok=True)

        # Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output_mp4), fourcc, VIDEO_FPS, (W, H))
        if not writer.isOpened():
             raise IOError(f"Could not open video writer for {args.output_mp4}")

        print("Starting video generation...")
        frame_count = 0
        # --- Initialize current time to ABSOLUTE start ---
        current_abs_event_time_us = abs_start_us
        start_loop_time = time.perf_counter()
        # --- End initialize absolute ---

        while frame_count < args.total_frames:
            frame_us = get_us_per_frame(frame_count, args.total_frames)
            frame_us = max(0.0, frame_us)

            # --- Pass ABSOLUTE start time to reader ---
            frame_events = reader.get_events_for_duration(current_abs_event_time_us, frame_us)
            # --- End pass absolute ---

            if frame_events is None:
                 print(f"\nReader signaled end of relevant stream after frame {frame_count-1}. Stopping.")
                 break

            num_events_in_frame = len(frame_events)
            frame_img = np.zeros((H, W, 3), dtype=np.uint8)

            if num_events_in_frame > 0:
                 valid_coords_mask = (frame_events[:, 0] >= 0) & (frame_events[:, 0] < W) & \
                                     (frame_events[:, 1] >= 0) & (frame_events[:, 1] < H)
                 filtered_events = frame_events[valid_coords_mask]
                 num_events_filtered = len(filtered_events)

                 if num_events_filtered > 0:
                      frame_img = events_to_image(
                          filtered_events, hw,
                          on_color=on_color_tuple, off_color=off_color_tuple
                      )
                      frames_with_events += 1

            writer.write(frame_img)

            frame_count += 1
            # --- Update ABSOLUTE time ---
            if frame_us > 0:
                current_abs_event_time_us += int(frame_us)
            # --- End update absolute ---


            if frame_count % 10 == 0 or frame_count == args.total_frames:
                 elapsed = time.perf_counter() - start_loop_time
                 fps_actual = frame_count / elapsed if elapsed > 0 else 0
                 # --- Report progress using relative time ---
                 data_time_rendered_abs = current_abs_event_time_us
                 data_time_rendered_rel = (data_time_rendered_abs - recording_start_ts) / 1e6
                 total_data_duration_rel = (abs_end_us - abs_start_us) / 1e6
                 print(f"\rGenerated frame {frame_count}/{args.total_frames} "
                       f"({fps_actual:.1f} FPS). "
                       f"Data time rendered (Rel): {data_time_rendered_rel:.3f} s / {total_data_duration_rel:.3f} s. "
                       f"Frames with events: {frames_with_events}",
                       end="")
                 # --- End report relative ---

            # --- Check using ABSOLUTE end time ---
            if current_abs_event_time_us > abs_end_us and frame_count > 0:
                 # --- Reporting uses relative time logic from above ---
                 if frame_count % 10 != 0: # Avoid double printing progress
                     elapsed = time.perf_counter() - start_loop_time
                     fps_actual = frame_count / elapsed if elapsed > 0 else 0
                     data_time_rendered_abs = current_abs_event_time_us
                     data_time_rendered_rel = (data_time_rendered_abs - recording_start_ts) / 1e6
                     total_data_duration_rel = (abs_end_us - abs_start_us) / 1e6
                     print(f"\rGenerated frame {frame_count}/{args.total_frames} "
                           f"({fps_actual:.1f} FPS). "
                           f"Data time rendered (Rel): {data_time_rendered_rel:.3f} s / {total_data_duration_rel:.3f} s. "
                           f"Frames with events: {frames_with_events}",
                           end="")
                 # --- End reporting ---
                 print(f"\nRendered events up to absolute time {current_abs_event_time_us} µs, covering requested end time {abs_end_us} µs. Stopping.")
                 break
             # --- End check absolute ---


    # ... (Error handling and finally block remains the same) ...
    finally:
        # ... (Cleanup code is the same) ...
        print(f"Total frames with non-zero, valid events rendered: {frames_with_events}") # Keep this final report


    elapsed_total = time.perf_counter() - start_loop_time
    print(f"Finished generating {frame_count} frames in {elapsed_total:.2f} seconds.")


if __name__ == "__main__":
    main()
