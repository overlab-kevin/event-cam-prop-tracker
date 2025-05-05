#!/usr/bin/env python
"""
benchmark.py - Run track_aedat.py on multiple AEDAT files and average metrics.

This script iterates through all *.aedat4 files in a specified directory,
runs scripts/track_aedat.py headlessly on each, parses the evaluation
output, and reports individual and average performance metrics.

Label files (.txt) are expected to reside in the same directory as their
corresponding .aedat4 files.

Example Usage:
-------------
python scripts/benchmark.py runs/propdet /path/to/aedat/directory --batch_us 10000 --det_thresh 1.5
"""

import argparse
from pathlib import Path
import subprocess
import sys
import re
import numpy as np
from typing import List, Dict, Any, Optional


# Matches the final performance summary line
# Example: Finished processing 1,234,567 events in 12.34s (100.0 kEv/s)
PERF_SUMMARY_RE = re.compile(
    r"Finished processing [\d,]+ events in ([\d.]+)s \(([\d.]+) kEv/s\)"
)

# Matches lines in the evaluation results block
# Example: Overall Accuracy (tracker center inside GT bbox): 0.950
# Example: Mean Pixel Error (Euclidean distance):          15.20 pixels
# Example: Mean Relative Error (norm. by min GT dim):      0.150 (from 123 valid labels)
# Example: Labels evaluated: 123
EVAL_METRIC_RE = re.compile(
    r"^\s*(Overall Accuracy|Mean Pixel Error|Mean Relative Error|Labels evaluated)" # Metric name
    r".*?:\s*([\d.]+)" # The numeric value (can be float or int)
)


def parse_tracker_output(output: str) -> Dict[str, Optional[float]]:
    """
    Parses the stdout of track_aedat.py to extract performance metrics.

    Args:
        output: The captured stdout string.

    Returns:
        A dictionary containing parsed metrics (float values).
        Returns None for metrics not found or if parsing fails.
    """
    results: Dict[str, Optional[float]] = {
        "Accuracy": None,
        "Pixel Error": None,
        "Relative Error": None,
        "Labels Evaluated": None,
        "kEv/s": None,
        "Processing Time (s)": None,
    }

    # Parse performance summary
    perf_match = PERF_SUMMARY_RE.search(output)
    if perf_match:
        try:
            results["Processing Time (s)"] = float(perf_match.group(1))
            results["kEv/s"] = float(perf_match.group(2))
        except (ValueError, IndexError):
            print("Warning: Could not parse performance summary line.")

    # Parse evaluation metrics
    lines = output.splitlines()
    in_eval_section = False
    for line in lines:
        if "--- Evaluation Results ---" in line:
            in_eval_section = True
            continue
        if in_eval_section:
            if "-------------------------" in line:
                in_eval_section = False # End of section
                continue

            match = EVAL_METRIC_RE.match(line)
            if match:
                metric_name = match.group(1).strip()
                try:
                    value = float(match.group(2))
                    if metric_name == "Overall Accuracy":
                        results["Accuracy"] = value
                    elif metric_name == "Mean Pixel Error":
                        results["Pixel Error"] = value
                    elif metric_name == "Mean Relative Error":
                        results["Relative Error"] = value
                    elif metric_name == "Labels evaluated":
                         results["Labels Evaluated"] = value
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse value for metric '{metric_name}'.")

    return results


def main():
    # --- Argument Parsing (Mirroring track_aedat.py, adjusting 'aedat') ---
    ap = argparse.ArgumentParser(
        description="Benchmark track_aedat.py across multiple AEDAT files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="Folder containing latest.pt and config.yaml")
    ap.add_argument("aedat_dir", type=Path, help="DIRECTORY containing input *.aedat4 recording files")
    # Passthrough arguments for track_aedat.py
    ap.add_argument("--batch_us", type=int, default=10_000, help="Temporal window size (microseconds)")
    ap.add_argument("--det_thresh", type=float, default=1.0, help="Logit threshold to initiate tracking")
    ap.add_argument("--centre_thresh", type=float, default=0.0, help="Logit threshold for centroid calculation")
    ap.add_argument("--crop_w", type=int, default=150, help="Width of the ROI when tracking")
    ap.add_argument("--crop_h", type=int, default=150, help="Height of the ROI when tracking")
    ap.add_argument("--max_misses", type=int, default=3, help="Number of misses before tracker resets")
    ap.add_argument("--reduce", choices=["last", "max", "mean"], default="max", help="Method to reduce confidence values per pixel")
    # --no-display and --record are handled internally or omitted

    args = ap.parse_args()

    # --- Validate Input Directory ---
    aedat_dir = args.aedat_dir.expanduser().resolve()
    if not aedat_dir.is_dir():
        print(f"Error: Provided AEDAT path is not a directory: {aedat_dir}")
        sys.exit(1)

    # --- Find AEDAT Files ---
    aedat_files = sorted(list(aedat_dir.glob("*.aedat4")))
    if not aedat_files:
        print(f"Error: No *.aedat4 files found in directory: {aedat_dir}")
        sys.exit(1)

    print(f"Found {len(aedat_files)} AEDAT files in {aedat_dir}")

    # --- Locate the track_aedat.py script ---
    # Assumes benchmark.py and track_aedat.py are in the same directory (scripts/)
    tracker_script_path = Path(__file__).parent / "track_aedat.py"
    if not tracker_script_path.is_file():
        print(f"Error: Could not find track_aedat.py at expected location: {tracker_script_path}")
        sys.exit(1)

    # --- Store results from each run ---
    all_results: List[Dict[str, Optional[float]]] = []

    # --- Loop Through Files and Run Tracker ---
    for i, aedat_file_path in enumerate(aedat_files):
        print("-" * 60)
        print(f"Processing file {i+1}/{len(aedat_files)}: {aedat_file_path.name}")
        print("-" * 60)

        # Construct the command for the subprocess
        command = [
            sys.executable, # Use the same python interpreter
            str(tracker_script_path),
            str(args.run_dir),
            str(aedat_file_path),
            "--no-display", # Ensure headless execution
            # Add other passthrough arguments
            "--batch_us", str(args.batch_us),
            "--det_thresh", str(args.det_thresh),
            "--centre_thresh", str(args.centre_thresh),
            "--crop_w", str(args.crop_w),
            "--crop_h", str(args.crop_h),
            "--max_misses", str(args.max_misses),
            "--reduce", args.reduce,
        ]

        try:
            # Run the tracker script as a subprocess
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True, # Raise exception on non-zero exit code
                encoding='utf-8' # Explicitly set encoding
            )

            # Parse the output
            metrics = parse_tracker_output(process.stdout)
            all_results.append(metrics)

            # Display individual results
            print(f"\nResults for {aedat_file_path.name}:")
            for key, value in metrics.items():
                 print(f"  {key:<20}: {value if value is not None else 'N/A'}")
            print("")

        except subprocess.CalledProcessError as e:
            print(f"\nError running track_aedat.py for {aedat_file_path.name}:")
            print(f"  Return Code: {e.returncode}")
            print("  --- STDOUT ---")
            print(e.stdout)
            print("  --- STDERR ---")
            print(e.stderr)
            print("-" * 60)
            # Add a placeholder result indicating failure
            all_results.append({key: None for key in all_results[0].keys()} if all_results else {})


        except Exception as e:
            print(f"\nAn unexpected error occurred processing {aedat_file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            # Add a placeholder result indicating failure
            all_results.append({key: None for key in all_results[0].keys()} if all_results else {})


    # --- Calculate and Display Average Metrics ---
    print("=" * 60)
    print("Benchmark Complete - Average Metrics")
    print("=" * 60)

    if not all_results:
        print("No results were collected.")
        sys.exit(0)

    # Use numpy's nanmean to ignore None/NaN values during averaging
    avg_metrics: Dict[str, float] = {}
    metric_keys = list(all_results[0].keys()) # Get keys from the first result dict

    for key in metric_keys:
        # Extract values for the current key, replacing None with np.nan
        values = [res.get(key, np.nan) if res else np.nan for res in all_results]
        # Filter out potential non-numeric types just in case (though None becomes nan)
        numeric_values = [v for v in values if isinstance(v, (int, float))]

        if numeric_values: # Check if there are any valid numbers to average
             avg_metrics[key] = np.nanmean(numeric_values).item() # Use nanmean
        else:
             avg_metrics[key] = np.nan # Assign NaN if no valid numbers

    # Print average results
    num_successful_runs = sum(1 for res in all_results if res and res.get("Accuracy") is not None) # Count runs with at least one metric
    print(f"Averaged over {num_successful_runs} successful run(s) out of {len(all_results)} total files.")

    for key, value in avg_metrics.items():
        print(f"  Average {key:<20}: {value:.3f}" if not np.isnan(value) else f"  Average {key:<20}: N/A")

    print("=" * 60)


if __name__ == "__main__":
    main()
