# Propeller-Based Drone Tracking with a Moving Neuromorphic Camera

## Description

This repository provides tools to process event camera data for detecting and tracking drone propellers using a sparse 3D UNet (`SpMiniUNet`). It utilizes sparse convolutions (`spconv.pytorch`) for efficiency and includes scripts for testing pre-trained models, labeling new data, and training the network.

## Testing the Tracker (with Pre-trained Weights)

1.  **Prerequisites:**
    * Install dependencies: PyTorch, `spconv.pytorch`, `dv_processing`, OpenCV (`opencv-python`), NumPy, OmegaConf. Optional: Open3D, Matplotlib for visualizations.
    * Have a trained model directory (e.g., `runs/your_run_name/`) containing `config.yaml` and a checkpoint file (e.g., `latest.pt` or `best.pt`).
    * Have an AEDAT4 recording file (`.aedat4`).
    * (Optional) For evaluation, place a ground truth label file (`.txt` with format `timestamp_us,xmin,ymin,xmax,ymax` per line) in the same directory as the `.aedat4` file, sharing the same base name.

2.  **Run the Tracker:**
    Execute the `track_aedat.py` script from the repository root, pointing it to your run directory and the recording file:
    ```bash
    python scripts/track_aedat.py <path/to/run_dir> <path/to/recording.aedat4>
    ```
    * **Example:**
        ```bash
        python scripts/track_aedat.py runs/propdet_baseline my_flight.aedat4
        ```

3.  **Common Options:**
    * `--record output.mp4`: Record the visualization (events view + heatmap) to a video file. Requires display to be enabled.
    * `--no-display`: Run without GUI windows (useful for evaluation only).
    * `--det_thresh <value>`: Adjust the detection logit threshold (default: 1.0).
    * `--crop_w <value>` / `--crop_h <value>`: Change the tracking ROI size (default: 150x150).

    *Hotkeys (display enabled): `Space` to pause/resume, `q` or `ESC` to quit.*

## Data

This project uses two main types of data:

1.  **AEDAT4 Recordings with Bounding Box Labels:**
    * Contains the raw event stream recordings (`.aedat4` files) used for testing the tracker and generating labeled data.
    * Corresponding text files (`.txt`) provide bounding box annotations (`timestamp_us,xmin,ymin,xmax,ymax`).
    * Organized into `train` and `test` sequences.
    * **Download Link:** [Google Drive Folder (AEDAT4 + Labels)](https://drive.google.com/drive/folders/1VMuQkUL7bAH954pupScn9juZC4hcdURY?usp=sharing)

2.  **Segmented Event Batches (.pt files):**
    * These are pre-processed, labeled event batches used directly for training and validation of the SpMiniUNet model.
    * Each `.pt` file contains a tensor of shape (N, 5) with columns `[x, y, t, polarity, label]`, typically representing a short time window (e.g., 10ms).
    * Generated from the *training* sequences of the AEDAT4 data using `scripts/label_events.py`.
    * Organized into `train` and `val` splits.
    * **Download Link:** [Google Drive Folder (Segmented Events)](https://drive.google.com/drive/folders/1SbzdJsArZM-xK7-K2D7oxdVgInsMXXsN?usp=sharing)
    * The default location expected by the training script (`main.py`) is `./data/`, with subfolders `./data/train/` and `./data/val/`.

## Labeling Data and Training

1.  **Labeling Event Data:**
    * Use the interactive labeling script `label_events.py` to create labeled training samples (`.pt` files) from `.aedat4` recordings.
    * Run the script, providing an input recording and an output directory for the `.pt` files:
        ```bash
        python scripts/label_events.py <path/to/recording.aedat4> -o <output_directory_for_pt_files> --batch_us 10000
        ```
    * **Instructions:** `Space` pauses playback. Select a bounding box around the propeller using the mouse. Press `Enter` or `Space` again to save the labeled batch as a `.pt` file. `c` cancels the selection, `q` or `ESC` quits.

2.  **Training the Model:**
    * **Organize Data:** Download the Segmented Event Batches (or use your own labeled data) and place the `.pt` files into `train/` and `val/` subdirectories within a root data folder (default: `./data/`, configurable in `config/config.py` via `dataset.labelled_root`).
    * **Configure:** Adjust training parameters (e.g., `output_dir`, `train.batch_size`, `train.epochs`, `train.lr`) in `config/config.py` if needed. The default output directory is `./runs/debug`.
    * **Start Training:** Run the main training script from the repository root:
        ```bash
        python main.py
        ```
    * **Resume Training:** To continue training from a checkpoint:
        ```bash
        python main.py --resume <path/to/run_dir>/latest.pt
        ```
    * **Monitor:** Track progress using TensorBoard:
        ```bash
        tensorboard --logdir <your_output_dir>
        # Example: tensorboard --logdir ./runs/debug
        ```
    * Checkpoints (`latest.pt`, `best.pt`) and the configuration (`config.yaml`) will be saved in the specified `output_dir`.

## Baseline Method

We implemented this paper as the `baseline` method:

```
T. Stewart, M.-A. Drouin, M. Picard, F. Billy Djupkep, A. Orth, and
G. Gagné, “Using neuromorphic cameras to track quadcopters,” in
Proceedings of the 2023 International Conference on Neuromorphic
Systems, ser. ICONS ’23. New York, NY, USA: Association for
Computing Machinery, 2023. [Online]. Available: https://doi.org/10.
1145/3589737.3605987
```

Our implementation can be found here:  https://github.com/overlab-kevin/neuromorphic_quadcopters
