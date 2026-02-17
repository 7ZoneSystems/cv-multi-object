# cv-multi-object

Multiple object detection using advanced algorithms

# Adaptive Multi-Source Object Perception & Tracking Engine (CPU-Optimized)

## Overview

This project implements a **real-time multi-object perception and tracking pipeline optimized for low-resource hardware** (CPU-only systems, integrated GPUs, and embedded environments).
The system fuses multiple perception sources - **global visual proposals, depth-based proposals, appearance-adaptive segmentation, temporal geometry memory, and motion tracking** to produce stable, persistent object identities across frames.

Unlike heavy neural detection pipelines that require GPUs, this system emphasizes:

* CPU-efficient algorithms
* Temporal stability
* Occlusion recovery
* Multi-object persistence
* Robust performance in low-resolution or noisy camera feeds

It is designed as a **foundational perception layer** that can be integrated into robotics, AGI systems, embedded vision applications, surveillance, or edge-AI platforms.

---

## Installation (From Release)

Download the following files from the latest **GitHub Release**:

* `recon.py` - main perception and tracking pipeline
* `vision.py` - subordinate IPC (depth/proposal) generator

Place both files in the same directory.

---

## Running the System

Open **two terminals** in the same folder.

### Terminal 1 - Start IPC Vision Source

Run:

```bash
python vision.py --run
```

This starts the perception proposal stream and creates the IPC pipe used by the tracking engine.

---

### Terminal 2 - Start Tracking Engine

```bash
python recon.py
```

Press **q** to exit the display window.

---

## Camera Configuration

If your camera device is different, edit the camera path inside **recon.py**.

Find this line:

```python
CAMERA_DEVICE = '/dev/video11'
```

Change it to your camera device, for example:

```python
CAMERA_DEVICE = '/dev/video0'
```

Linux users can check available cameras using:

```bash
v4l2-ctl --list-devices
```

---

## Optional: Video File Input

To use a video file instead of a camera, modify the same line:

```python
cap = cv2.VideoCapture("video.mp4")
```

---

## Key Features

### Multi-Source Proposal Generation

* Edge-based contour detection
* Motion-aware detection (optional)
* Temporal persistence scoring
* Depth-based external proposals (via pipe input)
* Proposal merging and Non-Maximum Suppression

### Proposal Fusion Layer

* IoU-based merging
* Area-weighted scoring
* Contour solidity evaluation
* Depth confidence bonuses
* Final ranking and pruning

### Adaptive Appearance Extraction

* Adaptive HSV clustering
* Shadow-aware compensation
* Edge mask reinforcement
* Region growing for spatial continuity
* Contour confidence scoring

### Lightweight Multi-Object Tracker

* Persistent object IDs
* Distance-weighted matching
* Color-histogram similarity matching
* Optical-flow motion consistency verification
* Occlusion recovery with lost-frame memory
* Temporal persistence scoring

### Temporal Geometry Memory

* Contour smoothing
* Kalman-filtered centroid estimation
* Lost-object prediction
* Bounding-box reconstruction
* Shape similarity matching via IoU

---

## Requirements

* Python 3.8+
* OpenCV (`opencv-python`)
* NumPy

Optional:

* `v4l2loopback` (Linux virtual cameras)
* External depth proposal generator

---

## Applications

* Robotics perception layers
* Autonomous navigation prototypes
* Edge-AI visual monitoring
* Experimental AGI perception stacks
* Embedded surveillance analysis
* Assistive vision systems

---

## License

MIT License - free to use, modify, and redistribute.

---

## Summary

This repository provides a **complete lightweight perception backbone** capable of multi-object detection, segmentation, temporal tracking, and identity persistence using CPU-efficient classical vision techniques.
It is intended as a foundational layer for advanced robotics, adaptive perception systems, and research-grade real-time visual intelligence pipelines.
