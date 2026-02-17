# cv-multi-object
multiple object detction using advance algorithms
# Adaptive Multi-Source Object Perception & Tracking Engine (CPU-Optimized)

## Overview

This project implements a **real-time multi-object perception and tracking pipeline optimized for low-resource hardware** (CPU-only systems, integrated GPUs, and embedded environments).
The system fuses multiple perception sources — **global visual proposals, depth-based proposals, appearance-adaptive segmentation, temporal geometry memory, and motion tracking** — to produce stable, persistent object identities across frames.

Unlike heavy neural detection pipelines that require GPUs, this system emphasizes:

* CPU-efficient algorithms
* Temporal stability
* Occlusion recovery
* Multi-object persistence
* Robust performance in low-resolution or noisy camera feeds

It is designed as a **foundational perception layer** that can be integrated into robotics, AGI systems, embedded vision applications, surveillance, or edge-AI platforms.

---

## Key Features

### 1. Multi-Source Proposal Generation

Objects are detected using a combination of:

* Edge-based contour detection
* Motion-aware detection (optional)
* Temporal persistence scoring
* Depth-based external proposals (via pipe input)
* Proposal merging and Non-Maximum Suppression

This hybrid approach ensures object detection even when any single signal becomes unreliable.

---

### 2. Proposal Fusion Layer

Global visual proposals and depth-derived proposals are fused into a unified proposal set using:

* IoU-based merging
* Area-weighted scoring
* Contour solidity evaluation
* Depth confidence bonuses
* Final ranking and pruning

This produces highly stable candidate regions for further processing.

---

### 3. Adaptive Appearance Extraction

Each candidate region is refined using a **multi-mask segmentation engine** that combines:

* Adaptive HSV clustering
* Shadow-aware compensation
* Edge mask reinforcement
* Region growing for spatial continuity
* Contour confidence scoring

This enables robust extraction even under:

* Illumination variation
* Shadows
* Low-contrast scenes
* Low-resolution cameras

---

### 4. Lightweight Multi-Object Tracker

The system includes a custom low-resource tracker featuring:

* Persistent object IDs
* Distance-weighted matching
* Color-histogram similarity matching
* Optical-flow motion consistency verification
* Occlusion recovery with lost-frame memory
* Temporal persistence scoring

This ensures stable tracking even during partial object disappearance.

---

### 5. Temporal Geometry Memory

Each object maintains a short-term geometry memory that provides:

* Contour smoothing
* Kalman-filtered centroid estimation
* Lost-object prediction
* Bounding-box reconstruction
* Shape similarity matching via IoU

This allows the system to maintain identity continuity through noise, occlusions, and camera jitter.

---

### 6. Attention-Driven Primary Focus Selection

A dynamic attention scoring mechanism selects the primary focus object using:

* Persistence confidence
* Relative object size
* Motion strength
* Temporal stability

This can be used to drive higher-level decision systems or adaptive compute allocation.

---

## Architecture Pipeline

The system operates in the following stages each frame:

1. **Frame Capture**
2. **Global Proposal Generation**
3. **Depth Proposal Integration**
4. **Proposal Fusion & Ranking**
5. **Appearance-Adaptive Segmentation**
6. **Multi-Object Tracking Update**
7. **Temporal Geometry Memory Update**
8. **Primary Focus Selection**
9. **Visualization / Output**

This layered design allows modules to be replaced independently without redesigning the pipeline.

---

## Hardware Design Goals

The system is engineered for:

* CPU-only inference environments
* Integrated graphics systems
* Robotics compute boards
* Low-cost edge deployments
* Real-time performance under limited compute budgets

Typical performance depends on camera resolution and contour density but is designed for **real-time operation on mid-range CPUs**.

---

## Applications

Potential applications include:

* Robotics perception layers
* Autonomous navigation prototypes
* Edge-AI visual monitoring
* Human-object interaction systems
* Experimental AGI perception stacks
* Embedded surveillance analysis
* Assistive vision systems

Because the pipeline does not depend on large neural models, it is particularly suited to **compute-limited environments**.

---

## Running the System

### Requirements

* Python 3.8+
* OpenCV (opencv-python or opencv-contrib)
* NumPy

Optional:

* v4l2loopback for virtual camera inputs
* External depth proposal generator (pipe input)

---

### Execution

1. Ensure the camera device is available (default: `/dev/video11`)
2. Ensure the pipe file `/tmp/vision_pipe` exists if depth input is used
3. Run:

```bash
python receiver.py
```

Press `q` to exit.

---

## Configuration Notes

Key adjustable parameters:

* Tracker persistence thresholds
* Optical flow feature count
* Proposal count limits
* Contour minimum area
* Appearance extraction tolerances
* Kalman filter noise parameters
* Lock-on object stability thresholds

These can be tuned depending on scene complexity and camera characteristics.

---

## Design Philosophy

The system is built around the principle that **stable perception does not require large neural models**, but rather:

* multi-signal fusion
* temporal reasoning
* lightweight probabilistic tracking
* adaptive segmentation
* persistence-aware matching

This allows perception stacks to operate efficiently on hardware where deep learning pipelines are impractical.

---

## Future Extensions

Potential enhancements include:

* GPU-accelerated segmentation modules
* Learned appearance embeddings
* Multi-camera fusion
* Depth-aware 3D object persistence
* Semantic classification layers
* Active perception attention routing

The modular structure allows incremental upgrades without redesigning the core engine.

---

## License

MIT License — free to use, modify, and redistribute.

---

## Summary

This repository provides a **complete lightweight perception backbone** capable of multi-object detection, segmentation, temporal tracking, and identity persistence using CPU-efficient classical vision techniques.
It is intended as a foundational layer for advanced robotics, adaptive perception systems, and research-grade real-time visual intelligence pipelines.
