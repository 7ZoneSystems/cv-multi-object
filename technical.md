# Integration & Parameter Tuning Guide

*(Configuration Reference for Threshold Editing and Practical Adjustments)*

This document explains **which lines control behavior**, **what each parameter does**, and **how to modify them safely** when integrating the perception pipeline into another system.

The system is intentionally designed so that **most runtime behavior can be tuned by editing a small set of constants**, without changing core logic.

---

# 1. Tracking Behavior Controls

Located in `ObjectTracker(...)` initialization:

```python
tracker = ObjectTracker(
    max_lost_frames=8,
    distance_threshold=40,
    persistence_decay=0.6,
    min_persistence=0.25
)
```

### Parameters

| Parameter            | Purpose                                       | When to Increase    | When to Decrease            |
| -------------------- | --------------------------------------------- | ------------------- | --------------------------- |
| `max_lost_frames`    | Frames object is kept after disappearance     | frequent occlusions | fast object turnover scenes |
| `distance_threshold` | Max centroid movement allowed for ID matching | fast moving objects | crowded scenes              |
| `persistence_decay`  | How fast confidence drops with movement       | jittery detections  | very stable camera          |
| `min_persistence`    | Minimum score to accept match                 | unstable matching   | objects switching IDs       |

**Example**

```python
distance_threshold = 70
```

Use for drones, vehicles, or fast-motion scenes.

---

# 2. Optical Flow Stability Controls

Inside `ObjectTracker.__init__`:

```python
self.flow_points_per_obj = 15
self.flow_consistency_thresh = 5.0
```

| Parameter                 | Purpose                                |
| ------------------------- | -------------------------------------- |
| `flow_points_per_obj`     | number of tracked keypoints per object |
| `flow_consistency_thresh` | motion disagreement tolerance          |

**Example**

For noisy video:

```python
flow_consistency_thresh = 8.0
```

---

# 3. Global Proposal Generator Controls

Located inside `generate_global_proposals()`:

### Minimum detection size

```python
adaptive_min_area = max(30, int(frame_area * 0.0005))
```

**Examples**

Detect smaller objects:

```python
frame_area * 0.0003
```

Ignore tiny noise:

```python
frame_area * 0.001
```

---

### Flat-region rejection (texture filter)

```python
if mean_grad < 5.0:
```

Increase for stronger filtering:

```python
mean_grad < 8.0
```

Lower for weak-texture scenes:

```python
mean_grad < 3.0
```

---

### Aspect ratio rejection

```python
if w / h > 10 or h / w > 10:
```

For detecting thin objects (wires, poles):

```python
> 20
```

---

### Center bias strength

```python
bonus = 1.0 - 0.2 * (dist / max_dist)
```

Disable center preference:

```python
bonus = 1.0
```

---

# 4. Proposal Fusion Controls

Located in `fuse_proposals()`:

```python
depth_bonus = 0.2
overlap_threshold = 0.4
```

| Parameter           | Effect                     |
| ------------------- | -------------------------- |
| `depth_bonus`       | strength of depth guidance |
| `overlap_threshold` | NMS suppression strength   |

**Examples**

Depth-dominant system:

```python
depth_bonus = 0.35
```

Aggressive NMS:

```python
overlap_threshold = 0.55
```

---

# 5. Appearance Extraction Controls

Inside `AppearanceExtractor(...)`:

```python
extractor = AppearanceExtractor(
    roi_margin=0.12,
    min_contour_area=100,
    shadow_threshold=0.3,
    edge_threshold1=40,
    edge_threshold2=120
)
```

| Parameter           | Purpose                          |
| ------------------- | -------------------------------- |
| `roi_margin`        | search expansion around proposal |
| `min_contour_area`  | reject small segmentation noise  |
| `shadow_threshold`  | shadow compensation strength     |
| `edge_threshold1/2` | edge detector sensitivity        |

**Examples**

Small-object environments:

```python
min_contour_area = 40
```

Low-light camera:

```python
edge_threshold1 = 25
edge_threshold2 = 80
```

---

# 6. Lock-On Attention System Controls

Located in main loop:

```python
LOCK_CONFIDENCE_THRESHOLD = 5
LOCK_PERSISTENCE_THRESHOLD = 0.7
```

| Parameter                    | Purpose                        |
| ---------------------------- | ------------------------------ |
| `LOCK_CONFIDENCE_THRESHOLD`  | frames required before locking |
| `LOCK_PERSISTENCE_THRESHOLD` | confidence needed for locking  |

**Examples**

Fast auto-lock:

```python
LOCK_CONFIDENCE_THRESHOLD = 2
```

Highly stable lock:

```python
LOCK_PERSISTENCE_THRESHOLD = 0.85
```

---

# 7. EMA Bounding Box Smoothing

```python
alpha = 0.3
```

| Value        | Effect              |
| ------------ | ------------------- |
| lower (0.1)  | smoother but slower |
| higher (0.6) | faster but jittery  |

---

# 8. Recommended Tuning Workflow

When integrating:

1. Tune **proposal min area**
2. Tune **tracker distance threshold**
3. Tune **appearance min contour area**
4. Tune **lock persistence threshold**
5. Tune **optical flow consistency**

This order avoids unstable tracking cascades.

---

# 9. Quick Profiles

### Small indoor robotics

```
distance_threshold = 30
min_contour_area = 40
flow_consistency_thresh = 6
```

### Outdoor long-range camera

```
distance_threshold = 70
adaptive_min_area *= 2
depth_bonus = 0.3
```

### Fast motion scene

```
distance_threshold = 90
persistence_decay = 0.5
flow_consistency_thresh = 10
```

---

# 10. Safe Editing Rule

Only modify:

* initialization parameters
* threshold constants
* weighting coefficients

Do **not modify matching logic or Kalman flow sections** unless redesigning the tracker.

---

# Summary

Most real-world deployment tuning requires editing **10–15 numeric constants only**, primarily:

* tracker distance
* proposal area thresholds
* persistence thresholds
* edge/texture thresholds
* NMS overlap

These allow adaptation to **different cameras, environments, and motion profiles** without architectural changes.
