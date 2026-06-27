# Architecture Comparison: Standard YOLOv8 vs. AMT-YOLO (Adaptive Memory Trajectory YOLO)

## 1. Architecture (Network Structure)

### Standard YOLOv8

The Standard YOLOv8 architecture is a static object detector that processes only one image at a time. Each input image is passed through the Backbone for feature extraction, followed by the Neck (FPN/PAN) for multi-scale feature fusion, and finally the Detection Head, which predicts bounding boxes and object classes. Since every image is processed independently, the model has no memory of previous frames.

Processing pipeline:

* Input Image
* Backbone
* Neck (FPN/PAN)
* Detection Head
* Output: Bounding Boxes and Object Classes

### AMT-YOLO (Adaptive Memory Trajectory YOLO)

AMT-YOLO extends the conventional YOLOv8 architecture by introducing temporal memory and a dual-head prediction mechanism. Instead of processing frames independently, each video frame is analyzed while considering historical information stored in memory. After feature extraction using the YOLOv8 backbone, features are refined by the Adaptive Feature Fusion Neck, passed through the Temporal Memory Module (ConvGRU), and finally forwarded to two prediction heads.

The first head performs conventional object detection, while the second predicts future object trajectories approximately 5–10 frames ahead.

Processing pipeline:

* Input Video Frame
* YOLOv8 Backbone
* Adaptive Feature Fusion Neck (Attention-based Feature Fusion)
* Temporal Memory Module (ConvGRU)
* Detection Head (Current Object Detection)
* Trajectory Head (Future Motion Prediction)

---

# 2. Processing Pipeline

## Standard YOLOv8

The Standard YOLOv8 processes every frame independently.

* Frame 1 is processed from scratch and produces bounding boxes.
* Frame 2 is processed again from scratch.
* Information from Frame 1 is completely discarded.

As a result, the detector has no temporal awareness and cannot utilize information from previous frames.

## AMT-YOLO

AMT-YOLO maintains temporal continuity across consecutive frames.

* Frame 1 is processed to extract features.
* The model generates object detections and stores the corresponding Hidden State.
* When Frame 2 arrives, its extracted features are combined with the Hidden State from Frame 1.
* The updated memory improves current detection accuracy while simultaneously predicting the object's movement over the next several frames.

This temporal memory enables the detector to maintain object consistency across video sequences.

---

# 3. Additional Capabilities

Compared with Standard YOLOv8, AMT-YOLO introduces several additional capabilities.

### Future Trajectory Prediction

The model predicts an object's future movement before it actually occurs by estimating its trajectory several frames ahead.

### Occlusion Immunity

Temporary occlusions no longer immediately cause object loss. Because historical motion information is stored in memory, the detector can continue tracking objects even when they are briefly hidden behind obstacles such as poles, trees, or vehicles.

### Adaptive Resolution

When distant objects become difficult to recognize, the system can dynamically request higher image resolution or perform intelligent zooming to improve detection performance.

---

# 4. Trade-Offs

The additional capabilities of AMT-YOLO introduce several computational trade-offs.

### Detection Speed

* **YOLOv8 Nano:** Approximately 120 FPS on an NVIDIA RTX 4050 GPU.
* **AMT-YOLO:** The ConvGRU memory module introduces approximately 20–30% computational overhead, reducing inference speed to roughly 80 FPS.

Despite this reduction, AMT-YOLO remains suitable for real-time applications operating at 60 FPS.

### Memory Consumption

* **YOLOv8:** Requires relatively little GPU memory because only the current frame is stored during inference.
* **AMT-YOLO:** Requires additional VRAM to maintain temporal hidden states. A GPU with at least 6 GB of VRAM, such as the RTX 4050, is recommended to avoid Out-of-Memory errors.

### Training Complexity

* **YOLOv8:** Training uses randomly shuffled images.
* **AMT-YOLO:** Training requires ordered video sequences and Backpropagation Through Time (BPTT), making the overall training process approximately three times longer.

---

# 5. Technical Architecture (Deep Dive)

## Standard YOLOv8

The input consists of an image tensor with dimensions **[B, 3, H, W]**.

### Backbone (Modified CSPDarknet53)

The backbone contains:

* Initial convolution layer (Kernel = 3, Stride = 2)
* C2f modules for efficient feature extraction
* Spatial Pyramid Pooling Fast (SPPF) module

The backbone outputs three multi-scale feature maps:

* P3
* P4
* P5

These feature maps represent different spatial resolutions for detecting both small and large objects.

### Neck (Path Aggregation Network)

The PANet neck fuses P3, P4, and P5 using top-down and bottom-up pathways.

Its objectives include:

* Combining semantic information from multiple scales.
* Improving small-object detection.
* Providing richer contextual information.
* Refining feature representations before prediction.

### Detection Head

YOLOv8 employs a decoupled detection head consisting of two branches:

* Bounding Box Regression Head
* Classification Head

The detector follows an anchor-free design that directly predicts object centers without predefined anchor boxes.

---

## AMT-YOLO

The input consists of a video tensor with dimensions **[B, T, 3, H, W]**.

### YOLOv8 Feature Extractor

The backbone is identical to the original YOLOv8 architecture.

During the initial stage of training, the backbone is frozen while feature maps are extracted for every frame.

Outputs include:

* P3[t]
* P4[t]
* P5[t]

### Adaptive Feature Fusion Neck

Unlike the conventional PANet, this module integrates an Attention Mechanism.

Its primary responsibilities are:

* Fusing multi-scale feature maps.
* Learning adaptive feature importance.
* Assigning attention weights to different feature levels.
* Reducing channel dimensions to decrease VRAM usage before entering the temporal memory module.

### Temporal Memory Module (ConvGRU)

The ConvGRU module introduces temporal memory into the detection pipeline.

For each frame, the module receives:

* Current fused features.
* Hidden State from the previous frame.

The hidden state is updated using four sequential computations:

* Update Gate
* Reset Gate
* Candidate Memory
* Hidden State Update

The corresponding equations are:

**Update Gate**

Zₜ = Sigmoid(Conv(Features[t] + Hidden[t−1]))

**Reset Gate**

Rₜ = Sigmoid(Conv(Features[t] + Hidden[t−1]))

**Candidate Memory**

Candidate = Tanh(Conv(Features[t] + (Rₜ × Hidden[t−1])))

**Updated Hidden State**

Hidden[t] = (1 − Zₜ) × Hidden[t−1] + Zₜ × Candidate

The updated Hidden State is then stored in GPU memory and reused during the processing of subsequent frames.

### Dual Prediction Heads

The Detection Head performs conventional object detection by predicting:

* Current bounding boxes.
* Object categories.

The additional Trajectory Head estimates future object movement by regressing motion vectors:

* ΔX
* ΔY

These displacement vectors enable the model to predict object locations several frames into the future, providing trajectory forecasting in addition to standard object detection.
