# RT-DETR Small Object Detection Improvements: Experimental Report

## 1. Introduction

This report presents experimental results from the **RT-DETR_SOD_IMP** project, which investigates architectural modifications to improve small object detection (SOD) in the RT-DETR (Real-Time Detection Transformer) framework. All experiments were conducted using RT-DETR with a ResNet-18 backbone trained and evaluated on the **VisDrone2019** dataset — a challenging aerial/drone-view dataset dominated by small and densely-packed objects.

### Motivation

The VisDrone2019 validation set has a highly skewed object-size distribution:

| Size Category | Count  | Percentage | Avg. Area (px²) |
|:---:|:---:|:---:|:---:|
| Small (< 32×32) | 23,217 | **65.9%** | 364.7 |
| Medium | 10,841 | 30.8% | 2,821.5 |
| Large | 1,193 | 3.4% | 17,998.9 |

With nearly two-thirds of all objects falling into the "small" category, improving small object AP is the primary objective of this work.

---

## 2. Dataset

**VisDrone2019** is a large-scale benchmark for drone-based vision tasks. It contains 10 object classes captured from various altitudes and perspectives, making it a standard benchmark for small object detection research.

- **Classes**: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
- **Split**: Standard train/val split
- **Challenge**: High proportion of small, densely-packed, partially-occluded objects

---

## 3. Architecture Overview

### 3.1 Baseline: RT-DETR-R18

The baseline is the standard RT-DETR architecture with a ResNet-18 (PResNet-18) backbone, fine-tuned on VisDrone2019. It uses the default 3-scale feature pyramid (P3/S8, P4/S16, P5/S32) with the standard Hybrid Encoder (AIFI + CCFM).

### 3.2 Modification 1 — GnConv Neck (Experiment 5)

The **Gated Normal Convolution (GnConv)** replaces the standard RepVgg fusion blocks inside the CCFM module of the Hybrid Encoder. GnConv uses a hierarchical decomposition of feature channels with depth-wise convolutions and gated interactions to capture both local and global spatial context efficiently.

- Encoder backbone: `HybridEncoderGnConvCCFM`
- Transformer AIFI attention block: **unchanged** (standard)
- CSP fusion blocks: replaced with `GnConvFusionBlock`
- Feature scales: 3 (P3/S8, P4/S16, P5/S32) — **same as baseline**

### 3.3 Modification 2 — P2 Feature Layer (Experiment 6)

This modification adds a **P2 (stride-4) feature map** from the ResNet-18 backbone (`return_idx: [0, 1, 2, 3]`), providing higher-resolution features for small object detection. The encoder and decoder are extended to handle **4 feature scales** (P2/S4, P3/S8, P4/S16, P5/S32).

- Encoder: `HybridEncoderGnConvCCFM` with `in_channels: [64, 128, 256, 512]`
- Decoder: `RTDETRTransformer` with `num_levels: 4`
- P2 features allow the model to detect objects at a much finer spatial resolution

### 3.4 Modification 3 — GnConv + P2 (Experiment 7)

This experiment combines both modifications: the 4-scale P2 feature pyramid with GnConv-based fusion in the CCFM. This is expected to combine the benefits of high-resolution features (P2) and improved multi-scale feature fusion (GnConv).

- Encoder: `HybridEncoderGnConvCCFM` with 4-scale inputs (`in_channels: [64, 128, 256, 512]`)
- Fusion blocks: GnConv-based
- Decoder: `RTDETRTransformer` with `num_levels: 4`

---

## 4. Experimental Setup

| Setting | Value |
|:---:|:---:|
| Backbone | ResNet-18 (PResNet-18) |
| Input resolution | 640×640 |
| Optimizer | AdamW |
| Training epochs | 90 (experiments 5–7); 93 (baseline) |
| Batch size | 2–4 |
| Hardware | NVIDIA GPU (CUDA) |
| Evaluation metric | COCO-style AP/AR |

---

## 5. Results

### 5.1 Detection Performance

| Model | AP@0.5:0.95 | AP@0.5 | AP@0.75 | AP_small | AP_medium | AP_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** (R18) | 0.218 | 0.362 | 0.222 | 0.136 | 0.310 | 0.439 |
| **+ GnConv** | 0.213 | 0.356 | 0.214 | 0.130 | 0.300 | 0.428 |
| **+ P2 Layer** | 0.248 | 0.399 | 0.257 | **0.171** | 0.336 | 0.448 |
| **+ GnConv + P2** | **0.267** | **0.431** | **0.277** | **0.185** | **0.363** | **0.483** |

#### Change vs Baseline (Δ)

| Model | ΔAP@0.5:0.95 | ΔAP@0.5 | ΔAP_small | ΔAP_medium | ΔAP_large |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **+ GnConv** | −0.005 (−2.4%) | −0.006 | −0.006 | −0.010 | −0.011 |
| **+ P2 Layer** | +0.029 (+**13.5%**) | +0.037 | +0.035 (+**26.0%**) | +0.025 | +0.009 |
| **+ GnConv + P2** | +0.048 (+**22.1%**) | +0.069 | +0.049 (+**35.8%**) | +0.053 | +0.044 |

### 5.2 Recall Metrics

| Model | AR@1 | AR@10 | AR@100 | AR_small | AR_medium | AR_large |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 0.097 | 0.272 | 0.368 | 0.275 | 0.488 | 0.583 |
| **+ GnConv** | 0.097 | 0.271 | 0.362 | 0.269 | 0.482 | 0.631 |
| **+ P2 Layer** | 0.107 | 0.304 | 0.404 | 0.324 | 0.504 | 0.639 |
| **+ GnConv + P2** | **0.115** | **0.314** | **0.412** | **0.329** | **0.521** | **0.652** |

### 5.3 Model Efficiency

| Model | Params (M) | GFLOPs | FPS | Latency (ms) | GPU Mem (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 21.9 | 25.2 | **46.6** | 21.5 | 96.4 |
| **+ GnConv** | **19.0** | **22.9** | 32.9 | 30.4 | **85.4** |
| **+ P2 Layer** | 24.9 | 50.9 | 37.4 | 26.7 | 108.8 |
| **+ GnConv + P2** | 28.1 | 69.8 | 14.7 | 68.1 | 121.0 |

> FPS measured at 640×640 on a single NVIDIA T4 GPU (Kaggle).

### 5.4 Model Architecture Details

| Model | Backbone Params (M) | Encoder Params (M) | Decoder Params (M) | Encoder Layers |
|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 11.19 | 6.74 | 3.93 | 165 |
| **+ GnConv** | 11.19 | 3.87 | 3.93 | 237 |
| **+ P2 Layer** | 11.19 | 9.61 | 4.07 | 240 |
| **+ GnConv + P2** | 11.19 | 9.36 | 7.54 | 330 |

---

## 6. Analysis

### 6.1 GnConv Neck (Experiment 5)

Replacing the standard RepVgg fusion blocks with GnConv blocks in the 3-scale CCFM **slightly reduced** overall AP (−2.4%) and AP_small (−4.3%). The GnConv architecture reduced the number of encoder parameters from 6.74M to 3.87M and GFLOPs from 25.2 to 22.9, but inference **slowed down** (32.9 vs 46.6 FPS) — likely due to the increased depth (165 → 237 layers) and the serial nature of GnConv's gated recursive decomposition, which is not as hardware-friendly as simple RepVgg blocks.

**Conclusion**: GnConv alone is not beneficial for this task. The parameter reduction comes at the cost of both accuracy and speed.

### 6.2 P2 Layer (Experiment 6)

Adding the high-resolution P2 (stride-4) feature level produced **significant gains**:
- AP@0.5:0.95: +13.5%
- AP_small: +26.0% (0.136 → 0.171)
- AR_small: +17.8% (0.275 → 0.324)

The P2 feature map has 4× the spatial resolution of P3, providing the decoder with much finer-grained spatial information for small objects. The cost is a ~2× increase in GFLOPs (25.2 → 50.9) and reduced FPS (46.6 → 37.4).

**Conclusion**: P2 layer addition provides a favorable accuracy-speed trade-off for small object-heavy datasets like VisDrone.

### 6.3 GnConv + P2 (Experiment 7)

The combined approach delivers the **best overall performance**:
- AP@0.5:0.95: +22.1% (0.218 → 0.267)
- AP@0.5: +19.1% (0.362 → 0.431)
- AP_small: +35.8% (0.136 → 0.185)
- AR_small: +19.6% (0.275 → 0.329)

However, this comes with a **substantial computational cost**:
- GFLOPs increased from 25.2 to 69.8 (+177%)
- FPS dropped from 46.6 to 14.7 (−68%)
- Parameters increased from 21.9M to 28.1M (+28%)

The severe FPS drop (below real-time for most applications) is primarily attributed to the P2 feature level, which dramatically increases spatial computation throughout the encoder, compounded by the complexity of GnConv blocks.

**Conclusion**: The combined model achieves the best accuracy but is not suitable for real-time deployment. It is best suited for offline analysis or scenarios where accuracy is prioritized over speed.

---

## 7. Summary Table

| Model | AP@0.5:0.95 | AP_small | AR_small | FPS | GFLOPs | Params (M) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 0.218 | 0.136 | 0.275 | 46.6 | 25.2 | 21.9 |
| **+ GnConv** | 0.213 (↓) | 0.130 (↓) | 0.269 (↓) | 32.9 (↓) | 22.9 (↓) | 19.0 (↓) |
| **+ P2 Layer** | 0.248 (↑) | 0.171 (↑) | 0.324 (↑) | 37.4 (↓) | 50.9 (↑) | 24.9 (↑) |
| **+ GnConv + P2** | **0.267** (↑) | **0.185** (↑) | **0.329** (↑) | 14.7 (↓) | 69.8 (↑) | 28.1 (↑) |

---

## 8. Conclusions

1. **The P2 feature level is the most impactful single modification** for small object detection on VisDrone2019. It delivers a +26% improvement in AP_small with a manageable computation increase.

2. **GnConv in isolation does not improve performance** for this task. While it reduces encoder parameters, its inference speed is lower than the baseline due to increased model depth and less hardware-friendly operations.

3. **GnConv + P2 combined** achieves the best AP across all size categories, with a +35.8% improvement in AP_small over the baseline, but at a significant inference speed penalty (−68% FPS). This trade-off may be acceptable for offline or non-real-time applications.

4. **VisDrone is a challenging dataset** for real-time detectors: even the best model achieves only AP_small = 0.185, highlighting the difficulty of detecting objects that occupy fewer than 1024 pixels in area.

5. **Recommended next steps**:
   - Explore more efficient implementations of GnConv (e.g., faster depthwise convolutions) to recover inference speed
   - Investigate multi-scale training with larger input resolutions (e.g., 800×800) for P2-enabled models
   - Explore knowledge distillation from the GnConv+P2 model to a more efficient student model

---

## 9. References

- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)
- [HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions (gnConv)](https://arxiv.org/abs/2207.14284)
- [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)
