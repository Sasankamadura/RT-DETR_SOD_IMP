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

See **Section 10** for a deep technical analysis of exactly why GnConv underperformed here.

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

5. **Recommended next steps** (see Section 10.8 for full analysis):
   - Re-run GnConv with `expansion=1.0` and `batch_size=4` to isolate the capacity bottleneck effect
   - Test `HybridEncoderFastGnConv` (GnConv on P3–P5, RepVgg on P2) — config already exists in `rtdetr_r18vd_p2_fast_gnconv.yml`
   - Test `HybridEncoderP2GnConv` (GnConv only on P2, RepVgg on P3–P5) to evaluate scale-selective GnConv
   - Investigate multi-scale training with larger input resolutions (e.g., 800×800) for P2-enabled models
   - Explore knowledge distillation from the GnConv+P2 model to a more efficient student model

---

## 9. References

- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)
- [HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions (gnConv)](https://arxiv.org/abs/2207.14284)
- [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)

---

## 10. Deep Analysis: Why Did GnConv Underperform?

This section provides a detailed technical explanation for why GnConv alone (Experiment 5) failed to improve detection performance, addressing all contributing factors found in the code.

### 10.1 Misconception: Was GnConv Only Applied to P2?

**No. GnConv was applied to ALL feature scales.**

A common question is whether GnConv underperformed because it was only applied to the P2 layer. This is incorrect. In Experiment 5, the `HybridEncoderGnConvCCFM` class applied `CSPGnConvLayer` (GnConv-based fusion) to **every single FPN and PAN block** — meaning all three scales: P3 (S8), P4 (S16), and P5 (S32). In Experiment 7, GnConv similarly covers all four scales including P2 (S4).

There are in fact two alternative encoder classes in this codebase that selectively apply GnConv by scale, but **neither was used in the reported experiments**:

| Class | Strategy | Config File |
|:---|:---|:---|
| `HybridEncoderP2GnConv` | GnConv **only on P2** (stride ≤ 4), RepVgg on P3–P5 | `rtdetr_r18vd_p2_gnconv.yml` *(note: the config with this name uses `HybridEncoderGnConvCCFM`, not this class — this class is defined but not yet wired up to a config)* |
| `HybridEncoderFastGnConv` | RepVgg on P2 (fast, high-res), GnConv **only on P3–P5** (stride ≥ 8) | `rtdetr_r18vd_p2_fast_gnconv.yml` |

These represent two theoretically valid strategies that have not yet been evaluated. See Section 10.6 for a discussion.

---

### 10.2 The `expansion=0.5` Bottleneck Severely Limits GnConv's Capacity

The GnConv neck config (`rtdetr_r18vd_gnconv_neck.yml`) sets `expansion: 0.5`. This is a critical difference from the baseline which uses `expansion: 1.0`.

**How the CSP layers differ:**

| Setting | Baseline (`CSPRepLayer`) | GnConv Exp. 5 (`CSPGnConvLayer`) |
|:---|:---|:---|
| Input channels | 512 (= 256 × 2 merged features) | 512 |
| expansion | 1.0 | **0.5** |
| `hidden_channels` | 256 | **128** |
| Bottleneck blocks | 3 × RepVggBlock (256-dim) | 3 × GnConvFusionBlock (128-dim) |
| Output dim | 256 | 256 (via extra conv3) |

With `hidden_channels = 128`, GnConv's 4-order hierarchical channel split produces:

```
dims = [16, 32, 64, 128]  (reversed from 128 // 2^0..2^3)
pwa channels (initial gate seed): 16  ← only 12.5% of total features!
abc channels for DWConv: 240
```

The **16-channel gating seed** (`pwa`) is extremely thin — it provides the initial spatial gate that is then multiplied and projected up through 3 recursive levels. This bottleneck means the first gating operation has very limited representational capacity, which degrades the quality of all subsequent gated features. The baseline uses full 256-dim RepVgg blocks throughout and does not suffer this limitation.

> **Fix**: Set `expansion: 1.0` in the GnConv config to match the baseline's capacity. This was correctly applied in Experiment 7 (`expansion: 1.0`).

---

### 10.3 The 7×7 DWConv Is Overly Aggressive at Coarse Feature Scales

GnConv's core spatial operation is a **7×7 depthwise convolution** (DWConv) applied to all ABC channels:

```python
self.dwconv = nn.Conv2d(sum(dims), sum(dims), kernel_size=7, padding=3, groups=sum(dims))
```

At each feature pyramid level, the 7×7 DWConv covers a different fraction of the feature map:

| Scale | Feat. Map | 7×7 DWConv covers | Equiv. in original image |
|:---:|:---:|:---:|:---:|
| P3 (S8) | 80×80 | 8.8% of width | 56×56 px patch |
| P4 (S16) | 40×40 | 17.5% of width | 112×112 px patch |
| **P5 (S32)** | **20×20** | **35.0% of width** | **224×224 px patch** |

At P5 (only 20×20 tokens), a 7×7 kernel covers over a third of the feature map in each direction. This is nearly a **global average over the P5 features** for every spatial position. While this provides rich global context, it **washes out the spatial precision** needed for object localization — especially for small objects that are represented by just 1–3 tokens at P3 and a fraction of a token at P4/P5.

In contrast, **RepVgg's 3×3 kernel** covers a much more conservative local region:

| Scale | RepVgg 3×3 coverage | Equiv. in original image |
|:---:|:---:|:---:|
| P3 (S8) | 3.8% | 24×24 px |
| P4 (S16) | 7.5% | 48×48 px |
| P5 (S32) | 15.0% | 96×96 px |

RepVgg preserves much tighter spatial locality, which is beneficial for detecting small objects (average area 365 px² ≈ 19×19 px) that need precise spatial discrimination.

> **Key Insight**: GnConv's large kernel was designed for image classification with ViT-style global feature extraction. Detection — especially for small objects — needs **local spatial precision**, not global averaging.

---

### 10.4 GnConv Is a Context Operator, Not a Localization Operator

GnConv was originally proposed in **HorNet** (NeurIPS 2022) for image **classification**, where the goal is to recognize "what" is in an image, not "where" it is. Its recursive gating produces features rich in global context at the cost of spatial precision.

Object detection, and especially small object detection, requires:
1. **Spatial precision**: The model must distinguish an object at location (x, y) from background at (x+1, y).
2. **Scale-sensitive features**: Small objects need fine-grained features; large objects need broader context.
3. **Dense prediction at all spatial positions**: Every feature map cell must independently predict whether an object is present.

GnConv's recursive gating propagates information non-locally: the feature at position (i,j) is influenced by positions up to 7 pixels away (or more through the recursive multiplication chain). This creates strong long-range correlations that help classification but can hurt detection by making it harder to sharply localize small, closely-spaced objects such as those in VisDrone.

---

### 10.5 Double Residual Connections Cause "Lazy Learning"

`GnConvFusionBlock` has its own internal residual connection:

```python
def forward(self, x):
    residual = x
    x = self.norm(x)
    x = self.gnconv(x)
    x = x + residual   # ← internal residual
    return x
```

This block is then used **inside** `CSPGnConvLayer`, which also has a cross-path residual:

```python
def forward(self, x):
    x_1 = self.conv1(x)
    x_1 = self.bottlenecks(x_1)   # 3 GnConvFusionBlocks, each with internal residual
    x_2 = self.conv2(x)           # shortcut branch
    return self.conv3(x_1 + x_2)  # ← outer residual addition
```

This **nested residual structure** means gradients have multiple bypass paths to flow backwards, making it easy for the GnConv blocks to learn near-zero transformations (effectively becoming skip connections). The network may simply learn to ignore the GnConv spatial operations and rely entirely on the shortcut paths — explaining why accuracy barely changes despite the architectural modification.

> The baseline's `CSPRepLayer` has the same outer CSP structure, but `RepVggBlock` itself uses two **parallel** branches (3×3 and 1×1) that are **added together and then activated** — this is a structural re-parametrization trick that gets folded into a single 3×3 conv at inference time. During training, both branches participate fully in forward and backward passes (they are not skip/identity paths — the 1×1 branch acts as a shortcut over the 3×3 but both compute and are summed before the activation). Crucially, RepVgg does **not** add the block input `x` back after computing `conv1(x) + conv2(x)`. The block must always learn a useful spatial transformation; it has no pure identity skip. `GnConvFusionBlock`, by contrast, always preserves `x` unchanged via `x = gnconv(x) + residual`, making a zero-output from gnConv perfectly gradient-stable.

---

### 10.6 Without P2 Features, GnConv Has Nothing Useful to Contextualize

A fundamental issue is that GnConv's long-range context operations are most valuable when there are rich, fine-grained local features to contextualize. In the 3-scale (P3–P5) configuration of Experiment 5:

- Small objects (avg. 365 px²) produce, at most, a **2–4 token footprint at P3** (stride=8, so 19px → ~2.4 tokens wide)
- At P4 and P5, these objects produce **less than 1 token** on average

With so few tokens per small object, GnConv's recursive gating has little spatial detail to aggregate across. The context it assembles is dominated by background and neighboring objects, not by the target object's features.

In Experiment 7 (GnConv + P2), the addition of P2 (stride=4) means small objects now produce a **4–5 token footprint**. GnConv can now perform meaningful local-to-global aggregation on these richer feature maps. This explains why the combination works (Exp 7 is better than Exp 6 by +0.019 AP@0.5:0.95) even though GnConv alone does not.

---

### 10.7 Training Configuration Differences

Two additional training factors compounded GnConv's disadvantage in Experiment 5:

| Factor | Baseline | GnConv (Exp 5) |
|:---|:---:|:---:|
| Batch size | 4 | **2** |
| Training epochs | 93 | 90 |
| Normalization (fusion blocks) | BatchNorm | **LayerNorm** (in GnConvFusionBlock) |

**Batch size 2** vs 4 means gradients are noisier, which can slow convergence and destabilize the LayerNorm statistics inside GnConvFusionBlock. BatchNorm (used in all surrounding ConvNormLayer blocks) and LayerNorm have fundamentally different normalization behaviors — BatchNorm normalizes across the batch for each channel, while LayerNorm normalizes across channels for each sample. This normalization mismatch throughout the encoder may produce inconsistent feature statistics.

---

### 10.8 Untested Strategies Worth Exploring

The codebase contains two additional encoder variants that attempt to address the scale-specificity of GnConv that were **never evaluated**:

#### Strategy A: GnConv Only on P2 (`HybridEncoderP2GnConv`)
Apply GnConv only to the P2 fusion blocks (stride=4), where the feature maps are large enough (160×160) for the 7×7 DWConv to be local rather than global. Use RepVgg on P3–P5 for speed and precision.

**Hypothesis**: P2 feature maps are large enough (160×160) that 7×7 DWConv covers only 4.4% of the map — close to RepVgg's 3.8% at the same scale. Here GnConv's recursive gating can aggregate neighborhood information without losing spatial precision. Combined, P3–P5 retain their fast RepVgg precision.

#### Strategy B: RepVgg on P2, GnConv only on P3–P5 (`HybridEncoderFastGnConv`, config: `rtdetr_r18vd_p2_fast_gnconv.yml`)
Apply RepVgg on P2 (for efficient high-resolution processing), and GnConv on P3–P5 (where feature maps are smaller and GnConv's semi-global context gathering may be more appropriate for multi-scale reasoning).

**Hypothesis**: At P3 (80×80), GnConv's 56px-equivalent receptive field is large enough to see entire small objects and their context. At P4/P5, it can capture medium-to-large object context. Meanwhile, the expensive P2 computation stays fast with RepVgg.

#### Recommended Priority Order for Next Experiments

| Experiment | Strategy | Expected Outcome |
|:---|:---|:---|
| **A** | GnConv only on P2 (`HybridEncoderP2GnConv`), `expansion=1.0` | Best of both: fine-detail GnConv context at P2 + fast RepVgg at P3–P5 |
| **B** | Fast Hybrid (`HybridEncoderFastGnConv`), batch_size=4, `expansion=1.0` | Better context at P3–P5 while keeping P2 efficient |
| **C** | GnConv everywhere with `expansion=1.0`, batch_size=4 | Re-run Exp 5 with correct capacity settings to isolate the expansion effect |

---

### 10.9 Summary of Root Causes

| Root Cause | Severity | Fix |
|:---|:---:|:---|
| GnConv applied to ALL scales (not just P2) | Medium | Test selective application (Strategies A and B above) |
| `expansion=0.5` → 16-ch gating seed too weak | **High** | Use `expansion=1.0` |
| 7×7 DWConv too global at P4/P5 (20×20 maps) | **High** | Limit GnConv to large feature maps (P2/P3) |
| GnConv is a context op, not a localization op | **High** | Combine with P2 (as in Exp 7); or only apply at appropriate scales |
| Double residual → lazy learning | Medium | Remove internal residual in `GnConvFusionBlock` when inside CSP |
| No P2 features → nothing to contextualize | **High** | Always pair GnConv with P2 features |
| Batch size 2 (vs 4 baseline) | Medium | Use batch_size=4 |
| LayerNorm + BatchNorm normalization mismatch | Low | Use consistent normalization throughout the encoder |
| 3 fewer training epochs | Low | Train to same epoch count |

