# RT-DETR_SOD_IMP — Technical Documentation

This document provides a comprehensive technical reference for the **RT-DETR_SOD_IMP** repository, which extends the official [RT-DETR](https://arxiv.org/abs/2304.08069) and [RT-DETRv2](https://arxiv.org/abs/2407.17140) implementations with custom improvements for **Small Object Detection (SOD)**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Implementations](#3-implementations)
   - [RT-DETR PyTorch (v1)](#31-rt-detr-pytorch-v1)
   - [RT-DETRv2 PyTorch (v2)](#32-rt-detrv2-pytorch-v2)
   - [RT-DETR PaddlePaddle](#33-rt-detr-paddlepaddle)
   - [RT-DETRv2 PaddlePaddle](#34-rt-detrv2-paddlepaddle)
4. [SOD-Specific Improvements](#4-sod-specific-improvements)
   - [CoordGnConv — Novel Architecture](#41-coordgnconv--novel-architecture)
   - [SPDConv — Space-to-Depth Convolution](#42-spdconv--space-to-depth-convolution)
   - [Hybrid Encoder Variants](#43-hybrid-encoder-variants)
5. [Model Architecture](#5-model-architecture)
   - [Backbone Networks](#51-backbone-networks)
   - [Hybrid Encoder (CCFM)](#52-hybrid-encoder-ccfm)
   - [Transformer Decoder](#53-transformer-decoder)
   - [Loss and Matching](#54-loss-and-matching)
6. [Model Zoo & Performance](#6-model-zoo--performance)
7. [Dataset Support](#7-dataset-support)
8. [Installation](#8-installation)
9. [Training](#9-training)
10. [Evaluation](#10-evaluation)
11. [Export & Deployment](#11-export--deployment)
12. [Configuration System](#12-configuration-system)
13. [PyTorch Hub](#13-pytorch-hub)
14. [Benchmarking](#14-benchmarking)
15. [Citation](#15-citation)

---

## 1. Project Overview

**RT-DETR** (*Real-Time DEtection TRansformer*) is the first end-to-end real-time object detector that outperforms YOLO models in both speed and accuracy. Published at **CVPR 2024**, it eliminates the need for non-maximum suppression (NMS) post-processing by using a DETR-style decoder with learned query selection.

This fork, **RT-DETR_SOD_IMP**, builds on top of the official codebase and introduces novel encoder architectures targeting **Small Object Detection**:

- **CoordGnConv**: Coordinate-Aware Gated Normalised Convolution that replaces the standard recursive gnConv with a parallel gating mechanism sensitive to spatial coordinates.
- **SPDConv**: Space-to-Depth Convolution that preserves all pixel information during downsampling, avoiding information loss that harms small object features.
- Multiple **HybridEncoder** variants that integrate the above modules into the Feature Pyramid Network (FPN) and Path Aggregation Network (PAN) neck.

**Key properties of the base RT-DETR:**

| Property | Description |
|---|---|
| End-to-end detection | No NMS required |
| Real-time | 74–217 FPS on T4 GPU (TRT FP16) |
| Flexible speed | Adjustable by varying decoder layers without retraining |
| Strong accuracy | 46.4–54.8 AP on COCO val2017 |
| Pre-trained weights | Available for COCO and Objects365 |

---

## 2. Repository Structure

```
RT-DETR_SOD_IMP/
├── README.md                    # English overview
├── README_cn.md                 # Chinese overview
├── LICENSE                      # Apache 2.0
├── .gitignore
├── hubconf.py                   # PyTorch Hub integration
├── benchmark/                   # Benchmarking utilities
│   ├── dataset.py
│   ├── trtinfer.py
│   ├── utils.py
│   ├── yolov8_onnx.py
│   ├── README.md
│   └── trtexec.md
├── rtdetr_pytorch/              # RT-DETR v1 — PyTorch
├── rtdetrv2_pytorch/            # RT-DETRv2 — PyTorch
├── rtdetr_paddle/               # RT-DETR v1 — PaddlePaddle
└── rtdetrv2_paddle/             # RT-DETRv2 — PaddlePaddle
```

### `rtdetr_pytorch/` — detailed layout

```
rtdetr_pytorch/
├── requirements.txt
├── configs/
│   ├── runtime.yml
│   ├── dataset/
│   │   └── coco_detection.yml
│   └── rtdetr/
│       ├── include/                      # Shared sub-configs
│       ├── rtdetr_r18vd_6x_coco.yml
│       ├── rtdetr_r50vd_6x_coco.yml
│       ├── rtdetr_regnet_6x_coco.yml
│       ├── rtdetr_visdrone_r18vd.yml
│       ├── rtdetr_r18vd_sod_novel.yml    # Novel SOD config
│       ├── rtdetr_r18vd_gnconv_neck.yml  # GNConv encoder
│       ├── rtdetr_r18vd_p2_gnconv.yml    # P2-level GNConv
│       ├── rtdetr_r18vd_p2_fast_gnconv.yml
│       └── rtdetr_r18vd_p2_gnconv_repvgg.yml
├── tools/
│   ├── train.py
│   ├── infer.py
│   ├── export_onnx.py
│   └── README.md
├── src/
│   ├── core/                  # Config system (YAML + DI)
│   ├── data/                  # Dataset & transforms
│   ├── nn/backbone/           # PResNet, RegNet, DLA
│   ├── optim/                 # Optimizer, EMA, AMP
│   ├── solver/                # Training/eval engine
│   └── zoo/rtdetr/            # Model definitions
│       ├── rtdetr.py
│       ├── hybrid_encoder.py
│       ├── hybrid_encoder_fast_gnconv.py
│       ├── hybrid_encoder_p2_gnconv.py
│       ├── hybrid_encoder_gnconv_ccfm.py
│       ├── hybrid_encoder_sod_novel.py   # Novel SOD encoder
│       ├── rtdetr_decoder.py
│       ├── rtdetr_criterion.py
│       ├── rtdetr_postprocessor.py
│       ├── matcher.py
│       ├── denoising.py
│       ├── box_ops.py
│       ├── coord_gnconv.py               # CoordGnConv + SPDConv
│       ├── gnconv_module.py
│       └── utils.py
├── evaluate_model.py           # Param/FLOPs/FPS analysis
├── detailed_coco_eval.py       # Per-class COCO evaluation
├── profile_model.py            # Complexity profiling
├── verify_novel_sod.py         # SOD architecture verification
├── test_gnconv_neck_integration.py
└── test_fast_gnconv.py
```

---

## 3. Implementations

### 3.1 RT-DETR PyTorch (v1)

Located in `rtdetr_pytorch/`. This is the primary implementation used for the SOD research contributions in this fork.

**Key files:**

| File | Description |
|---|---|
| `src/zoo/rtdetr/rtdetr.py` | Top-level model: backbone → encoder → decoder |
| `src/zoo/rtdetr/hybrid_encoder.py` | Standard efficient hybrid encoder (AIFI + CCFM) |
| `src/zoo/rtdetr/rtdetr_decoder.py` | Transformer decoder (574 lines): cross-attention, IoU-aware query selection |
| `src/zoo/rtdetr/rtdetr_criterion.py` | Hungarian-matching loss: L1 + GIoU + classification |
| `src/zoo/rtdetr/rtdetr_postprocessor.py` | Converts decoder outputs to box/score predictions |
| `src/zoo/rtdetr/denoising.py` | DINO-style denoising training with contrastive queries |
| `src/zoo/rtdetr/matcher.py` | Bipartite matching (HungarianMatcher) |
| `src/zoo/rtdetr/box_ops.py` | Box utilities: `box_cxcywh_to_xyxy`, GIoU, IoU |
| `src/nn/backbone/presnet.py` | PResNet: ResNet with optional depthwise-separable convolutions |
| `src/nn/backbone/regnet.py` | RegNet backbone |
| `src/nn/backbone/dla.py` | Deep Layer Aggregation (DLA-34) backbone |
| `src/solver/det_engine.py` | Per-epoch training and evaluation loops |
| `src/solver/det_solver.py` | Full training orchestration with AMP, EMA, DDP |
| `src/optim/ema.py` | Exponential Moving Average of model weights |
| `src/data/coco/coco_dataset.py` | COCO dataset loader with category remapping |

### 3.2 RT-DETRv2 PyTorch (v2)

Located in `rtdetrv2_pytorch/`. Adds "Bag-of-Freebies" improvements over v1.

**Additions over v1:**

| Feature | Description |
|---|---|
| `rtdetrv2_decoder.py` | Enhanced decoder with flexible sampling strategy (grid vs. discrete) |
| `rtdetrv2_criterion.py` | Improved loss combining multiple augmentation tricks |
| Mosaic augmentation | `src/data/transforms/mosaic.py` |
| HGNetv2 backbone | Lightweight efficient backbone for S/M/L/X model variants |
| CSP backbones | CSP-ResNet and CSP-Darknet |
| timm integration | Third-party backbone support via `timm` |
| VOC dataset | `src/data/dataset/voc_detection.py` |
| Docker support | `docker-compose.yml` for containerized training |

**Sampling strategies in v2:**

| Strategy | Description | Compatibility |
|---|---|---|
| `grid_sampling` | Uses `grid_sample` for attention map sampling. Full accuracy. | TensorRT ≥ 8.5 |
| `discrete_sampling` | Uses `index_select`. Slight accuracy drop (~0.2 AP), but compatible with older TensorRT. | TensorRT ≥ 8.4 |

### 3.3 RT-DETR PaddlePaddle

Located in `rtdetr_paddle/`. The original framework implementation; serves as the reference for model weights. Requires `paddlepaddle-gpu==2.4.2`.

### 3.4 RT-DETRv2 PaddlePaddle

Located in `rtdetrv2_paddle/`. PaddlePaddle implementation of v2.

---

## 4. SOD-Specific Improvements

These are the novel contributions of this fork, targeting detection of small objects.

### 4.1 CoordGnConv — Novel Architecture

**File:** `rtdetr_pytorch/src/zoo/rtdetr/coord_gnconv.py`

**Motivation:** Standard gnConv uses recursive high-order spatial interactions, which are computationally expensive and suboptimal for small objects that require fine-grained spatial attention. `CoordGnConv` replaces the recursive order-N gating with **parallel coordinate-aware gating**.

**Architecture:**

```
Input (B, C, H, W)
    │
    ├─ proj_in: Conv2d(C,C,1) → BN → SiLU
    │
    ├─ Coordinate Gating Branch:
    │   ├─ x_h = mean(x, dim=W)   → shape (B, C, H, 1)
    │   ├─ x_w = mean(x, dim=H)ᵀ → shape (B, C, W, 1) [transposed]
    │   ├─ concat([x_h, x_w], dim=H+W)
    │   ├─ conv1(C→mip) → BN → SiLU
    │   ├─ split → a_h, a_w
    │   ├─ a_h = conv_h(mip→C).sigmoid()
    │   ├─ a_w = conv_w(mip→C).sigmoid()
    │   └─ gated_x = x * a_h * a_w       ← multiplicative gating
    │
    ├─ Spatial Refinement:
    │   ├─ dwconv: DepthwiseConv2d(C,C,7,p=3) → BN
    │   └─ proj_out: Conv2d(C,C,1) → BN
    │
    └─ Output = proj_out(gated_x) + identity   ← residual connection
```

**Key design choices:**

- `mip = max(8, C // reduction)` — reduction ratio controls the bottleneck size (default `reduction=4`)
- Coordinate pooling uses `mean()` instead of `AdaptiveAvgPool2d` for ONNX export compatibility
- Depthwise conv with kernel=7 provides a large receptive field for local interaction
- `BatchNorm2d` preferred over `LayerNorm` for T4 GPU optimisation

### 4.2 SPDConv — Space-to-Depth Convolution

**File:** `rtdetr_pytorch/src/zoo/rtdetr/coord_gnconv.py`

**Reference:** [arXiv:2208.03640](https://arxiv.org/abs/2208.03640) — *SPD-Conv: Replacing the Strided Convolution / Pooling Layer in CNNs*

**Motivation:** Strided convolutions and max-pooling discard spatial information during downsampling. For small objects occupying only a few pixels, this information loss is catastrophic. SPDConv moves spatial pixels into the channel dimension (zero information loss) before applying a pointwise convolution.

**Implementation:**

```python
class SPDConv(nn.Module):
    def __init__(self, ch_in, ch_out, dimension=1):
        self.conv = nn.Conv2d(ch_in * 4, ch_out, kernel_size=1)
        self.bn   = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        # x: [B, C, H, W]
        x = torch.cat([
            x[..., 0::2, 0::2],   # top-left pixels
            x[..., 1::2, 0::2],   # bottom-left
            x[..., 0::2, 1::2],   # top-right
            x[..., 1::2, 1::2],   # bottom-right
        ], dim=1)
        # x: [B, 4C, H/2, W/2] — all spatial info preserved
        return self.bn(self.conv(x))
```

SPDConv replaces strided convolutions in the **PAN bottom-up pathway**, where downsampling is required to merge low-resolution high-semantics features with high-resolution low-semantics features.

### 4.3 Hybrid Encoder Variants

All variants are in `rtdetr_pytorch/src/zoo/rtdetr/`:

| Class | File | Description |
|---|---|---|
| `HybridEncoder` | `hybrid_encoder.py` | Baseline: AIFI + CSPRepLayer FPN/PAN |
| `HybridEncoderGnConvCCFM` | `hybrid_encoder_gnconv_ccfm.py` | Replaces CSPRepLayer with gnConv-based CSP blocks |
| `HybridEncoderFastGnConv` | `hybrid_encoder_fast_gnconv.py` | Faster gnConv variant with reduced computation |
| `HybridEncoderP2GnConv` | `hybrid_encoder_p2_gnconv.py` | Adds a P2 feature level (stride 4) for small objects |
| `HybridEncoderNovelCoordGnConv` | `hybrid_encoder_sod_novel.py` | Full SOD encoder: CoordGnConv FPN + SPDConv PAN |

**`HybridEncoderNovelCoordGnConv` — the flagship SOD encoder:**

```
Backbone outputs: [P3(s=8), P4(s=16), P5(s=32)]
         │
         ▼
   input_proj: Conv2d → BN  (per-level channel normalisation to hidden_dim)
         │
         ▼ (on highest-stride level)
   AIFI Transformer Encoder (self-attention on flattened P5 features)
         │
         ▼
   Top-Down FPN (CoordGnConv blocks):
   P5 → lateral_conv → upsample ──┬─ concat ─► CSPCoordGnLayer ─► P4_inner
                                  P4                               │
   P4_inner → upsample ───────────┬─ concat ─► CSPCoordGnLayer ─► P3_inner
                                  P3
         │
         ▼
   Bottom-Up PAN (SPDConv + CoordGnConv blocks):
   P3_inner → SPDConv ─┬─ concat ─► CSPCoordGnLayer ─► P4_out
                       P4_inner                         │
   P4_out → SPDConv ───┬─ concat ─► CSPCoordGnLayer ─► P5_out
                       P5_inner
         │
         ▼
   Output: [P3_out, P4_out, P5_out]
```

**`CSPCoordGnLayer`** (inside `hybrid_encoder_sod_novel.py`):
- CSP split: two parallel 1×1 conv branches
- Main branch: N × `CoordGnConvFusionBlock` stacked sequentially
- Residual branch: direct 1×1 conv
- Merged by element-wise addition + optional 1×1 proj

---

## 5. Model Architecture

### 5.1 Backbone Networks

| Backbone | File | Notes |
|---|---|---|
| PResNet-18/34/50/101 | `src/nn/backbone/presnet.py` | ResNet with optional depth-wise separable convs; variant "vd" uses avg-pool in stem |
| RegNet | `src/nn/backbone/regnet.py` | Efficient structured network design |
| DLA-34 | `src/nn/backbone/dla.py` | Deep Layer Aggregation; 34-layer variant |
| HGNetv2 (v2 only) | v2 implementation | Lightweight backbone for S/M/L/X variants |

Backbones output multi-scale feature maps `[C3, C4, C5]` with strides `[8, 16, 32]`.

### 5.2 Hybrid Encoder (CCFM)

The **Cross-scale Channel Fusion Module (CCFM)** consists of:

1. **AIFI** (Attention in Feature Interaction): A single-scale transformer encoder applied to the highest-stride (most semantic) feature map, providing global context.
2. **FPN** (top-down pathway): Upsamples and fuses from high-stride to low-stride using lateral convolutions + CSP-based fusion blocks.
3. **PAN** (bottom-up pathway): Downsamples back up using strided convolutions (or SPDConv in SOD variants) + CSP-based fusion blocks.

**Parameters (default for R18):**

| Parameter | Default | Description |
|---|---|---|
| `in_channels` | `[512, 1024, 2048]` | Channels from backbone (R50 example) |
| `hidden_dim` | `256` | Uniform channel width inside encoder |
| `nhead` | `8` | Number of attention heads in AIFI |
| `dim_feedforward` | `1024` | FFN width in AIFI |
| `use_encoder_idx` | `[2]` | Which scale indices to apply AIFI to |
| `num_encoder_layers` | `1` | AIFI transformer depth |
| `expansion` | `1.0` | CSP hidden channel expansion ratio |
| `depth_mult` | `1.0` | Scales number of CSP bottleneck blocks |

### 5.3 Transformer Decoder

**File:** `src/zoo/rtdetr/rtdetr_decoder.py`

The decoder implements iterative refinement over `num_decoder_layers` (default 6) cross-attention layers:

1. **IoU-Aware Query Selection**: Selects the top-K encoder features as initial object queries, ranked by a combined objectness score (`cls_score * iou_score`).
2. **Self-Attention**: Queries attend to each other for deduplication.
3. **Cross-Attention** (deformable or standard): Each query attends to multi-scale encoder features using reference point offsets.
4. **FFN**: Feed-forward refinement of each query embedding.
5. **Box/Class Prediction Heads**: Lightweight MLPs predicting `Δ(cx, cy, w, h)` offsets and class logits.

**Denoising Training (DINO-style):**
During training, noisy object queries (perturbed GT boxes + class labels) are mixed with the learned queries. The decoder is trained to reconstruct the clean targets from the noisy inputs, providing a contrastive denoising signal.

### 5.4 Loss and Matching

**File:** `src/zoo/rtdetr/rtdetr_criterion.py`, `src/zoo/rtdetr/matcher.py`

| Component | Details |
|---|---|
| **Matching** | One-to-one bipartite matching using the Hungarian algorithm |
| **Box loss** | L1 loss on normalised `(cx, cy, w, h)` coordinates |
| **GIoU loss** | Generalised IoU loss for better box localisation |
| **Classification loss** | Focal loss (varifocal in v2) |
| **Denoising loss** | Separate L1 + GIoU + classification on denoising queries |

Loss weights (default):
- `loss_vfl_weight = 1.0` (v2) / `loss_ce_weight = 1.0` (v1)
- `loss_bbox_weight = 5.0`
- `loss_giou_weight = 2.0`

---

## 6. Model Zoo & Performance

### RT-DETR v1 (PyTorch)

| Model | Dataset | Input | AP<sup>val</sup> | AP<sub>50</sub> | Params (M) | FLOPs (G) | FPS (T4 TRT FP16) | Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RT-DETR-R18 | COCO | 640 | 46.4 | 63.7 | 20 | 60 | 217 | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth) |
| RT-DETR-R34 | COCO | 640 | 48.9 | 66.8 | 31 | 92 | 161 | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth) |
| RT-DETR-R50-m | COCO | 640 | 51.3 | 69.5 | 36 | 100 | 145 | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth) |
| RT-DETR-R50 | COCO | 640 | 53.1 | 71.2 | 42 | 136 | 108 | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth) |
| RT-DETR-R101 | COCO | 640 | 54.3 | 72.8 | 76 | 259 | 74 | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth) |
| RT-DETR-R18 | COCO+Objects365 | 640 | **49.0** | 66.5 | 20 | 60 | 217 | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth) |
| RT-DETR-R50 | COCO+Objects365 | 640 | **55.2** | 73.4 | 42 | 136 | 108 | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth) |
| RT-DETR-R101 | COCO+Objects365 | 640 | **56.2** | 74.5 | 76 | 259 | 74 | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth) |
| RT-DETR-RegNet | COCO | 640 | 51.6 | 69.6 | 38 | — | 67 | [download](https://drive.google.com/file/d/1K2EXJgnaEUJcZCLULHrZ492EF4PdgVp9/view) |
| RT-DETR-DLA34 | COCO | 640 | 49.6 | 67.4 | 34 | — | 83 | [download](https://drive.google.com/file/d/1_rVpl-jIelwy2LDT3E4vdM4KCLBcOtzZ/view) |

> Weights are converted from the official PaddlePaddle checkpoints. Slight differences from the paper are expected.

### RT-DETRv2 (PyTorch) — Base Models

| Model | Dataset | Input | AP<sup>val</sup> | AP<sub>50</sub> | Params (M) | FPS | Config | Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RT-DETRv2-S | COCO | 640 | **48.1** (+1.6) | **65.1** | 20 | 217 | [config](rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml) | [download](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth) |
| RT-DETRv2-M* | COCO | 640 | **49.9** (+1.0) | **67.5** | 31 | 161 | [config](rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml) | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth) |
| RT-DETRv2-M | COCO | 640 | **51.9** (+0.6) | **69.9** | 36 | 145 | [config](rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml) | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth) |
| RT-DETRv2-L | COCO | 640 | **53.4** (+0.3) | **71.6** | 42 | 108 | [config](rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml) | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth) |
| RT-DETRv2-X | COCO | 640 | 54.3 | **72.8** (+0.1) | 76 | 74 | [config](rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml) | [download](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth) |

> AP improvements are compared to the corresponding RT-DETR v1 model.  
> FPS measured on T4 GPU, batch=1, FP16, TensorRT ≥ 8.5.1.

---

## 7. Dataset Support

### COCO 2017

```
path/to/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

Configure paths in `configs/dataset/coco_detection.yml`:
```yaml
img_folder: /path/to/coco/train2017/
ann_file: /path/to/coco/annotations/instances_train2017.json
```

**Category remapping:** By default, `remap_mscoco_category: True` remaps COCO category IDs (1-indexed, with gaps) to contiguous 0-indexed IDs. Disable this when training on custom datasets:
```yaml
remap_mscoco_category: False
```

### PASCAL VOC (RT-DETRv2 only)

Supported via `src/data/dataset/voc_detection.py` in `rtdetrv2_pytorch/`. Config: `configs/dataset/voc_detection.yml`.

### VisDrone (Small Object)

Config available at `configs/rtdetr/rtdetr_visdrone_r18vd.yml`. VisDrone is a drone-captured dataset with many small objects — an ideal benchmark for the SOD improvements in this fork.

---

## 8. Installation

### Requirements (PyTorch v1)

```bash
cd rtdetr_pytorch
pip install -r requirements.txt
```

```
torch==2.0.1
torchvision==0.15.2
onnx==1.14.0
onnxruntime==1.15.1
pycocotools
PyYAML
scipy
transformers
```

### Requirements (PyTorch v2)

```bash
cd rtdetrv2_pytorch
pip install -r requirements.txt
```

```
torch>=2.0.1
torchvision>=0.15.2
faster-coco-eval>=1.6.6
PyYAML
tensorboard
scipy
pycocotools
onnx
onnxruntime-gpu
```

**torch / torchvision compatibility:**

| torch | torchvision |
|:---:|:---:|
| 2.4 | 0.19 |
| 2.2 | 0.17 |
| 2.1 | 0.16 |
| 2.0 | 0.15 |

### Docker (RT-DETRv2)

```bash
cd rtdetrv2_pytorch
docker-compose up
```

---

## 9. Training

### Single GPU

```bash
cd rtdetr_pytorch
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

### Multiple GPUs (DDP)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

### Mixed Precision Training (AMP)

```bash
torchrun --nproc_per_node=4 tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --use-amp --seed=0
```

### Resume Training

```bash
python tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r path/to/checkpoint.pth
```

### Fine-tuning (Tuning)

Fine-tune from a pretrained checkpoint (e.g., Objects365 weights → COCO):

```bash
python tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -t path/to/pretrained.pth
```

### Novel SOD Encoder Training

Use the dedicated config:

```bash
python tools/train.py \
    -c configs/rtdetr/rtdetr_r18vd_sod_novel.yml
```

### Training Arguments

| Argument | Description |
|---|---|
| `-c / --config` | Path to YAML config file |
| `-r / --resume` | Resume from checkpoint path |
| `-t / --tuning` | Fine-tune from pretrained weights |
| `--test-only` | Run evaluation only (no training) |
| `--use-amp` | Enable automatic mixed precision |
| `--seed` | Random seed for reproducibility |

---

## 10. Evaluation

### Evaluation Only

```bash
# Single GPU
python tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r path/to/checkpoint.pth \
    --test-only

# Multi GPU
torchrun --nproc_per_node=4 tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r path/to/checkpoint.pth \
    --test-only
```

### Per-Class COCO Evaluation

```bash
python detailed_coco_eval.py \
    -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
    -r path/to/checkpoint.pth
```

### Model Complexity Analysis (Params / FLOPs / FPS)

```bash
python evaluate_model.py \
    -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml
```

### Architecture Verification (SOD)

```bash
python verify_novel_sod.py
```

---

## 11. Export & Deployment

### ONNX Export

```bash
# v1
python tools/export_onnx.py \
    -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
    -r path/to/checkpoint.pth \
    --check

# v2
python tools/export_onnx.py \
    -c path/to/config.yml \
    -r path/to/checkpoint.pth \
    --check
```

### TensorRT Export (v2)

```bash
python tools/export_trt.py -i path/to/model.onnx
```

### Inference Scripts

All inference backends are in `rtdetrv2_pytorch/references/deploy/`:

```bash
# ONNX Runtime
python references/deploy/rtdetrv2_onnxruntime.py \
    --onnx-file=model.onnx --im-file=image.jpg

# TensorRT
python references/deploy/rtdetrv2_tensorrt.py \
    --trt-file=model.trt --im-file=image.jpg

# OpenVINO
python references/deploy/rtdetrv2_openvino.py \
    --xml-file=model.xml --im-file=image.jpg

# PyTorch (reference)
python references/deploy/rtdetrv2_torch.py \
    -c path/to/config.yml \
    -r path/to/checkpoint.pth \
    --im-file=image.jpg \
    --device=cuda:0
```

---

## 12. Configuration System

Configurations use a **YAML-based dependency injection** system (`src/core/yaml_config.py`). Configs are composed via `__include__` directives:

```yaml
# rtdetr_r18vd_6x_coco.yml
__include__: [
    '../dataset/coco_detection.yml',
    'include/rtdetr_r18vd.yml',
    'include/optimizer.yml',
    'include/dataloader.yml',
]

# Overrides:
epoches: 72
```

**Key config sections:**

```yaml
# Model
model: RTDETR
  backbone: PResNet
    depth: 18
    pretrained: true
  encoder: HybridEncoder
    in_channels: [128, 256, 512]
    hidden_dim: 256
  decoder: RTDETRTransformer
    num_decoder_layers: 3    # fewer layers → faster, more layers → more accurate

# Training
optimizer: AdamW
  lr: 0.0001
  weight_decay: 0.0001
lr_warmup_scheduler:
  warmup_duration: 2000
epoches: 72

# Data
train_dataloader:
  dataset:
    img_folder: /path/to/coco/train2017
    ann_file: /path/to/coco/annotations/instances_train2017.json
  batch_size: 4
  num_workers: 4
```

**Flexible inference speed**: Varying `num_decoder_layers` (3, 4, or 6) adjusts the speed/accuracy trade-off without retraining the full model.

---

## 13. PyTorch Hub

Load pretrained models directly without cloning:

```python
import torch

# Load RT-DETR-R18 (46.4 AP, 217 FPS)
model = torch.hub.load('lyuwenyu/RT-DETR', 'rtdetr_r18vd', pretrained=True)

# Available models:
# rtdetr_r18vd, rtdetr_r34vd, rtdetr_r50vd_m, rtdetr_r50vd, rtdetr_r101vd
# rtdetrv2_r18vd, rtdetrv2_r34vd, rtdetrv2_r50vd_m, rtdetrv2_r50vd, rtdetrv2_r101vd
```

**hubconf.py** registers all 10+ models. Weights are downloaded automatically from GitHub releases on first use.

---

## 14. Benchmarking

The `benchmark/` directory contains tools for evaluating inference performance:

| Script | Description |
|---|---|
| `trtinfer.py` | Benchmarks TensorRT inference (FPS, latency) |
| `dataset.py` | Dataset loader utility for benchmarks |
| `utils.py` | Benchmark helper functions |
| `yolov8_onnx.py` | ONNX-based comparison with YOLOv8 models |

See `benchmark/README.md` and `benchmark/trtexec.md` for detailed usage instructions.

---

## 15. Citation

If you use this repository in your research, please cite the original papers:

```bibtex
@inproceedings{lv2023detrs,
  title     = {DETRs Beat YOLOs on Real-time Object Detection},
  author    = {Yian Zhao and Wenyu Lv and Shangliang Xu and Jinman Wei and
               Guanzhong Wang and Qingqing Dang and Yi Liu and Jie Chen},
  booktitle = {CVPR},
  year      = {2024},
  eprint    = {2304.08069},
  archivePrefix = {arXiv},
}

@misc{lv2024rtdetrv2,
  title         = {RT-DETRv2: Improved Baseline with Bag-of-Freebies for
                   Real-Time Detection Transformer},
  author        = {Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and
                   Guanzhong Wang and Yi Liu},
  year          = {2024},
  eprint        = {2407.17140},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2407.17140},
}
```

---

## License

This project is released under the [Apache 2.0 License](LICENSE).
