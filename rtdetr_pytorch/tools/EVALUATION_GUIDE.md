# RT-DETR Model Evaluation Tools

Comprehensive evaluation toolkit for analyzing trained RT-DETR models on VisDrone2019 dataset.

## Overview

This toolkit provides three main evaluation scripts:

1. **evaluate_model.py** - Comprehensive model profiling (parameters, GFLOPs, FPS, memory)
2. **profile_model.py** - Detailed layer-wise analysis and component breakdown
3. **detailed_coco_eval.py** - Enhanced COCO metrics with per-class analysis

## Prerequisites

Install required packages:

```powershell
pip install fvcore thop torchinfo
```

**Note**: If any package fails to install, the scripts will skip that specific metric and use alternatives.

## Quick Start

### 1. Analyze Checkpoint Contents

First, check what's in your checkpoint file:

```powershell
python tools/evaluate_model.py --checkpoint path/to/checkpoint0093.pth --analyze-checkpoint
```

**Output**:
- Checkpoint size and components
- Whether EMA weights are present
- Epoch number
- State dict sizes

### 2. Basic Model Profiling

Get parameter count and GFLOPs:

```powershell
python tools/evaluate_model.py `
    --config configs/rtdetr/your_config.yml `
    --checkpoint path/to/checkpoint0093.pth `
    --device cuda
```

**Output**:
- Total parameters (should be ~20-22M for R18, you noted 29M)
- GFLOPs calculation
- Layer-wise component breakdown
- GPU memory usage

### 3. FPS Benchmarking

Test inference speed:

```powershell
python tools/evaluate_model.py `
    --config configs/rtdetr/your_config.yml `
    --checkpoint path/to/checkpoint0093.pth `
    --benchmark-fps `
    --device cuda `
    --input-size 640 640 `
    --use-amp
```

**Output**:
- FPS (frames per second)
- Latency statistics (mean, std, min, max)
- Results for 640x640 input (add `--extra-sizes` for more resolutions)

### 4. Detailed Layer Analysis

Analyze model architecture in detail:

```powershell
python tools/profile_model.py `
    --config configs/rtdetr/your_config.yml `
    --checkpoint path/to/checkpoint0093.pth `
    --detailed `
    --model-type RT-DETR-R18
```

**Output**:
- Component-wise parameter breakdown (backbone, encoder, decoder)
- Backbone stage analysis
- Encoder/decoder configuration
- Comparison with expected baseline (20-22M params)
- Memory footprint estimation

### 5. Comprehensive COCO Evaluation

Run detailed evaluation on validation set:

```powershell
python tools/detailed_coco_eval.py `
    --config configs/rtdetr/your_config.yml `
    --checkpoint path/to/checkpoint0093.pth `
    --device cuda
```

**Output**:
- Overall COCO metrics (AP, AR at different IoUs)
- Per-class AP breakdown
- Performance by object size (small/medium/large)
- Size distribution in dataset
- Performance recommendations

## Understanding Your Results

### Expected RT-DETR R18 Baseline

According to official specs:
- **Parameters**: 20-22M (you have 29M - needs investigation)
- **GFLOPs**: ~16-20 GFLOPs @ 640x640
- **FPS**: 60-100 FPS on modern GPU (V100/A100)

### Your Training Results

Based on your training log (epoch 94):
```
AP @ IoU=0.50:0.95 = 0.219 (21.9%)
AP @ IoU=0.50      = 0.363 (36.3%)
Best epoch: 88
```

**Comparison with VisDrone Baselines**:
- VisDrone is challenging (small objects, dense scenes)
- AP of 21.9% is reasonable for this dataset
- Strong performance on large objects (AP=0.435)
- Room for improvement on small objects (AP=0.136)

## Investigating the 29M Parameter Count

Your model has ~29M parameters instead of expected ~20M. Use the profiler to investigate:

```powershell
python tools/profile_model.py `
    --config configs/rtdetr/your_config.yml `
    --checkpoint path/to/checkpoint0093.pth `
    --detailed `
    --model-type RT-DETR-R18
```

**Possible reasons**:
1. **More queries**: If `num_queries > 300`, this adds parameters
2. **Modified decoder**: Different `num_decoder_layers` or `hidden_dim`
3. **Extra heads**: Additional task-specific components
4. **Different encoder**: Modified hybrid encoder configuration

Check your config file for these settings.

## Script Options Reference

### evaluate_model.py

```
--config, -c          Path to config file
--checkpoint, -r      Path to checkpoint file
--device              Device (cuda/cpu)
--analyze-checkpoint  Only analyze checkpoint, don't load model
--skip-flops          Skip GFLOPs calculation
--skip-layer-analysis Skip layer-wise analysis
--input-size          Input size for FLOPs [height width]
--benchmark-fps       Run FPS benchmarking
--num-warmup          Warmup iterations (default: 50)
--num-iterations      Benchmark iterations (default: 200)
--use-amp             Use automatic mixed precision
--output, -o          Output JSON file
```

### profile_model.py

```
--config, -c          Path to config file
--checkpoint, -r      Path to checkpoint file
--detailed            Show detailed layer-by-layer info
--model-type          Model type for baseline comparison
--output, -o          Output JSON file
```

### detailed_coco_eval.py

```
--config, -c          Path to config file
--checkpoint, -r      Path to checkpoint file
--device              Device (cuda/cpu)
--skip-per-class      Skip per-class metrics (faster)
--save-predictions    Save predictions in COCO format
--output, -o          Output JSON file
```

## Example Workflow

Complete evaluation workflow for your trained model:

```powershell
# 1. Analyze checkpoint
python tools/evaluate_model.py `
    --checkpoint checkpoint0093.pth `
    --analyze-checkpoint `
    --output checkpoint_analysis.json

# 2. Profile model details
python tools/profile_model.py `
    --config configs/rtdetr/your_config.yml `
    --checkpoint checkpoint0093.pth `
    --detailed `
    --output model_profile.json

# 3. Get GFLOPs and basic metrics
python tools/evaluate_model.py `
    --config configs/rtdetr/your_config.yml `
    --checkpoint checkpoint0093.pth `
    --device cuda `
    --output model_metrics.json

# 4. Benchmark FPS
python tools/evaluate_model.py `
    --config configs/rtdetr/your_config.yml `
    --checkpoint checkpoint0093.pth `
    --benchmark-fps `
    --device cuda `
    --use-amp `
    --output fps_benchmark.json

# 5. Detailed COCO evaluation
python tools/detailed_coco_eval.py `
    --config configs/rtdetr/your_config.yml `
    --checkpoint checkpoint0093.pth `
    --device cuda `
    --output coco_detailed.json
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'fvcore'"

Install the package:
```powershell
pip install fvcore
```

Or skip FLOPs calculation:
```powershell
python tools/evaluate_model.py --skip-flops ...
```

### Out of memory during FPS benchmark

Reduce batch size or iterations:
```powershell
python tools/evaluate_model.py --benchmark-fps --num-iterations 50 ...
```

### Evaluation is slow

Skip per-class metrics:
```powershell
python tools/detailed_coco_eval.py --skip-per-class ...
```

## Output Files

All scripts save results to JSON files that can be:
- Analyzed programmatically
- Shared with collaborators
- Used for comparison across experiments
- Plotted/visualized with custom scripts

## Next Steps

After running evaluation:

1. **Compare results** with your training logs
2. **Identify bottlenecks** (parameters, speed, accuracy)
3. **Analyze per-class performance** to find weak categories
4. **Optimize model** based on findings (prune, quantize, architecture changes)

## Support

For issues or questions, check:
- Training logs in your output directory
- Config files for architecture settings
- This README for usage examples
