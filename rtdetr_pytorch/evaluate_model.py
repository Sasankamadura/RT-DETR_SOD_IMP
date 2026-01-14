"""
RT-DETR Model Evaluation Script
Comprehensive evaluation including:
- Model complexity (parameters, GFLOPs, memory)
- Inference speed (FPS) benchmarking
- Checkpoint analysis

Usage:
    python tools/evaluate_model.py --config <config_path> --checkpoint <checkpoint_path>
    python tools/evaluate_model.py --config <config_path> --checkpoint <checkpoint_path> --benchmark-fps --device cuda
    python tools/evaluate_model.py --checkpoint <checkpoint_path> --analyze-checkpoint
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast

from src.core import YAMLConfig
import src.zoo


def analyze_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Analyze checkpoint file contents."""
    print(f"\n{'='*80}")
    print(f"Analyzing Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'checkpoint_path': checkpoint_path,
        'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
        'keys': list(checkpoint.keys()),
    }
    
    # Analyze different components
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
        print(f"Epoch: {checkpoint['epoch']}")
    
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        info['model_keys'] = len(model_state.keys())
        info['model_size_mb'] = sum(p.numel() * p.element_size() for p in model_state.values()) / (1024 * 1024)
        print(f"Model state dict: {info['model_keys']} keys, {info['model_size_mb']:.2f} MB")
    
    if 'ema' in checkpoint:
        ema_state = checkpoint['ema']['module'] if 'module' in checkpoint['ema'] else checkpoint['ema']
        info['ema_keys'] = len(ema_state.keys())
        info['ema_size_mb'] = sum(p.numel() * p.element_size() for p in ema_state.values()) / (1024 * 1024)
        print(f"EMA state dict: {info['ema_keys']} keys, {info['ema_size_mb']:.2f} MB")
        info['has_ema'] = True
    else:
        info['has_ema'] = False
    
    if 'optimizer' in checkpoint:
        info['has_optimizer'] = True
        print(f"Optimizer state: present")
    else:
        info['has_optimizer'] = False
    
    if 'lr_scheduler' in checkpoint:
        info['has_lr_scheduler'] = True
        print(f"LR Scheduler state: present")
    else:
        info['has_lr_scheduler'] = False
    
    print(f"\nTotal checkpoint size: {info['file_size_mb']:.2f} MB")
    
    return info


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_gflops(model: nn.Module, input_size: Tuple[int, int] = (640, 640), device: str = 'cpu') -> Dict[str, float]:
    """Calculate GFLOPs using different methods."""
    results = {}
    
    # Try using fvcore (Facebook's FLOPs counter)
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy_input)
            total_flops = flops.total()
            results['fvcore_gflops'] = total_flops / 1e9
            results['fvcore_params'] = parameter_count(model)['']
            
        print(f"✓ fvcore: {results['fvcore_gflops']:.2f} GFLOPs")
        
    except ImportError:
        print("⚠ fvcore not installed. Install with: pip install fvcore")
    except Exception as e:
        print(f"⚠ fvcore calculation failed: {e}")
    
    # Try using thop (Torch-OpCounter)
    try:
        from thop import profile, clever_format
        
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        with torch.no_grad():
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            results['thop_gflops'] = macs / 1e9  # MACs ≈ FLOPs for most operations
            results['thop_params'] = params
            
        print(f"✓ thop: {results['thop_gflops']:.2f} GFLOPs, {results['thop_params']/1e6:.2f}M params")
        
    except ImportError:
        print("⚠ thop not installed. Install with: pip install thop")
    except Exception as e:
        print(f"⚠ thop calculation failed: {e}")
    
    # Try using torchinfo
    try:
        from torchinfo import summary
        
        model = model.to(device)
        model.eval()
        
        stats = summary(
            model, 
            input_size=(1, 3, input_size[0], input_size[1]),
            device=device,
            verbose=0,
            col_names=["num_params", "mult_adds"],
        )
        
        results['torchinfo_params'] = stats.total_params
        results['torchinfo_mult_adds'] = stats.total_mult_adds
        results['torchinfo_gflops'] = stats.total_mult_adds / 1e9
        
        print(f"✓ torchinfo: {results['torchinfo_gflops']:.2f} GFLOPs, {results['torchinfo_params']/1e6:.2f}M params")
        
    except ImportError:
        print("⚠ torchinfo not installed. Install with: pip install torchinfo")
    except Exception as e:
        print(f"⚠ torchinfo calculation failed: {e}")
    
    return results


def benchmark_fps(
    model: nn.Module,
    input_sizes: List[Tuple[int, int]],
    device: str = 'cuda',
    num_warmup: int = 50,
    num_iterations: int = 200,
    use_amp: bool = False
) -> Dict[str, Any]:
    """Benchmark inference speed (FPS)."""
    
    print(f"\n{'='*80}")
    print(f"FPS Benchmarking on {device.upper()}")
    print(f"{'='*80}\n")
    
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for img_h, img_w in input_sizes:
        print(f"Testing input size: {img_h}x{img_w}")
        
        dummy_input = torch.randn(1, 3, img_h, img_w).to(device)
        
        # Warmup
        print(f"  Warmup ({num_warmup} iterations)...")
        with torch.no_grad():
            for _ in range(num_warmup):
                if use_amp and device == 'cuda':
                    with autocast():
                        _ = model(dummy_input)
                else:
                    _ = model(dummy_input)
        
        # Synchronize GPU
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"  Benchmarking ({num_iterations} iterations)...")
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                
                if use_amp and device == 'cuda':
                    with autocast():
                        _ = model(dummy_input)
                else:
                    _ = model(dummy_input)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        size_key = f"{img_h}x{img_w}"
        results[size_key] = {
            'latency_mean_ms': float(latencies.mean()),
            'latency_std_ms': float(latencies.std()),
            'latency_min_ms': float(latencies.min()),
            'latency_max_ms': float(latencies.max()),
            'fps_mean': 1000.0 / latencies.mean(),
            'fps_std': 1000.0 / latencies.std() if latencies.std() > 0 else 0,
        }
        
        print(f"  Latency: {results[size_key]['latency_mean_ms']:.2f} ± {results[size_key]['latency_std_ms']:.2f} ms")
        print(f"  FPS: {results[size_key]['fps_mean']:.2f} ± {results[size_key]['fps_std']:.2f}")
        print()
    
    return results


def get_memory_usage(device: str = 'cuda') -> Dict[str, float]:
    """Get GPU memory usage."""
    if device == 'cuda' and torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
    return {}


def analyze_model_layers(model: nn.Module) -> Dict[str, Any]:
    """Analyze model layer-wise parameters."""
    
    print(f"\n{'='*80}")
    print(f"Layer-wise Parameter Analysis")
    print(f"{'='*80}\n")
    
    layer_stats = {}
    
    # Group by major components
    components = {
        'backbone': [],
        'encoder': [],
        'decoder': [],
        'other': []
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        
        if 'backbone' in name:
            components['backbone'].append((name, param_count))
        elif 'encoder' in name:
            components['encoder'].append((name, param_count))
        elif 'decoder' in name:
            components['decoder'].append((name, param_count))
        else:
            components['other'].append((name, param_count))
    
    # Print component summary
    for component, params in components.items():
        total = sum(p[1] for p in params)
        if total > 0:
            layer_stats[component] = {
                'total_params': total,
                'num_layers': len(params),
            }
            print(f"{component.capitalize():15}: {total/1e6:8.2f}M params ({len(params):3d} layers)")
    
    return layer_stats


def main(args):
    """Main evaluation function."""
    
    # Analyze checkpoint only
    if args.analyze_checkpoint and args.checkpoint:
        checkpoint_info = analyze_checkpoint(args.checkpoint)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            print(f"\nCheckpoint analysis saved to: {output_path}")
        
        return
    
    # Full model evaluation
    if not args.config or not args.checkpoint:
        print("Error: Both --config and --checkpoint are required for full evaluation")
        return
    
    print(f"\n{'='*80}")
    print(f"RT-DETR Model Evaluation")
    print(f"{'='*80}\n")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    
    # Load configuration
    cfg = YAMLConfig(args.config, resume=args.checkpoint)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
        print("Using EMA weights")
    else:
        state = checkpoint['model']
        print("Using model weights")
    
    # Load model
    cfg.model.load_state_dict(state)
    model = cfg.model.deploy() if hasattr(cfg.model, 'deploy') else cfg.model
    model = model.to(args.device)
    model.eval()
    
    # Results dictionary
    results = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': args.device,
    }
    
    # 1. Parameter count
    print(f"\n{'='*80}")
    print(f"Model Parameters")
    print(f"{'='*80}\n")
    
    total_params, trainable_params = count_parameters(model)
    results['total_params'] = total_params
    results['trainable_params'] = trainable_params
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Check parameter count expectation
    if total_params > 25e6:
        print(f"\n⚠  WARNING: Parameter count ({total_params/1e6:.2f}M) is higher than expected")
        print(f"   RT-DETR R18 typically has ~20-22M parameters")
        print(f"   Difference: +{(total_params - 20e6)/1e6:.2f}M parameters")
    
    # 2. Layer-wise analysis
    if not args.skip_layer_analysis:
        layer_stats = analyze_model_layers(model)
        results['layer_stats'] = layer_stats
    
    # 3. GFLOPs calculation
    if not args.skip_flops:
        print(f"\n{'='*80}")
        print(f"GFLOPs Calculation (Input: {args.input_size[0]}x{args.input_size[1]})")
        print(f"{'='*80}\n")
        
        flops_results = calculate_gflops(model, tuple(args.input_size), args.device)
        results['flops'] = flops_results
    
    # 4. Memory usage
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        dummy_input = torch.randn(1, 3, args.input_size[0], args.input_size[1]).to(args.device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        memory_stats = get_memory_usage(args.device)
        results['memory'] = memory_stats
        
        print(f"\n{'='*80}")
        print(f"GPU Memory Usage")
        print(f"{'='*80}\n")
        print(f"Allocated: {memory_stats['allocated_mb']:.2f} MB")
        print(f"Reserved: {memory_stats['reserved_mb']:.2f} MB")
        print(f"Peak: {memory_stats['max_allocated_mb']:.2f} MB")
    
    # 5. FPS Benchmarking
    if args.benchmark_fps:
        fps_results = benchmark_fps(
            model,
            [tuple(args.input_size)] + (args.extra_sizes if args.extra_sizes else []),
            device=args.device,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            use_amp=args.use_amp
        )
        results['fps_benchmark'] = fps_results
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*80}\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")
    print(f"Parameters: {total_params/1e6:.2f}M")
    
    if 'flops' in results and results['flops']:
        if 'fvcore_gflops' in results['flops']:
            print(f"GFLOPs (fvcore): {results['flops']['fvcore_gflops']:.2f}")
        elif 'thop_gflops' in results['flops']:
            print(f"GFLOPs (thop): {results['flops']['thop_gflops']:.2f}")
        elif 'torchinfo_gflops' in results['flops']:
            print(f"GFLOPs (torchinfo): {results['flops']['torchinfo_gflops']:.2f}")
    
    if 'fps_benchmark' in results:
        for size, stats in results['fps_benchmark'].items():
            print(f"FPS @ {size}: {stats['fps_mean']:.2f} ± {stats['fps_std']:.2f}")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RT-DETR Model Evaluation')
    
    # Required arguments for full evaluation
    parser.add_argument('--config', '-c', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', '-r', type=str, help='Path to checkpoint file')
    
    # Device options
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    # Analysis options
    parser.add_argument('--analyze-checkpoint', action='store_true',
                        help='Only analyze checkpoint without loading model')
    parser.add_argument('--skip-flops', action='store_true',
                        help='Skip GFLOPs calculation')
    parser.add_argument('--skip-layer-analysis', action='store_true',
                        help='Skip layer-wise parameter analysis')
    
    # Input size
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                        help='Input image size for FLOPs calculation [height width]')
    
    # FPS benchmarking
    parser.add_argument('--benchmark-fps', action='store_true',
                        help='Run FPS benchmarking')
    parser.add_argument('--num-warmup', type=int, default=50,
                        help='Number of warmup iterations for FPS benchmark')
    parser.add_argument('--num-iterations', type=int, default=200,
                        help='Number of iterations for FPS benchmark')
    parser.add_argument('--extra-sizes', type=int, nargs='+', action='append',
                        help='Additional input sizes to benchmark [height width]')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision for FPS benchmark')
    
    # Output
    parser.add_argument('--output', '-o', type=str, default='evaluation_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    main(args)
