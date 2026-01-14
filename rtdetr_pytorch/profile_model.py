"""
RT-DETR Model Profiler
Detailed layer-wise analysis and profiling for RT-DETR models

Usage:
    python tools/profile_model.py --config <config_path> --checkpoint <checkpoint_path>
    python tools/profile_model.py --config <config_path> --checkpoint <checkpoint_path> --detailed
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

from src.core import YAMLConfig
import src.zoo


def analyze_layer_parameters(model: nn.Module, detailed: bool = False) -> Dict[str, Any]:
    """Analyze parameters layer by layer."""
    
    print(f"\n{'='*80}")
    print(f"Layer-wise Parameter Analysis")
    print(f"{'='*80}\n")
    
    # Group by component
    components = defaultdict(lambda: {'params': 0, 'layers': []})
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_shape = tuple(param.shape)
        
        # Determine component
        if 'backbone' in name:
            component = 'backbone'
        elif 'encoder' in name:
            component = 'encoder'
        elif 'decoder' in name:
            component = 'decoder'
        elif 'criterion' in name or 'loss' in name:
            component = 'criterion'
        else:
            component = 'other'
        
        components[component]['params'] += param_count
        components[component]['layers'].append({
            'name': name,
            'params': param_count,
            'shape': param_shape,
            'trainable': param.requires_grad
        })
    
    # Print component summary
    print("Component Summary:")
    print(f"{'Component':<20} {'Parameters':>15} {'% Total':>10} {'Layers':>8}")
    print("-" * 60)
    
    total_params = sum(c['params'] for c in components.values())
    
    results = {
        'total_params': total_params,
        'components': {}
    }
    
    for component in sorted(components.keys()):
        comp_data = components[component]
        pct = (comp_data['params'] / total_params * 100) if total_params > 0 else 0
        
        print(f"{component:<20} {comp_data['params']:>15,} {pct:>9.2f}% {len(comp_data['layers']):>8}")
        
        results['components'][component] = {
            'total_params': comp_data['params'],
            'percentage': pct,
            'num_layers': len(comp_data['layers'])
        }
        
        if detailed:
            results['components'][component]['layers'] = comp_data['layers']
    
    print(f"{'TOTAL':<20} {total_params:>15,} {100.0:>9.2f}%")
    
    # Detailed layer listing
    if detailed:
        print(f"\n{'='*80}")
        print(f"Detailed Layer Information")
        print(f"{'='*80}\n")
        
        for component in sorted(components.keys()):
            print(f"\n{component.upper()}:")
            print(f"{'Layer Name':<60} {'Parameters':>15} {'Shape':<20}")
            print("-" * 100)
            
            for layer in components[component]['layers']:
                shape_str = str(layer['shape'])
                trainable_mark = "" if layer['trainable'] else " (frozen)"
                print(f"{layer['name']:<60} {layer['params']:>15,} {shape_str:<20}{trainable_mark}")
    
    return results


def analyze_backbone_structure(model: nn.Module) -> Dict[str, Any]:
    """Analyze backbone architecture."""
    
    print(f"\n{'='*80}")
    print(f"Backbone Analysis")
    print(f"{'='*80}\n")
    
    backbone_info = {
        'type': 'Unknown',
        'depth': 'Unknown',
        'stages': []
    }
    
    # Try to identify backbone
    for name, module in model.named_modules():
        if 'backbone' in name.lower():
            if hasattr(module, 'depth'):
                backbone_info['depth'] = module.depth
            
            # Detect ResNet
            if 'resnet' in str(type(module)).lower() or 'presnet' in str(type(module)).lower():
                backbone_info['type'] = 'ResNet'
                
                # Count stages
                stage_params = defaultdict(int)
                for param_name, param in module.named_parameters():
                    if 'layer' in param_name or 'stage' in param_name:
                        # Extract stage number
                        for i in range(1, 5):
                            if f'layer{i}' in param_name or f'stage{i}' in param_name:
                                stage_params[f'stage{i}'] += param.numel()
                
                backbone_info['stages'] = [
                    {'name': stage, 'params': params}
                    for stage, params in sorted(stage_params.items())
                ]
    
    if backbone_info['type'] != 'Unknown':
        print(f"Backbone Type: {backbone_info['type']}")
        print(f"Depth: {backbone_info['depth']}")
        
        if backbone_info['stages']:
            print(f"\nStage-wise Parameters:")
            for stage in backbone_info['stages']:
                print(f"  {stage['name']}: {stage['params']:,} ({stage['params']/1e6:.2f}M)")
    
    return backbone_info


def analyze_encoder_decoder(model: nn.Module) -> Dict[str, Any]:
    """Analyze encoder and decoder structures."""
    
    print(f"\n{'='*80}")
    print(f"Encoder & Decoder Analysis")
    print(f"{'='*80}\n")
    
    encoder_info = {'params': 0, 'layers': []}
    decoder_info = {'params': 0, 'layers': [], 'num_queries': 'Unknown', 'num_decoder_layers': 'Unknown'}
    
    for name, module in model.named_modules():
        # Encoder analysis
        if 'encoder' in name.lower() and not 'decoder' in name.lower():
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                encoder_info['params'] += params
                encoder_info['layers'].append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': params
                })
        
        # Decoder analysis
        if 'decoder' in name.lower():
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                decoder_info['params'] += params
                decoder_info['layers'].append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': params
                })
            
            # Try to extract decoder config
            if hasattr(module, 'num_queries'):
                decoder_info['num_queries'] = module.num_queries
            if hasattr(module, 'num_decoder_layers'):
                decoder_info['num_decoder_layers'] = module.num_decoder_layers
    
    print(f"Encoder:")
    print(f"  Total Parameters: {encoder_info['params']:,} ({encoder_info['params']/1e6:.2f}M)")
    print(f"  Number of Modules: {len(encoder_info['layers'])}")
    
    print(f"\nDecoder:")
    print(f"  Total Parameters: {decoder_info['params']:,} ({decoder_info['params']/1e6:.2f}M)")
    print(f"  Number of Modules: {len(decoder_info['layers'])}")
    print(f"  Num Queries: {decoder_info['num_queries']}")
    print(f"  Num Decoder Layers: {decoder_info['num_decoder_layers']}")
    
    return {
        'encoder': encoder_info,
        'decoder': decoder_info
    }


def compare_with_baseline(model_params: int, model_type: str = 'RT-DETR-R18') -> None:
    """Compare model parameters with expected baseline."""
    
    print(f"\n{'='*80}")
    print(f"Comparison with Baseline")
    print(f"{'='*80}\n")
    
    baselines = {
        'RT-DETR-R18': 20_000_000,
        'RT-DETR-R34': 31_000_000,
        'RT-DETR-R50': 42_000_000,
        'RT-DETR-R101': 76_000_000,
    }
    
    if model_type in baselines:
        expected = baselines[model_type]
        diff = model_params - expected
        diff_pct = (diff / expected) * 100
        
        print(f"Model Type: {model_type}")
        print(f"Expected Parameters: {expected:,} ({expected/1e6:.2f}M)")
        print(f"Actual Parameters: {model_params:,} ({model_params/1e6:.2f}M)")
        print(f"Difference: {diff:+,} ({diff/1e6:+.2f}M, {diff_pct:+.1f}%)")
        
        if abs(diff_pct) > 10:
            print(f"\n⚠  WARNING: Parameter count differs by more than 10% from baseline!")
            print(f"   Possible reasons:")
            print(f"   - Different number of queries")
            print(f"   - Modified encoder/decoder architecture")
            print(f"   - Additional task-specific heads")
            print(f"   - Different backbone configuration")
    else:
        print(f"No baseline available for {model_type}")
        print(f"Actual Parameters: {model_params:,} ({model_params/1e6:.2f}M)")


def profile_memory_footprint(model: nn.Module) -> Dict[str, Any]:
    """Estimate model memory footprint."""
    
    print(f"\n{'='*80}")
    print(f"Memory Footprint Analysis")
    print(f"{'='*80}\n")
    
    # Calculate parameter memory
    param_memory = 0
    buffer_memory = 0
    
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_memory += buffer.numel() * buffer.element_size()
    
    total_memory = param_memory + buffer_memory
    
    memory_info = {
        'param_memory_mb': param_memory / (1024 * 1024),
        'buffer_memory_mb': buffer_memory / (1024 * 1024),
        'total_memory_mb': total_memory / (1024 * 1024),
    }
    
    print(f"Parameter Memory: {memory_info['param_memory_mb']:.2f} MB")
    print(f"Buffer Memory: {memory_info['buffer_memory_mb']:.2f} MB")
    print(f"Total Model Memory: {memory_info['total_memory_mb']:.2f} MB")
    
    # Estimate activation memory for common input sizes
    print(f"\nEstimated Peak Memory (including activations):")
    for size in [640, 800, 1024]:
        # Rough estimate: model memory + activation memory
        # Activation memory is roughly 2-3x model size for transformers
        est_activation = memory_info['total_memory_mb'] * 2.5
        est_total = memory_info['total_memory_mb'] + est_activation
        print(f"  @ {size}x{size}: ~{est_total:.2f} MB (model + activations)")
    
    return memory_info


def main(args):
    """Main profiling function."""
    
    print(f"\n{'='*80}")
    print(f"RT-DETR Model Profiler")
    print(f"{'='*80}\n")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load configuration
    cfg = YAMLConfig(args.config, resume=args.checkpoint)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
        print("Using EMA weights\n")
    else:
        state = checkpoint['model']
        print("Using model weights\n")
    
    # Load model
    cfg.model.load_state_dict(state)
    model = cfg.model
    model.eval()
    
    results = {
        'config': args.config,
        'checkpoint': args.checkpoint,
    }
    
    # 1. Layer-wise parameter analysis
    layer_analysis = analyze_layer_parameters(model, detailed=args.detailed)
    results['layer_analysis'] = layer_analysis
    
    # 2. Backbone structure
    backbone_info = analyze_backbone_structure(model)
    results['backbone'] = backbone_info
    
    # 3. Encoder/Decoder analysis
    enc_dec_info = analyze_encoder_decoder(model)
    results['encoder_decoder'] = enc_dec_info
    
    # 4. Compare with baseline
    compare_with_baseline(layer_analysis['total_params'], args.model_type)
    
    # 5. Memory footprint
    memory_info = profile_memory_footprint(model)
    results['memory'] = memory_info
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Profiling results saved to: {output_path}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RT-DETR Model Profiler')
    
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', '-r', type=str, required=True,
                        help='Path to checkpoint file')
    
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed layer-by-layer information')
    parser.add_argument('--model-type', type=str, default='RT-DETR-R18',
                        choices=['RT-DETR-R18', 'RT-DETR-R34', 'RT-DETR-R50', 'RT-DETR-R101'],
                        help='Model type for baseline comparison')
    
    parser.add_argument('--output', '-o', type=str, default='profiling_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    main(args)
