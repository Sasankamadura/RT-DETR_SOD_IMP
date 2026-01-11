"""
Detailed COCO Evaluation Script for RT-DETR
Enhanced evaluation with per-class metrics and visualizations

Usage:
    python tools/detailed_coco_eval.py --config <config_path> --checkpoint <checkpoint_path>
    python tools/detailed_coco_eval.py --config <config_path> --checkpoint <checkpoint_path> --save-predictions
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.core import YAMLConfig
from src.data import get_coco_api_from_dataset
from src.solver.det_engine import evaluate
import src.misc.dist as dist


def extract_per_class_metrics(coco_eval: COCOeval, coco_gt: COCO) -> Dict[str, Any]:
    """Extract per-class AP metrics from COCO evaluation."""
    
    print(f"\n{'='*80}")
    print(f"Per-Class Metrics")
    print(f"{'='*80}\n")
    
    # Get category information
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = {cat['id']: cat['name'] for cat in cats}
    
    per_class_results = {}
    
    # Compute per-category AP
    print(f"{'Category ID':<12} {'Category Name':<30} {'AP@0.5:0.95':>12} {'AP@0.5':>10} {'AP@0.75':>10}")
    print("-" * 80)
    
    for cat_id in sorted(cat_names.keys()):
        cat_name = cat_names[cat_id]
        
        # Set category filter
        coco_eval.params.catIds = [cat_id]
        
        # Recompute stats for this category
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        # Extract metrics
        stats = coco_eval.stats
        
        per_class_results[cat_id] = {
            'name': cat_name,
            'ap_50_95': float(stats[0]),
            'ap_50': float(stats[1]),
            'ap_75': float(stats[2]),
            'ap_small': float(stats[3]),
            'ap_medium': float(stats[4]),
            'ap_large': float(stats[5]),
            'ar_1': float(stats[6]),
            'ar_10': float(stats[7]),
            'ar_100': float(stats[8]),
            'ar_small': float(stats[9]),
            'ar_medium': float(stats[10]),
            'ar_large': float(stats[11]),
        }
        
        print(f"{cat_id:<12} {cat_name:<30} {stats[0]:>12.3f} {stats[1]:>10.3f} {stats[2]:>10.3f}")
    
    # Reset to all categories
    coco_eval.params.catIds = []
    
    return per_class_results


def analyze_size_distribution(coco_gt: COCO) -> Dict[str, Any]:
    """Analyze object size distribution in dataset."""
    
    print(f"\n{'='*80}")
    print(f"Object Size Distribution")
    print(f"{'='*80}\n")
    
    anns = coco_gt.loadAnns(coco_gt.getAnnIds())
    
    size_counts = {'small': 0, 'medium': 0, 'large': 0}
    size_areas = {'small': [], 'medium': [], 'large': []}
    
    for ann in anns:
        area = ann['area']
        
        if area < 32**2:
            size_counts['small'] += 1
            size_areas['small'].append(area)
        elif area < 96**2:
            size_counts['medium'] += 1
            size_areas['medium'].append(area)
        else:
            size_counts['large'] += 1
            size_areas['large'].append(area)
    
    total = sum(size_counts.values())
    
    print(f"{'Size Category':<15} {'Count':>10} {'Percentage':>12} {'Avg Area':>12}")
    print("-" * 55)
    
    for size_cat in ['small', 'medium', 'large']:
        count = size_counts[size_cat]
        pct = (count / total * 100) if total > 0 else 0
        avg_area = np.mean(size_areas[size_cat]) if size_areas[size_cat] else 0
        
        print(f"{size_cat.capitalize():<15} {count:>10} {pct:>11.2f}% {avg_area:>11.1f}")
    
    print(f"{'Total':<15} {total:>10} {100.0:>11.2f}%")
    
    return {
        'counts': size_counts,
        'percentages': {k: (v/total*100 if total > 0 else 0) for k, v in size_counts.items()},
        'avg_areas': {k: (np.mean(v) if v else 0) for k, v in size_areas.items()}
    }


def analyze_predictions_by_size(results: List[Dict], coco_gt: COCO) -> Dict[str, Any]:
    """Analyze prediction performance by object size."""
    
    print(f"\n{'='*80}")
    print(f"Detection Performance by Size")
    print(f"{'='*80}\n")
    
    # Group predictions by size
    size_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    # This is a simplified analysis - full implementation would require
    # matching predictions to ground truth
    
    print("Note: Full detection performance analysis requires ground truth matching")
    print("Use the COCO evaluation metrics above for accurate size-based performance")
    
    return {}


def summarize_results(coco_eval: COCOeval) -> Dict[str, float]:
    """Summarize COCO evaluation results."""
    
    print(f"\n{'='*80}")
    print(f"Overall Detection Metrics Summary")
    print(f"{'='*80}\n")
    
    stats = coco_eval.stats
    
    metrics = {
        'AP@0.5:0.95': float(stats[0]),
        'AP@0.5': float(stats[1]),
        'AP@0.75': float(stats[2]),
        'AP_small': float(stats[3]),
        'AP_medium': float(stats[4]),
        'AP_large': float(stats[5]),
        'AR@1': float(stats[6]),
        'AR@10': float(stats[7]),
        'AR@100': float(stats[8]),
        'AR_small': float(stats[9]),
        'AR_medium': float(stats[10]),
        'AR_large': float(stats[11]),
    }
    
    # Print formatted summary
    print("Average Precision (AP):")
    print(f"  AP @ IoU=0.50:0.95: {metrics['AP@0.5:0.95']:.3f}")
    print(f"  AP @ IoU=0.50:      {metrics['AP@0.5']:.3f}")
    print(f"  AP @ IoU=0.75:      {metrics['AP@0.75']:.3f}")
    print(f"\nAP by Object Size:")
    print(f"  AP (small):         {metrics['AP_small']:.3f}")
    print(f"  AP (medium):        {metrics['AP_medium']:.3f}")
    print(f"  AP (large):         {metrics['AP_large']:.3f}")
    print(f"\nAverage Recall (AR):")
    print(f"  AR @ 1 det:         {metrics['AR@1']:.3f}")
    print(f"  AR @ 10 dets:       {metrics['AR@10']:.3f}")
    print(f"  AR @ 100 dets:      {metrics['AR@100']:.3f}")
    print(f"\nAR by Object Size:")
    print(f"  AR (small):         {metrics['AR_small']:.3f}")
    print(f"  AR (medium):        {metrics['AR_medium']:.3f}")
    print(f"  AR (large):         {metrics['AR_large']:.3f}")
    
    return metrics


def generate_recommendations(metrics: Dict[str, float], per_class: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on performance."""
    
    print(f"\n{'='*80}")
    print(f"Performance Analysis & Recommendations")
    print(f"{'='*80}\n")
    
    recommendations = []
    
    # Analyze small object performance
    if metrics['AP_small'] < metrics['AP@0.5:0.95'] * 0.7:
        rec = "⚠  Small object AP is significantly lower than overall AP"
        recommendations.append(rec)
        print(rec)
        print("   → Consider: P2 fusion, multi-scale training, higher resolution input")
    
    # Analyze large object performance
    if metrics['AP_large'] > metrics['AP@0.5:0.95'] * 1.3:
        rec = "✓ Strong performance on large objects"
        recommendations.append(rec)
        print(rec)
    
    # Analyze recall
    if metrics['AR@100'] - metrics['AP@0.5:0.95'] > 0.15:
        rec = "⚠  Large gap between recall and precision"
        recommendations.append(rec)
        print(rec)
        print("   → Consider: adjusting confidence thresholds, NMS tuning")
    
    # Analyze class imbalance
    if per_class:
        aps = [v['ap_50_95'] for v in per_class.values()]
        if len(aps) > 1:
            ap_std = np.std(aps)
            if ap_std > 0.2:
                rec = f"⚠  High variance in per-class AP (std={ap_std:.3f})"
                recommendations.append(rec)
                print(rec)
                print("   → Consider: class-balanced sampling, per-class loss weighting")
    
    # Overall performance assessment
    if metrics['AP@0.5:0.95'] > 0.3:
        print("\n✓ Overall good detection performance")
    elif metrics['AP@0.5:0.95'] > 0.2:
        print("\n• Moderate detection performance - room for improvement")
    else:
        print("\n⚠  Lower than expected performance - consider reviewing training setup")
    
    return recommendations


def main(args):
    """Main evaluation function."""
    
    print(f"\n{'='*80}")
    print(f"Detailed COCO Evaluation for RT-DETR")
    print(f"{'='*80}\n")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Initialize distributed mode
    dist.init_distributed()
    
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
    
    print(f"Checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}\n")
    
    # Load model
    cfg.model.load_state_dict(state)
    model = cfg.model.deploy() if hasattr(cfg.model, 'deploy') else cfg.model
    model = model.to(args.device)
    model.eval()
    
    # Create data loader
    val_dataloader = cfg.val_dataloader
    
    # Get COCO API
    base_ds = get_coco_api_from_dataset(val_dataloader.dataset)
    
    # Run evaluation
    print(f"Running evaluation on validation set...")
    print(f"{'='*80}\n")
    
    test_stats, coco_evaluator = evaluate(
        model, 
        cfg.criterion, 
        cfg.postprocessor,
        val_dataloader, 
        base_ds, 
        args.device, 
        None  # output_dir
    )
    
    # Get COCO eval object
    coco_eval = coco_evaluator.coco_eval['bbox']
    
    # Collect results
    results = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'epoch': checkpoint.get('epoch', 'unknown'),
    }
    
    # 1. Overall metrics
    overall_metrics = summarize_results(coco_eval)
    results['overall_metrics'] = overall_metrics
    
    # 2. Per-class metrics
    if not args.skip_per_class:
        per_class_metrics = extract_per_class_metrics(coco_eval, base_ds)
        results['per_class_metrics'] = per_class_metrics
    else:
        per_class_metrics = {}
    
    # 3. Size distribution analysis
    size_distribution = analyze_size_distribution(base_ds)
    results['size_distribution'] = size_distribution
    
    # 4. Generate recommendations
    recommendations = generate_recommendations(overall_metrics, per_class_metrics)
    results['recommendations'] = recommendations
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Detailed evaluation results saved to: {output_path}")
        print(f"{'='*80}\n")
    
    # Save predictions if requested
    if args.save_predictions and dist.is_main_process():
        pred_path = Path(args.output).parent / 'predictions.json'
        
        # Save COCO-format predictions
        if coco_evaluator.coco_eval is not None:
            predictions = coco_evaluator.coco_eval['bbox'].cocoDt.dataset['annotations']
            
            with open(pred_path, 'w') as f:
                json.dump(predictions, f)
            
            print(f"Predictions saved to: {pred_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detailed COCO Evaluation for RT-DETR')
    
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', '-r', type=str, required=True,
                        help='Path to checkpoint file')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    parser.add_argument('--skip-per-class', action='store_true',
                        help='Skip per-class metric computation (faster)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions in COCO format')
    
    parser.add_argument('--output', '-o', type=str, default='detailed_evaluation_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    main(args)
