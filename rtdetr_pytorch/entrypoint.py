
import argparse
import os
import sys
import yaml
import kagglehub
import glob
from pathlib import Path

def find_dataset_paths(dataset_root):
    """
    Attempts to find train/val image folders and annotation files 
    within the downloaded dataset directory.
    """
    dataset_root = Path(dataset_root)
    paths = {
        'train_img': None,
        'train_ann': None,
        'val_img': None,
        'val_ann': None
    }

    # Helper to find path case-insensitive
    def search(pattern, root=dataset_root):
        matches = list(root.rglob(pattern))
        return matches[0] if matches else None

    # Search strategy
    # Images: Look for directories 'train', 'val'
    # Annotations: Look for json files 'train', 'val'
    
    # 1. Image Folders
    # Common patterns: 'train', 'train2017', 'VisDrone2019-DET-train', etc.
    for p in root.rglob("*"):
        if p.is_dir():
            name = p.name.lower()
            if 'train' in name and paths['train_img'] is None:
                paths['train_img'] = str(p)
            elif ('val' in name or 'test' in name) and paths['val_img'] is None:
                 # Prefer val over test if both exist, but take what we get
                paths['val_img'] = str(p)
    
    # 2. Annotation Files
    # Look for .json files
    json_files = list(root.rglob("*.json"))
    for jf in json_files:
        name = jf.name.lower()
        if 'train' in name and paths['train_ann'] is None:
            paths['train_ann'] = str(jf)
        elif 'val' in name and paths['val_ann'] is None:
            paths['val_ann'] = str(jf)

    return paths

def main():
    parser = argparse.ArgumentParser()
    # Accept all arguments that tools/train.py accepts, plus --dataset_handle
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset_handle', type=str, default="duwipurnamasidik/visdrone-2019-coco-format", help='Kaggle dataset handle')
    parser.add_argument('--download_dataset', action='store_true', help='Download dataset using kagglehub')
    
    # Parse known args, keep the rest for train.py
    args, unknown_args = parser.parse_known_args()

    print(f"Downloading dataset: {args.dataset_handle}...")
    try:
        dataset_path = kagglehub.dataset_download(args.dataset_handle)
        print(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        sys.exit(1)

    # Find paths
    paths = find_dataset_paths(dataset_path)
    print("Found dataset paths:", paths)

    if not all(paths.values()):
        print("Warning: Could not automatically detect all train/val paths. Please check the dataset structure.")
        # Proceeding might fail, but let's try or error out?
        # Let's inspect what we missed
        missing = [k for k, v in paths.items() if v is None]
        print(f"Missing: {missing}")

    # Load config
    with open(args.config, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    # Helper to update nested dict
    def update_config(cfg, paths):
        if 'train_dataloader' in cfg and 'dataset' in cfg['train_dataloader']:
            if paths['train_img']: cfg['train_dataloader']['dataset']['img_folder'] = paths['train_img']
            if paths['train_ann']: cfg['train_dataloader']['dataset']['ann_file'] = paths['train_ann']
        
        if 'val_dataloader' in cfg and 'dataset' in cfg['val_dataloader']:
             if paths['val_img']: cfg['val_dataloader']['dataset']['img_folder'] = paths['val_img']
             if paths['val_ann']: cfg['val_dataloader']['dataset']['ann_file'] = paths['val_ann']
        
        return cfg

    config = update_config(config, paths)

    # Write temp config
    temp_config_path = "temp_config.yml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Created temporary config with updated paths: {temp_config_path}")

    # Build command
    cmd = [sys.executable, "tools/train.py", "-c", temp_config_path] + unknown_args
    
    print(f"Executing: {' '.join(cmd)}")
    os.execv(sys.executable, cmd)

if __name__ == "__main__":
    main()
