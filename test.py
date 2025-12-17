"""
Test script for Glaucoma Detection
"""
import torch
import argparse
import os
from utils.dataset import get_data_loaders
from utils.model import create_model
from utils.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description='Test Glaucoma Classifier')
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset-001',
        help='Path to dataset directory (root folder containing train/val/test)'
    )
    
    # ✅ Not required anymore; defaults to checkpoints/best_model.pt
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=os.path.join('checkpoints', 'best_model.pt'),
        help='Path to model checkpoint (default: checkpoints/best_model.pt)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for testing'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of data loader workers'
    )
    parser.add_argument(
        '--use_metadata',
        action='store_true',
        default=True,
        help='Use metadata (age, MD) in classification'
    )
    parser.add_argument(
        '--save_plots',
        action='store_true',
        help='Save confusion matrix plot'
    )
    
    args = parser.parse_args()
    
    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at: {checkpoint_path}\n"
            f"Make sure best_model.pt is saved in 'checkpoints/' "
            f"or pass --checkpoint /full/path/to/best_model.pt"
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders (we only need test_loader)
    print("Loading test dataset...")
    _, _, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        num_classes=2,
        pretrained=False,  # Not needed for inference
        use_metadata=args.use_metadata,
        device=device
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy (at save): {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    evaluator = Evaluator(model, test_loader, device)
    results = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(results)
    
    # Save confusion matrix plot
    if args.save_plots:
        out_path = 'test_confusion_matrix.png'
        evaluator.plot_confusion_matrix(results, save_path=out_path)
        print(f"Confusion matrix saved to {out_path}")

if __name__ == '__main__':
    main()
