"""
Comprehensive evaluation script for Glaucoma Detection
Evaluates on both validation and test sets
"""
import torch
import argparse
import os
from utils.dataset import get_data_loaders
from utils.model import create_model
from utils.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description='Evaluate Glaucoma Classifier')
    parser.add_argument('--data_dir', type=str, default='dataset-001',
                        help='Path to dataset directory')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',   # <-- default best model
        help='Path to model checkpoint'
    )
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--use_metadata', action='store_true', default=True,
                        help='Use metadata (age, MD) in classification')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save confusion matrix plots')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    print(f"Using checkpoint: {args.checkpoint}")
    
    # Create data loaders
    print("Loading datasets...")
    _, val_loader, test_loader = get_data_loaders(
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
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        print(f"Training validation accuracy: {checkpoint['val_acc']:.2f}%")
    print("\n" + "=" * 50)
    
    # Evaluate on validation set
    print("EVALUATING ON VALIDATION SET")
    print("=" * 50)
    val_evaluator = Evaluator(model, val_loader, device)
    val_results = val_evaluator.evaluate()
    val_evaluator.print_results(val_results)
    
    if args.save_plots:
        val_evaluator.plot_confusion_matrix(val_results, save_path='val_confusion_matrix.png')
    
    print("\n" + "=" * 50)
    
    # Evaluate on test set
    print("EVALUATING ON TEST SET")
    print("=" * 50)
    test_evaluator = Evaluator(model, test_loader, device)
    test_results = test_evaluator.evaluate()
    test_evaluator.print_results(test_results)
    
    if args.save_plots:
        test_evaluator.plot_confusion_matrix(test_results, save_path='test_confusion_matrix.png')
    
    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    val_auc = f"{val_results['auc_roc']:.4f}" if val_results['auc_roc'] is not None else 'N/A'
    test_auc = f"{test_results['auc_roc']:.4f}" if test_results['auc_roc'] is not None else 'N/A'
    print(f"Validation Set - Accuracy: {val_results['accuracy']*100:.2f}%, "
          f"Precision: {val_results['precision']:.4f}, "
          f"Recall: {val_results['recall']:.4f}, "
          f"F1: {val_results['f1_score']:.4f}, "
          f"AUC-ROC: {val_auc}")
    print(f"Test Set - Accuracy: {test_results['accuracy']*100:.2f}%, "
          f"Precision: {test_results['precision']:.4f}, "
          f"Recall: {test_results['recall']:.4f}, "
          f"F1: {test_results['f1_score']:.4f}, "
          f"AUC-ROC: {test_auc}")
    print("=" * 50)

if __name__ == '__main__':
    main()
