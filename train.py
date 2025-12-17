"""
Training script for Glaucoma Detection
"""
import torch
import argparse
from utils.dataset import get_data_loaders
from utils.model import create_model
from utils.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Train Glaucoma Classifier')
    parser.add_argument('--data_dir', type=str, default='dataset-001',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--use_metadata', action='store_true', default=True,
                        help='Use metadata (age, MD) in classification')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained weights')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        num_classes=2,
        pretrained=not args.no_pretrained,
        use_metadata=args.use_metadata,
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )
    
    # Train
    history, best_model_path = trainer.train()
    
    print(f"\nTraining completed! Best model saved at: {best_model_path}")

if __name__ == '__main__':
    main()

