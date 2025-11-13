"""
Training script for text-conditioned UNet separation model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from model import TextConditionedUNet, TextAgnosticUNet
from dataset import ESC50MixtureDataset, collate_fn
from config import SAMPLE_RATE, STFT_N_FFT, STFT_HOP


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        mix_mag = batch['mix_mag'].to(device)
        text_emb = batch['text_emb'].to(device)
        target_mask = batch['target_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        if isinstance(model, TextConditionedUNet):
            pred_mask = model(mix_mag, text_emb)
        else:  # TextAgnosticUNet
            pred_mask = model(mix_mag)
        
        # Compute loss
        loss = criterion(pred_mask, target_mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            mix_mag = batch['mix_mag'].to(device)
            text_emb = batch['text_emb'].to(device)
            target_mask = batch['target_mask'].to(device)
            
            # Forward pass
            if isinstance(model, TextConditionedUNet):
                pred_mask = model(mix_mag, text_emb)
            else:
                pred_mask = model(mix_mag)
            
            # Compute loss
            loss = criterion(pred_mask, target_mask)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    if args.baseline:
        print("Training text-agnostic baseline model...")
        model = TextAgnosticUNet()
        model_name = "baseline_unet"
    else:
        print("Training text-conditioned model...")
        model = TextConditionedUNet(clap_dim=1024)
        model_name = "text_conditioned_unet"
    
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ESC50MixtureDataset(
        esc50_path=args.esc50_path,
        split="train",
        train_fold_ids=args.train_folds,
        val_fold_ids=args.val_folds,
        sr=SAMPLE_RATE,
        n_fft=STFT_N_FFT,
        hop=STFT_HOP,
        snr_range=(args.min_snr, args.max_snr),
        cache_embeddings=True
    )
    
    val_dataset = ESC50MixtureDataset(
        esc50_path=args.esc50_path,
        split="val",
        train_fold_ids=args.train_folds,
        val_fold_ids=args.val_folds,
        sr=SAMPLE_RATE,
        n_fft=STFT_N_FFT,
        hop=STFT_HOP,
        snr_range=(args.min_snr, args.max_snr),
        cache_embeddings=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # L1 loss for mask prediction
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / f"{model_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"{model_name}_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint")
        
        print()
    
    # Save final model
    final_path = output_dir / f"{model_name}_final.pth"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
    }, final_path)
    
    # Save training history
    history_path = output_dir / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text-conditioned UNet for audio separation")
    
    # Data
    parser.add_argument('--esc50-path', type=str, 
                       default='../data/ESC-50-master/ESC-50-master',
                       help='Path to ESC-50 dataset')
    parser.add_argument('--train-folds', type=int, nargs='+', default=[1, 2, 3, 4],
                       help='Fold IDs for training')
    parser.add_argument('--val-folds', type=int, nargs='+', default=[5],
                       help='Fold IDs for validation')
    parser.add_argument('--min-snr', type=float, default=-5.0,
                       help='Minimum SNR for mixing (dB)')
    parser.add_argument('--max-snr', type=float, default=5.0,
                       help='Maximum SNR for mixing (dB)')
    
    # Model
    parser.add_argument('--baseline', action='store_true',
                       help='Train text-agnostic baseline instead of text-conditioned model')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of dataloader workers')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU training')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='../checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)
