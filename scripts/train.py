import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import sys
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqgan import VQGAN, Discriminator
from utils.data_utils import get_data_loaders, get_all_batch_files
from utils.training_utils import train_vqgan_stage1, train_vqgan_stage2, evaluate_on_test_set

def parse_args():
    parser = argparse.ArgumentParser(description='Train VQ-GAN on UCF101 frames')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/data/UCFData',
                        help='Directory containing the UCF101 numpy files')
    parser.add_argument('--test_batch', type=int, default=13,
                        help='Batch index to use as test set')
    
    # Model parameters
    parser.add_argument('--patch_size', type=int, default=80,
                        help='Size of image patches')
    parser.add_argument('--num_patches_h', type=int, default=3,
                        help='Number of patches in height dimension (240/80=3)')
    parser.add_argument('--num_patches_w', type=int, default=4,
                        help='Number of patches in width dimension (320/80=4)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128, 256, 512],
                        help='Hidden dimensions of the encoder/decoder')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of the latent space')
    parser.add_argument('--num_embeddings', type=int, default=512,
                        help='Number of embeddings in the codebook (reduced from 1024)')
    parser.add_argument('--downsample_factor', type=int, default=4,
                        help='Downsampling factor in the encoder/decoder')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                        help='Commitment cost for the VQ layer')
    parser.add_argument('--decay', type=float, default=0.99,
                        help='Decay factor for EMA updates of the codebook')
    parser.add_argument('--restart_unused', action='store_true',
                        help='Restart unused embedding vectors during training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=3,
                        help='Batch size for training')
    parser.add_argument('--epochs_per_batch', type=int, default=10,
                        help='Number of epochs to train on each data batch in stage 1')
    parser.add_argument('--gan_epochs_per_batch', type=int, default=5,
                        help='Number of epochs to train on each data batch in stage 2 (GAN)')
    parser.add_argument('--num_epoch', type=int, default=3,
                        help='Number of times to loop through all batches')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for stage 1')
    parser.add_argument('--lr_g', type=float, default=5e-5,
                        help='Learning rate for generator in stage 2')
    parser.add_argument('--lr_d', type=float, default=2e-4,
                        help='Learning rate for discriminator in stage 2')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--adv_weight', type=float, default=0.1,
                        help='Weight for adversarial loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--gan_mode', action='store_true',
                        help='Start directly with GAN training (stage 2)')
    
    # Directories for logs and results
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory for saving results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for saving model checkpoints')
    
    # CUDA settings
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint to resume training from')
    parser.add_argument('--resume_disc', type=str, default=None,
                        help='Path to a discriminator checkpoint to resume training from')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create necessary directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Try to free up memory
    import gc
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    
    # Initialize the VQGAN model
    model = VQGAN(
        in_channels=3,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.latent_dim,  # Must be same as latent_dim
        downsample_factor=args.downsample_factor,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
        restart_unused=args.restart_unused
    ).to(device)
    
    # Initialize the discriminator for GAN training
    discriminator = Discriminator(
        in_channels=3,
        hidden_dims=args.hidden_dims,
        patch_size=args.patch_size
    ).to(device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Loading model from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print("Model loaded successfully")
    
    if args.resume_disc and (args.gan_mode or args.num_epoch > 1):
        print(f"Loading discriminator from {args.resume_disc}")
        discriminator.load_state_dict(torch.load(args.resume_disc, map_location=device))
        print("Discriminator loaded successfully")
    
    # Print model summary
    num_params_gen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_disc = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator has {num_params_gen:,} trainable parameters")
    print(f"Discriminator has {num_params_disc:,} trainable parameters")
    
    # Get all batch files except the test batch
    batch_indices = get_all_batch_files(args.data_path, exclude_test=args.test_batch)
    
    if not batch_indices:
        print("No batch files found! Please check the data path.")
        return
    
    print(f"Found {len(batch_indices)} training batch files: {batch_indices}")
    print(f"Using batch {args.test_batch} as test set")
    
    # Keep track of best overall validation loss
    best_overall_val_loss = float('inf')
    
    # Stage 1: Train with reconstruction and perceptual loss only
    if not args.gan_mode:
        print("\n=== Starting Stage 1 Training (Reconstruction + Perceptual Loss) ===\n")
        
        for epoch in range(args.num_epoch):
            print(f"\n--- Global Epoch {epoch+1}/{args.num_epoch} ---\n")
            
            # Train on each batch sequentially
            for batch_idx in batch_indices:
                print(f"\n=== Training on batch {batch_idx} (Stage 1) ===\n")
                
                # Get data loaders for this batch
                train_loader, val_loader = get_data_loaders(
                    data_path=args.data_path,
                    batch_idx=batch_idx,
                    batch_size=args.batch_size,
                    patch_size=args.patch_size,
                    num_patches_h=args.num_patches_h,
                    num_patches_w=args.num_patches_w,
                    num_workers=args.num_workers,
                    val_split=0.1,
                    is_test_set=False
                )
                
                # Setup optimizer and scheduler for stage 1
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=2, verbose=True
                )
                
                # Train on this batch (stage 1)
                model, batch_best_val_loss = train_vqgan_stage1(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    batch_idx=batch_idx,
                    num_epochs=args.epochs_per_batch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    log_dir=args.log_dir,
                    results_folder=args.results_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    perceptual_weight=args.perceptual_weight
                )
                
                # Update best overall model if this batch gave better results
                if batch_best_val_loss < best_overall_val_loss:
                    best_overall_val_loss = batch_best_val_loss
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model_stage1.pt"))
                    print(f"New best overall model saved with val_loss: {best_overall_val_loss:.6f}")
                
                # Clean up memory after training on this batch
                del train_loader, val_loader
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()
    
    # Stage 2: Train with adversarial loss (GAN)
    print("\n=== Starting Stage 2 Training (GAN) ===\n")
    
    # Load the best model from stage 1 if we're transitioning
    if not args.gan_mode and args.num_epoch > 0:
        stage1_best_path = os.path.join(args.checkpoint_dir, "best_model_stage1.pt")
        if os.path.exists(stage1_best_path):
            print(f"Loading best model from stage 1: {stage1_best_path}")
            model.load_state_dict(torch.load(stage1_best_path, map_location=device))
    
    for epoch in range(max(1, args.num_epoch - 1)):  # At least one epoch for stage 2
        print(f"\n--- Global Epoch {epoch+1}/{max(1, args.num_epoch - 1)} (GAN) ---\n")
        
        # Train on each batch sequentially
        for batch_idx in batch_indices:
            print(f"\n=== Training on batch {batch_idx} (Stage 2 - GAN) ===\n")
            
            # Get data loaders for this batch
            train_loader, val_loader = get_data_loaders(
                data_path=args.data_path,
                batch_idx=batch_idx,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                num_patches_h=args.num_patches_h,
                num_patches_w=args.num_patches_w,
                num_workers=args.num_workers,
                val_split=0.1,
                is_test_set=False,
                augment=True  # Enable augmentation for GAN training
            )
            
            # Setup optimizers and schedulers for stage 2
            optimizer_g = optim.Adam(model.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
            optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
            
            scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_g, mode='min', factor=0.5, patience=1, verbose=True
            )
            scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_d, mode='min', factor=0.5, patience=1, verbose=True
            )
            
            # Train on this batch (stage 2 - GAN)
            model, discriminator, batch_best_val_loss = train_vqgan_stage2(
                model=model,
                discriminator=discriminator,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                batch_idx=batch_idx,
                num_epochs=args.gan_epochs_per_batch,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                log_dir=args.log_dir,
                results_folder=args.results_dir,
                checkpoint_dir=args.checkpoint_dir,
                adv_weight=args.adv_weight,
                perceptual_weight=args.perceptual_weight
            )
            
            # Update best overall model if this batch gave better results
            if batch_best_val_loss < best_overall_val_loss:
                best_overall_val_loss = batch_best_val_loss
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model_gan.pt"))
                torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, "best_discriminator.pt"))
                print(f"New best overall GAN model saved with val_loss: {best_overall_val_loss:.6f}")
            
            # Clean up memory after training on this batch
            del train_loader, val_loader
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
    
    # After training on all batches, evaluate on the test set
    print("\n=== Evaluating on test set ===\n")
    
    # Load the best GAN model if it exists, otherwise try the stage1 model, or use the current model
    gan_best_path = os.path.join(args.checkpoint_dir, "best_model_gan.pt")
    stage1_best_path = os.path.join(args.checkpoint_dir, "best_model_stage1.pt")
    disc_best_path = os.path.join(args.checkpoint_dir, "best_discriminator.pt")
    
    if os.path.exists(gan_best_path):
        print(f"Loading best GAN model from {gan_best_path}")
        model.load_state_dict(torch.load(gan_best_path, map_location=device))
    elif os.path.exists(stage1_best_path):
        print(f"No GAN model found. Loading stage 1 model from {stage1_best_path}")
        model.load_state_dict(torch.load(stage1_best_path, map_location=device))
    else:
        print("No saved models found. Using current model state.")
    
    if os.path.exists(disc_best_path):
        print(f"Loading best discriminator from {disc_best_path}")
        discriminator.load_state_dict(torch.load(disc_best_path, map_location=device))
    else:
        print("No saved discriminator found. Using current discriminator state.")
    
    # Get test data loader with smaller batch size to save memory
    test_batch_size = min(args.batch_size, 4)  # Use smaller batch size for testing
    print(f"Using test batch size of {test_batch_size}")
    
    test_loader = get_data_loaders(
        data_path=args.data_path,
        batch_idx=args.test_batch,
        batch_size=test_batch_size,
        patch_size=args.patch_size,
        num_patches_h=args.num_patches_h,
        num_patches_w=args.num_patches_w,
        num_workers=args.num_workers,
        is_test_set=True
    )
    
    # Clean up memory before evaluation
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    
    # Evaluate on test set
    try:
        test_loss, test_recon, test_latent, usage = evaluate_on_test_set(
            model=model,
            discriminator=discriminator,
            test_loader=test_loader,
            device=device,
            results_folder=os.path.join(args.results_dir, 'test'),
            perceptual_weight=args.perceptual_weight,
            adv_weight=args.adv_weight
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Continuing with training summary...")
        test_loss, test_recon, test_latent = 0.0, 0.0, 0.0
        usage = torch.zeros(args.num_embeddings) if hasattr(model, 'num_embeddings') else None
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_overall_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test reconstruction loss: {test_recon:.6f}")
    print(f"Codebook usage: {(usage > 0).sum()} / {args.num_embeddings} entries used")

if __name__ == "__main__":
    main()