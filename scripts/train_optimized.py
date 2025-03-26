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

# Force matplotlib to use Agg backend
import matplotlib
matplotlib.use('Agg')

from models.vqgan_optimized import OptimizedVQGAN, Discriminator
from utils.data_utils import get_data_loaders, get_all_batch_files
from utils.training_utils import train_vqgan_stage1, train_vqgan_stage2, evaluate_on_test_set

def parse_args():
    parser = argparse.ArgumentParser(description='Train optimized VQ-GAN on UCF101 frames')
    
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
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Hidden dimensions of the encoder/decoder (reduced from original)')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Dimension of the latent space (reduced from original)')
    parser.add_argument('--num_embeddings', type=int, default=256,
                        help='Number of embeddings in the codebook (reduced from original)')
    parser.add_argument('--downsample_factor', type=int, default=4,
                        help='Downsampling factor in the encoder/decoder')
    parser.add_argument('--commitment_cost', type=float, default=0.5,
                        help='Commitment cost for the VQ layer')
    parser.add_argument('--decay', type=float, default=0.99,
                        help='Decay factor for EMA updates of the codebook')
    parser.add_argument('--restart_unused', action='store_true',
                        help='Restart unused embedding vectors during training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of steps to accumulate gradients over')
    parser.add_argument('--epochs_per_batch', type=int, default=5,
                        help='Number of epochs to train on each data batch in stage 1')
    parser.add_argument('--gan_epochs_per_batch', type=int, default=3,
                        help='Number of epochs to train on each data batch in stage 2 (GAN)')
    parser.add_argument('--num_epoch', type=int, default=2,
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
    parser.add_argument('--num_workers', type=int, default=2,  # Reduced from 4
                        help='Number of workers for data loading')
    parser.add_argument('--gan_mode', action='store_true',
                        help='Start directly with GAN training (stage 2)')
    
    # Performance optimization
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Enable mixed precision training')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                        help='Frequency of saving checkpoints (in epochs)')
    parser.add_argument('--vis_freq', type=int, default=5, 
                        help='Frequency of saving visualizations (in epochs)')
    parser.add_argument('--cache_mode', action='store_true', default=False,
                        help='Enable caching of dataset on GPU')
    
    # Directories for logs and results
    parser.add_argument('--log_dir', type=str, default='logs_optimized',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--results_dir', type=str, default='results_optimized',
                        help='Directory for saving results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_optimized',
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
    parser.add_argument('--resume', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/checkpoints_optimized/best_model_gan.pt',
                        help='Path to a checkpoint to resume training from')
    parser.add_argument('--resume_disc', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/checkpoints_optimized/best_discriminator.pt',
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

def train_stage1_with_accumulation(model, train_loader, val_loader, device, 
                                  optimizer, scheduler, args, batch_idx, writer):
    """
    Train the model using gradient accumulation for memory efficiency
    """
    import lpips
    import matplotlib.pyplot as plt
    from utils.training_utils import save_image_grid, visualize_codebook_usage
    from torch.utils.tensorboard import SummaryWriter
    
    # Setup perceptual loss
    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    
    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == 'cuda' else None
    
    # Create folders for results
    batch_results_folder = os.path.join(args.results_dir, f'batch_{batch_idx}')
    os.makedirs(os.path.join(batch_results_folder, 'reconstructions'), exist_ok=True)
    os.makedirs(os.path.join(batch_results_folder, 'codebook'), exist_ok=True)
    
    # Initialize losses
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs_per_batch):
        model.train()
        train_losses = {
            'total': 0.0,
            'recon': 0.0,
            'vq': 0.0,
            'perceptual': 0.0
        }
        
        # Reset gradients at the beginning of each epoch
        optimizer.zero_grad()
        
        # Use gradient accumulation to effectively increase batch size
        progress_bar = tqdm(enumerate(train_loader), 
                           desc=f"Epoch {epoch+1}/{args.epochs_per_batch}",
                           total=len(train_loader))
        
        for i, (images, _) in progress_bar:
            images = images.to(device, non_blocking=True)
            
            # Calculate loss
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                    
                    # Calculate losses
                    recon_loss = nn.functional.mse_loss(reconstructions, images)
                    p_loss = perceptual_loss(reconstructions * 2.0 - 1.0, images * 2.0 - 1.0).mean() if args.perceptual_weight > 0 else torch.tensor(0.0, device=device)
                    
                    # Combined loss
                    loss = recon_loss + vq_loss + args.perceptual_weight * p_loss
                    
                    # Normalize loss by accumulation steps
                    loss = loss / args.gradient_accumulation_steps
                
                # Accumulate gradients
                scaler.scale(loss).backward()
                
                # Step optimizer after accumulation
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Forward pass (no mixed precision)
                reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                
                # Calculate losses
                recon_loss = nn.functional.mse_loss(reconstructions, images)
                p_loss = perceptual_loss(reconstructions * 2.0 - 1.0, images * 2.0 - 1.0).mean() if args.perceptual_weight > 0 else torch.tensor(0.0, device=device)
                
                # Combined loss
                loss = recon_loss + vq_loss + args.perceptual_weight * p_loss
                
                # Normalize loss by accumulation steps
                loss = loss / args.gradient_accumulation_steps
                
                # Accumulate gradients
                loss.backward()
                
                # Step optimizer after accumulation
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update train losses (without the accumulation scaling)
            train_losses['total'] += loss.item() * args.gradient_accumulation_steps
            train_losses['recon'] += recon_loss.item()
            train_losses['vq'] += vq_loss.item()
            train_losses['perceptual'] += p_loss.item() if args.perceptual_weight > 0 else 0
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'vq': f"{vq_loss.item():.4f}",
                'p_loss': f"{p_loss.item():.4f}" if args.perceptual_weight > 0 else "0.0000"
            })
            
            # Clear memory
            if device.type == 'cuda' and (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
        
        # Calculate average train losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
            writer.add_scalar(f'train/{key}_loss', train_losses[key], epoch)
        
        # Validation
        model.eval()
        val_losses = {
            'total': 0.0,
            'recon': 0.0,
            'vq': 0.0,
            'perceptual': 0.0
        }
        
        with torch.no_grad():
            for i, (images, _) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                
                # Use mixed precision for validation as well
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                        
                        # Calculate losses
                        recon_loss = nn.functional.mse_loss(reconstructions, images)
                        p_loss = perceptual_loss(reconstructions * 2.0 - 1.0, images * 2.0 - 1.0).mean() if args.perceptual_weight > 0 else torch.tensor(0.0, device=device)
                        
                        # Combined loss
                        loss = recon_loss + vq_loss + args.perceptual_weight * p_loss
                else:
                    # Forward pass without mixed precision
                    reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                    
                    # Calculate losses
                    recon_loss = nn.functional.mse_loss(reconstructions, images)
                    p_loss = perceptual_loss(reconstructions * 2.0 - 1.0, images * 2.0 - 1.0).mean() if args.perceptual_weight > 0 else torch.tensor(0.0, device=device)
                    
                    # Combined loss
                    loss = recon_loss + vq_loss + args.perceptual_weight * p_loss
                
                # Update val losses
                val_losses['total'] += loss.item() * images.size(0)
                val_losses['recon'] += recon_loss.item() * images.size(0)
                val_losses['vq'] += vq_loss.item() * images.size(0)
                val_losses['perceptual'] += p_loss.item() * images.size(0) if args.perceptual_weight > 0 else 0
                
                # Save reconstructions only when needed to save memory
                if i == 0 and (epoch % args.vis_freq == 0 or epoch == args.epochs_per_batch - 1):
                    # Limited sample size
                    n_samples = min(4, images.shape[0])
                    
                    reconstructions_path = os.path.join(
                        batch_results_folder, 
                        'reconstructions', 
                        f'batch_{batch_idx}_reconstruction_epoch_{epoch}.png'
                    )
                    
                    # Combine original and reconstructed images
                    comparison = torch.cat([images[:n_samples], reconstructions[:n_samples]], dim=0)
                    grid = save_image_grid(comparison, reconstructions_path, nrow=n_samples, value_range=(0, 1))
                    writer.add_image(f'val/reconstructions', grid, epoch)
                    
                    # Save codebook usage visualization
                    if hasattr(model, 'get_codebook_usage'):
                        usage = model.get_codebook_usage().cpu().numpy()
                        codebook_path = os.path.join(
                            batch_results_folder, 
                            'codebook', 
                            f'batch_{batch_idx}_codebook_usage_epoch_{epoch}.png'
                        )
                        visualize_codebook_usage(usage, codebook_path, model.num_embeddings)
                        
                        # Log codebook metrics
                        num_active = (usage > 0).sum()
                        writer.add_scalar('codebook/active_entries', num_active, epoch)
                        writer.add_scalar('codebook/usage_entropy', 
                                          -(usage / usage.sum() * np.log(usage / usage.sum() + 1e-10)).sum(), 
                                          epoch)
                
                # Close all matplotlib figures to avoid memory leaks
                plt.close('all')
                
                # Clear memory after each batch
                if device.type == 'cuda' and (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate average val losses
        num_val_samples = len(val_loader.dataset)
        for key in val_losses:
            val_losses[key] /= num_val_samples
            writer.add_scalar(f'val/{key}_loss', val_losses[key], epoch)
        
        # Print epoch summary
        print(f"Epoch {epoch+1} summary:")
        print(f"  Train: total={train_losses['total']:.4f}, recon={train_losses['recon']:.4f}, " + 
              f"vq={train_losses['vq']:.4f}, perceptual={train_losses['perceptual']:.4f}")
        print(f"  Val: total={val_losses['total']:.4f}, recon={val_losses['recon']:.4f}, " + 
              f"vq={val_losses['vq']:.4f}, perceptual={val_losses['perceptual']:.4f}")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_losses['total'])
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint if we have improved
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = os.path.join(args.checkpoint_dir, f'batch_{batch_idx}_best_model.pt')
            torch.save(model.state_dict(), checkpoint_path)
            # Also save for overall best model
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model_stage1.pt"))
            print(f"  New best model saved at {checkpoint_path}")
            
        # Save regular checkpoint at specified frequency
        if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs_per_batch - 1:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'batch_{batch_idx}_checkpoint_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, f'batch_{batch_idx}_final_checkpoint.pt')
    torch.save(model.state_dict(), final_checkpoint_path)
    
    # Ensure we clear memory before returning
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    plt.close('all')
    
    return model, best_val_loss

def train_stage2_with_accumulation(model, discriminator, train_loader, val_loader, device, 
                                  optimizer_g, optimizer_d, scheduler_g, scheduler_d, 
                                  args, batch_idx, writer):
    """
    Train the model in GAN mode using gradient accumulation for memory efficiency
    """
    import lpips
    import matplotlib.pyplot as plt
    from utils.training_utils import save_image_grid, visualize_codebook_usage
    
    # Setup perceptual loss
    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    
    # Setup mixed precision training
    scaler_g = torch.cuda.amp.GradScaler() if args.fp16 and device.type == 'cuda' else None
    scaler_d = torch.cuda.amp.GradScaler() if args.fp16 and device.type == 'cuda' else None
    
    # Create folders for results
    batch_results_folder = os.path.join(args.results_dir, f'batch_{batch_idx}_adv')
    os.makedirs(os.path.join(batch_results_folder, 'reconstructions'), exist_ok=True)
    os.makedirs(os.path.join(batch_results_folder, 'codebook'), exist_ok=True)
    
    # Initialize losses
    best_val_loss = float('inf')
    
    for epoch in range(args.gan_epochs_per_batch):
        model.train()
        discriminator.train()
        
        train_losses = {
            'total_g': 0.0,
            'recon': 0.0,
            'vq': 0.0,
            'perceptual': 0.0,
            'adv_g': 0.0,
            'adv_d': 0.0
        }
        
        # Reset gradients
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        progress_bar = tqdm(enumerate(train_loader), 
                           desc=f"Epoch {epoch+1}/{args.gan_epochs_per_batch}",
                           total=len(train_loader))
        
        for i, (images, _) in progress_bar:
            images = images.to(device, non_blocking=True)
            
            # ======== Train Discriminator ========
            if scaler_d is not None:
                with torch.cuda.amp.autocast():
                    # Get reconstructions
                    with torch.no_grad():
                        reconstructions, _, _, _ = model(images, return_perceptual=True)
                    
                    # Real images
                    real_preds = discriminator(images)
                    real_targets = torch.ones_like(real_preds)
                    loss_d_real = nn.functional.binary_cross_entropy_with_logits(real_preds, real_targets)
                    
                    # Fake images (reconstructions)
                    fake_preds = discriminator(reconstructions.detach())
                    fake_targets = torch.zeros_like(fake_preds)
                    loss_d_fake = nn.functional.binary_cross_entropy_with_logits(fake_preds, fake_targets)
                    
                    # Combined discriminator loss
                    loss_d = 0.5 * (loss_d_real + loss_d_fake)
                    # Normalize by accumulation steps
                    loss_d = loss_d / args.gradient_accumulation_steps
                
                # Accumulate gradients
                scaler_d.scale(loss_d).backward()
                
                # Update after accumulation
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad()
            else:
                # No mixed precision
                with torch.no_grad():
                    reconstructions, _, _, _ = model(images, return_perceptual=True)
                
                # Real images
                real_preds = discriminator(images)
                real_targets = torch.ones_like(real_preds)
                loss_d_real = nn.functional.binary_cross_entropy_with_logits(real_preds, real_targets)
                
                # Fake images (reconstructions)
                fake_preds = discriminator(reconstructions.detach())
                fake_targets = torch.zeros_like(fake_preds)
                loss_d_fake = nn.functional.binary_cross_entropy_with_logits(fake_preds, fake_targets)
                
                # Combined discriminator loss
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                # Normalize by accumulation steps
                loss_d = loss_d / args.gradient_accumulation_steps
                
                # Accumulate gradients
                loss_d.backward()
                
                # Update after accumulation
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            
            train_losses['adv_d'] += loss_d.item() * args.gradient_accumulation_steps
            
            # ======== Train Generator ========
            if scaler_g is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                    
                    # Calculate losses
                    recon_loss = nn.functional.mse_loss(reconstructions, images)
                    p_loss = perceptual_loss(reconstructions * 2.0 - 1.0, images * 2.0 - 1.0).mean() if args.perceptual_weight > 0 else torch.tensor(0.0, device=device)
                    
                    # Adversarial loss (fool the discriminator)
                    fake_preds = discriminator(reconstructions)
                    real_targets = torch.ones_like(fake_preds)
                    adv_loss = nn.functional.binary_cross_entropy_with_logits(fake_preds, real_targets)
                    
                    # Combined generator loss
                    loss_g = recon_loss + vq_loss + args.perceptual_weight * p_loss + args.adv_weight * adv_loss
                    # Normalize by accumulation steps
                    loss_g = loss_g / args.gradient_accumulation_steps
                
                # Accumulate gradients
                scaler_g.scale(loss_g).backward()
                
                # Update after accumulation
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                    optimizer_g.zero_grad()
            else:
                # Forward pass without mixed precision
                reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                
                # Calculate losses
                recon_loss = nn.functional.mse_loss(reconstructions, images)
                p_loss = perceptual_loss(reconstructions * 2.0 - 1.0, images * 2.0 - 1.0).mean() if args.perceptual_weight > 0 else torch.tensor(0.0, device=device)
                
                # Adversarial loss (fool the discriminator)
                fake_preds = discriminator(reconstructions)
                real_targets = torch.ones_like(fake_preds)
                adv_loss = nn.functional.binary_cross_entropy_with_logits(fake_preds, real_targets)
                
                # Combined generator loss
                loss_g = recon_loss + vq_loss + args.perceptual_weight * p_loss + args.adv_weight * adv_loss
                # Normalize by accumulation steps
                loss_g = loss_g / args.gradient_accumulation_steps
                
                # Accumulate gradients
                loss_g.backward()
                
                # Update after accumulation
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer_g.step()
                    optimizer_g.zero_grad()
            
            # Update train losses (reversing the accumulation scaling)
            train_losses['total_g'] += loss_g.item() * args.gradient_accumulation_steps
            train_losses['recon'] += recon_loss.item()
            train_losses['vq'] += vq_loss.item()
            train_losses['perceptual'] += p_loss.item() if args.perceptual_weight > 0 else 0
            train_losses['adv_g'] += adv_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'g_loss': f"{loss_g.item() * args.gradient_accumulation_steps:.4f}",
                'd_loss': f"{loss_d.item() * args.gradient_accumulation_steps:.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'vq': f"{vq_loss.item():.4f}"
            })
            
            # Clear memory occasionally
            if device.type == 'cuda' and (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
        
        # Calculate average train losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
            writer.add_scalar(f'train/{key}_loss', train_losses[key], epoch)
        
        # Validation
        model.eval()
        discriminator.eval()
        
        val_losses = {
            'total_g': 0.0,
            'recon': 0.0,
            'vq': 0.0,
            'perceptual': 0.0,
            'adv_g': 0.0,
            'adv_d': 0.0
        }
        
        with torch.no_grad():
            for i, (images, _) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                
                # Use mixed precision for validation
                if scaler_g is not None:
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                        
                        # Calculate losses
                        recon_loss = nn.functional.mse_loss(reconstructions, images)
                        p_loss = perceptual_loss(reconstructions * 2.0 - 1.0, images * 2.0 - 1.0).mean() if args.perceptual_weight > 0 else torch.tensor(0.0, device=device)
                        
                        # Discriminator evaluation
                        real_preds = discriminator(images)
                        real_targets = torch.ones_like(real_preds)
                        loss_d_real = nn.functional.binary_cross_entropy_with_logits(real_preds, real_targets)
                        
                        fake_preds = discriminator(reconstructions)
                        fake_targets = torch.zeros_like(fake_preds)
                        loss_d_fake = nn.functional.binary_cross_entropy_with_logits(fake_preds, fake_targets)
                        
                        loss_d = 0.5 * (loss_d_real + loss_d_fake)
                        
                        # Generator adversarial loss
                        adv_loss = nn.functional.binary_cross_entropy_with_logits(fake_preds, real_targets)
                        
                        # Combined generator loss
                        loss_g = recon_loss + vq_loss + args.perceptual_weight * p_loss + args.adv_weight * adv_loss
                else:
                    # Forward pass without mixed precision
                    reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                    
                    # Calculate losses
                    recon_loss = nn.functional.mse_loss(reconstructions, images)
                    p_loss = perceptual_loss(reconstructions * 2.0 - 1.0, images * 2.0 - 1.0).mean() if args.perceptual_weight > 0 else torch.tensor(0.0, device=device)
                    
                    # Discriminator evaluation
                    real_preds = discriminator(images)
                    real_targets = torch.ones_like(real_preds)
                    loss_d_real = nn.functional.binary_cross_entropy_with_logits(real_preds, real_targets)
                    
                    fake_preds = discriminator(reconstructions)
                    fake_targets = torch.zeros_like(fake_preds)
                    loss_d_fake = nn.functional.binary_cross_entropy_with_logits(fake_preds, fake_targets)
                    
                    loss_d = 0.5 * (loss_d_real + loss_d_fake)
                    
                    # Generator adversarial loss
                    adv_loss = nn.functional.binary_cross_entropy_with_logits(fake_preds, real_targets)
                    
                    # Combined generator loss
                    loss_g = recon_loss + vq_loss + args.perceptual_weight * p_loss + args.adv_weight * adv_loss
                
                # Update val losses
                val_losses['total_g'] += loss_g.item() * images.size(0)
                val_losses['recon'] += recon_loss.item() * images.size(0)
                val_losses['vq'] += vq_loss.item() * images.size(0)
                val_losses['perceptual'] += p_loss.item() * images.size(0) if args.perceptual_weight > 0 else 0
                val_losses['adv_g'] += adv_loss.item() * images.size(0)
                val_losses['adv_d'] += loss_d.item() * images.size(0)
                
                # Save reconstructions at specified frequency
                if i == 0 and (epoch % args.vis_freq == 0 or epoch == args.gan_epochs_per_batch - 1):
                    # Limited sample size to save memory
                    n_samples = min(4, images.shape[0])
                    
                    reconstructions_path = os.path.join(
                        batch_results_folder, 
                        'reconstructions', 
                        f'batch_{batch_idx}_reconstruction_epoch_{epoch}.png'
                    )
                    
                    # Combine original and reconstructed images
                    comparison = torch.cat([images[:n_samples], reconstructions[:n_samples]], dim=0)
                    grid = save_image_grid(comparison, reconstructions_path, nrow=n_samples, value_range=(0, 1))
                    writer.add_image(f'val/reconstructions', grid, epoch)
                    
                    # Save codebook usage visualization
                    if hasattr(model, 'get_codebook_usage'):
                        usage = model.get_codebook_usage().cpu().numpy()
                        codebook_path = os.path.join(
                            batch_results_folder, 
                            'codebook', 
                            f'batch_{batch_idx}_codebook_usage_epoch_{epoch}.png'
                        )
                        visualize_codebook_usage(usage, codebook_path, model.num_embeddings)
                        
                        # Log codebook metrics
                        num_active = (usage > 0).sum()
                        writer.add_scalar('codebook/active_entries', num_active, epoch)
                        writer.add_scalar('codebook/usage_entropy', 
                                         -(usage / usage.sum() * np.log(usage / usage.sum() + 1e-10)).sum(), 
                                         epoch)
                
                # Close matplotlib figures
                plt.close('all')
                
                # Clear memory after each validation batch
                if device.type == 'cuda' and (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate average val losses
        num_val_samples = len(val_loader.dataset)
        for key in val_losses:
            val_losses[key] /= num_val_samples
            writer.add_scalar(f'val/{key}_loss', val_losses[key], epoch)
        
        # Print epoch summary
        print(f"Epoch {epoch+1} summary:")
        print(f"  Train - G: {train_losses['total_g']:.4f}, D: {train_losses['adv_d']:.4f}, " + 
              f"recon: {train_losses['recon']:.4f}, vq: {train_losses['vq']:.4f}")
        print(f"  Val - G: {val_losses['total_g']:.4f}, D: {val_losses['adv_d']:.4f}, " + 
              f"recon: {val_losses['recon']:.4f}, vq: {val_losses['vq']:.4f}")
        
        # Update learning rates
        if scheduler_g is not None:
            scheduler_g.step(val_losses['total_g'])
            writer.add_scalar('train/lr_g', optimizer_g.param_groups[0]['lr'], epoch)
        
        if scheduler_d is not None:
            scheduler_d.step(val_losses['adv_d'])
            writer.add_scalar('train/lr_d', optimizer_d.param_groups[0]['lr'], epoch)
        
        # Save checkpoint if we have improved
        if val_losses['total_g'] < best_val_loss:
            best_val_loss = val_losses['total_g']
            # Save generator
            checkpoint_path = os.path.join(args.checkpoint_dir, f'batch_{batch_idx}_best_model_gan.pt')
            torch.save(model.state_dict(), checkpoint_path)
            
            # Also save to the main best model path for testing
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model_gan.pt"))
            
            # Save discriminator
            discriminator_path = os.path.join(args.checkpoint_dir, f'batch_{batch_idx}_best_discriminator.pt')
            torch.save(discriminator.state_dict(), discriminator_path)
            torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, "best_discriminator.pt"))
            
            print(f"  New best model saved at {checkpoint_path}")
            
        # Save regular checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.gan_epochs_per_batch - 1:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'batch_{batch_idx}_checkpoint_gan_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            discriminator_path = os.path.join(args.checkpoint_dir, f'batch_{batch_idx}_checkpoint_disc_epoch_{epoch+1}.pt')
            torch.save(discriminator.state_dict(), discriminator_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, f'batch_{batch_idx}_final_checkpoint_gan.pt')
    torch.save(model.state_dict(), final_checkpoint_path)
    final_discriminator_path = os.path.join(args.checkpoint_dir, f'batch_{batch_idx}_final_discriminator.pt')
    torch.save(discriminator.state_dict(), final_discriminator_path)
    
    # Ensure we clean up before returning
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    plt.close('all')
    
    return model, discriminator, best_val_loss

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
    
    # Set benchmark mode for faster training
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    
    # Initialize the optimized VQGAN model
    model = OptimizedVQGAN(
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
    print(f"Generator has {num_params_gen:,} trainable parameters (optimized)")
    print(f"Discriminator has {num_params_disc:,} trainable parameters (optimized)")
    
    # Get all batch files except the test batch
    batch_indices = get_all_batch_files(args.data_path, exclude_test=args.test_batch)
    
    if not batch_indices:
        print("No batch files found! Please check the data path.")
        return
    
    print(f"Found {len(batch_indices)} training batch files: {batch_indices}")
    print(f"Using batch {args.test_batch} as test set")
    
    # Keep track of best overall validation loss
    best_overall_val_loss = float('inf')
    
    # Config for data loading
    data_loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': True,
        'prefetch_factor': 2 if args.num_workers > 0 else None,
    }
    
    # Stage 1: Train with reconstruction and perceptual loss only
    if not args.gan_mode:
        print("\n=== Starting Stage 1 Training (Reconstruction + Perceptual Loss) ===\n")
        
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.log_dir, f'stage1'))
        
        for epoch in range(args.num_epoch):
            print(f"\n--- Global Epoch {epoch+1}/{args.num_epoch} ---\n")
            
            # Train on each batch sequentially
            for batch_idx in batch_indices:
                print(f"\n=== Training on batch {batch_idx} (Stage 1) ===\n")
                
                # Get data loaders for this batch with smaller batch size
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
                
                # Train on this batch (stage 1) with gradient accumulation
                model, batch_best_val_loss = train_stage1_with_accumulation(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    batch_idx=batch_idx,
                    writer=writer
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
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(args.log_dir, f'stage2'))
    
    for epoch in range(max(1, args.num_epoch - 1)):  # At least one epoch for stage 2
        print(f"\n--- Global Epoch {epoch+1}/{max(1, args.num_epoch - 1)} (GAN) ---\n")
        
        # Train on each batch sequentially
        for batch_idx in batch_indices:
            print(f"\n=== Training on batch {batch_idx} (Stage 2 - GAN) ===\n")
            
            # Get data loaders for this batch with smaller batch size
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
            
            # Train on this batch (stage 2 - GAN) with gradient accumulation
            model, discriminator, batch_best_val_loss = train_stage2_with_accumulation(
                model=model,
                discriminator=discriminator,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                args=args,
                batch_idx=batch_idx,
                writer=writer
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
    test_batch_size = max(1, args.batch_size // 2)  # Use smaller batch size for testing
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
    if usage is not None:
        print(f"Codebook usage: {(usage > 0).sum()} / {args.num_embeddings} entries used")

if __name__ == "__main__":
    main()