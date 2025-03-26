import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import lpips
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent tkinter issues
import matplotlib.pyplot as plt

def save_image_grid(images, filepath, nrow=8, normalize=True, value_range=(-1, 1)):
    """Save a grid of images during training"""
    if normalize:
        images = normalize_images(images, value_range)
    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    vutils.save_image(grid, filepath, normalize=False)
    return grid

def normalize_images(images, value_range=(-1, 1)):
    """Normalize images to [0, 1] range"""
    min_val, max_val = value_range
    return (images - min_val) / (max_val - min_val)

def visualize_codebook_usage(usage, filepath, codebook_size):
    """Create visualization of codebook usage"""
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(usage)), usage)
    plt.title("Codebook Usage Distribution")
    plt.xlabel("Codebook Index")
    plt.ylabel("Usage Count")
    plt.savefig(filepath)
    plt.close()
    
    # Create heatmap version for larger codebooks
    if codebook_size > 256:
        side_length = int(np.sqrt(codebook_size))
        if side_length**2 == codebook_size:  # Perfect square
            usage_map = usage.reshape(side_length, side_length)
            plt.figure(figsize=(10, 10))
            plt.imshow(usage_map, cmap='viridis')
            plt.colorbar(label='Usage Count')
            plt.title("Codebook Usage Heatmap")
            plt.savefig(filepath.replace('.png', '_heatmap.png'))
            plt.close()

class PerceptualLoss(nn.Module):
    """Perceptual loss using LPIPS"""
    def __init__(self, net='vgg'):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net=net)
        
    def forward(self, pred, target):
        """Calculate perceptual loss between prediction and target"""
        # LPIPS expects inputs in range [-1, 1]
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        return self.loss_fn(pred_scaled, target_scaled)

def train_vqgan_stage1(model, train_loader, val_loader, device, batch_idx,
                      num_epochs, optimizer, scheduler, log_dir, results_folder, 
                      checkpoint_dir, adv_weight=0.0, perceptual_weight=0.1):
    """
    First stage of training: VQ-VAE with reconstruction loss and perceptual loss
    """
    # Setup logging
    writer = SummaryWriter(os.path.join(log_dir, f'batch_{batch_idx}'))
    
    # Create folders for results
    batch_results_folder = os.path.join(results_folder, f'batch_{batch_idx}')
    os.makedirs(os.path.join(batch_results_folder, 'reconstructions'), exist_ok=True)
    os.makedirs(os.path.join(batch_results_folder, 'codebook'), exist_ok=True)
    
    # Initialize losses
    best_val_loss = float('inf')
    
    # Setup loss functions
    perceptual_loss = PerceptualLoss().to(device)
    
    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = {
            'total': 0.0,
            'recon': 0.0,
            'vq': 0.0,
            'perceptual': 0.0
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            
            # Train the generator (VQ-VAE part)
            optimizer.zero_grad()
            
            # Use mixed precision for forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                    
                    # Calculate losses
                    recon_loss = F.mse_loss(reconstructions, images)
                    p_loss = perceptual_loss(reconstructions, images).mean() if perceptual_weight > 0 else torch.tensor(0.0, device=device)
                    
                    # Combined loss
                    loss = recon_loss + vq_loss + perceptual_weight * p_loss
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass (no mixed precision)
                reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                
                # Calculate losses
                recon_loss = F.mse_loss(reconstructions, images)
                p_loss = perceptual_loss(reconstructions, images).mean() if perceptual_weight > 0 else torch.tensor(0.0, device=device)
                
                # Combined loss
                loss = recon_loss + vq_loss + perceptual_weight * p_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Update train losses
            train_losses['total'] += loss.item()
            train_losses['recon'] += recon_loss.item()
            train_losses['vq'] += vq_loss.item()
            train_losses['perceptual'] += p_loss.item() if perceptual_weight > 0 else 0
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'vq': f"{vq_loss.item():.4f}",
                'p_loss': f"{p_loss.item():.4f}" if perceptual_weight > 0 else "0.0000"
            })
            
            # Clear memory
            torch.cuda.empty_cache() if device.type == 'cuda' else None
        
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
                images = images.to(device)
                
                # Use mixed precision for validation as well (no gradient scaling needed)
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                        
                        # Calculate losses
                        recon_loss = F.mse_loss(reconstructions, images)
                        p_loss = perceptual_loss(reconstructions, images).mean() if perceptual_weight > 0 else torch.tensor(0.0, device=device)
                        
                        # Combined loss
                        loss = recon_loss + vq_loss + perceptual_weight * p_loss
                else:
                    # Forward pass without mixed precision
                    reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                    
                    # Calculate losses
                    recon_loss = F.mse_loss(reconstructions, images)
                    p_loss = perceptual_loss(reconstructions, images).mean() if perceptual_weight > 0 else torch.tensor(0.0, device=device)
                    
                    # Combined loss
                    loss = recon_loss + vq_loss + perceptual_weight * p_loss
                
                # Update val losses
                val_losses['total'] += loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['vq'] += vq_loss.item()
                val_losses['perceptual'] += p_loss.item() if perceptual_weight > 0 else 0
                
                # Save some validation reconstructions (only for the first batch to save memory)
                if i == 0:
                    # Save at most 8 images or all available if < 8
                    n_samples = min(8, images.shape[0])
                    
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
                        
                    # Make sure to close all matplotlib figures to avoid memory leaks
                    plt.close('all')
                
                # Clear memory after each validation batch
                torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Calculate average val losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)
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
            checkpoint_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}_best_model.pt')
            torch.save(model.state_dict(), checkpoint_path)
            # Also save for overall best model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model_stage1.pt"))
            print(f"  New best model saved at {checkpoint_path}")
            
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}_checkpoint_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}_final_checkpoint.pt')
    torch.save(model.state_dict(), final_checkpoint_path)
    
    # Ensure we clear memory before returning
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    plt.close('all')
    
    return model, best_val_loss

def train_vqgan_stage2(model, discriminator, train_loader, val_loader, device, batch_idx,
                       num_epochs, optimizer_g, optimizer_d, scheduler_g, scheduler_d,
                       log_dir, results_folder, checkpoint_dir, 
                       adv_weight=0.1, perceptual_weight=0.1):
    """
    Second stage of training: VQ-GAN with adversarial and perceptual loss
    """
    # Setup logging
    writer = SummaryWriter(os.path.join(log_dir, f'batch_{batch_idx}_adv'))
    
    # Create folders for results
    batch_results_folder = os.path.join(results_folder, f'batch_{batch_idx}_adv')
    os.makedirs(os.path.join(batch_results_folder, 'reconstructions'), exist_ok=True)
    os.makedirs(os.path.join(batch_results_folder, 'codebook'), exist_ok=True)
    
    # Initialize losses
    best_val_loss = float('inf')
    
    # Setup loss functions
    perceptual_loss = PerceptualLoss().to(device)
    
    # Setup mixed precision training
    scaler_g = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    scaler_d = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for epoch in range(num_epochs):
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
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            
            # Train discriminator with mixed precision
            optimizer_d.zero_grad()
            
            if scaler_d is not None:
                with torch.cuda.amp.autocast():
                    # Get reconstructions
                    with torch.no_grad():
                        reconstructions, _, _, _ = model(images, return_perceptual=True)
                    
                    # Real images
                    real_preds = discriminator(images)
                    real_targets = torch.ones_like(real_preds)
                    loss_d_real = F.binary_cross_entropy_with_logits(real_preds, real_targets)
                    
                    # Fake images (reconstructions)
                    fake_preds = discriminator(reconstructions.detach())
                    fake_targets = torch.zeros_like(fake_preds)
                    loss_d_fake = F.binary_cross_entropy_with_logits(fake_preds, fake_targets)
                    
                    # Combined discriminator loss
                    loss_d = 0.5 * (loss_d_real + loss_d_fake)
                
                # Scale the gradients and update
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                # No mixed precision
                with torch.no_grad():
                    reconstructions, _, _, _ = model(images, return_perceptual=True)
                
                # Real images
                real_preds = discriminator(images)
                real_targets = torch.ones_like(real_preds)
                loss_d_real = F.binary_cross_entropy_with_logits(real_preds, real_targets)
                
                # Fake images (reconstructions)
                fake_preds = discriminator(reconstructions.detach())
                fake_targets = torch.zeros_like(fake_preds)
                loss_d_fake = F.binary_cross_entropy_with_logits(fake_preds, fake_targets)
                
                # Combined discriminator loss
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                loss_d.backward()
                optimizer_d.step()
            
            train_losses['adv_d'] += loss_d.item()
            
            # Train generator (VQ-GAN) with mixed precision
            optimizer_g.zero_grad()
            
            if scaler_g is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                    
                    # Calculate losses
                    recon_loss = F.mse_loss(reconstructions, images)
                    p_loss = perceptual_loss(reconstructions, images).mean() if perceptual_weight > 0 else torch.tensor(0.0, device=device)
                    
                    # Adversarial loss (fool the discriminator)
                    fake_preds = discriminator(reconstructions)
                    real_targets = torch.ones_like(fake_preds)
                    adv_loss = F.binary_cross_entropy_with_logits(fake_preds, real_targets)
                    
                    # Combined generator loss
                    loss_g = recon_loss + vq_loss + perceptual_weight * p_loss + adv_weight * adv_loss
                
                # Scale the gradients and update
                scaler_g.scale(loss_g).backward()
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                # Forward pass without mixed precision
                reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                
                # Calculate losses
                recon_loss = F.mse_loss(reconstructions, images)
                p_loss = perceptual_loss(reconstructions, images).mean() if perceptual_weight > 0 else torch.tensor(0.0, device=device)
                
                # Adversarial loss (fool the discriminator)
                fake_preds = discriminator(reconstructions)
                real_targets = torch.ones_like(fake_preds)
                adv_loss = F.binary_cross_entropy_with_logits(fake_preds, real_targets)
                
                # Combined generator loss
                loss_g = recon_loss + vq_loss + perceptual_weight * p_loss + adv_weight * adv_loss
                
                # Backward pass and update
                loss_g.backward()
                optimizer_g.step()
            
            # Update train losses
            train_losses['total_g'] += loss_g.item()
            train_losses['recon'] += recon_loss.item()
            train_losses['vq'] += vq_loss.item()
            train_losses['perceptual'] += p_loss.item() if perceptual_weight > 0 else 0
            train_losses['adv_g'] += adv_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'g_loss': f"{loss_g.item():.4f}",
                'd_loss': f"{loss_d.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'vq': f"{vq_loss.item():.4f}"
            })
            
            # Clear memory after each batch
            torch.cuda.empty_cache() if device.type == 'cuda' else None
        
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
                images = images.to(device)
                
                # Use mixed precision for validation as well
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                        
                        # Calculate losses
                        recon_loss = F.mse_loss(reconstructions, images)
                        p_loss = perceptual_loss(reconstructions, images).mean() if perceptual_weight > 0 else torch.tensor(0.0, device=device)
                        
                        # Discriminator evaluation
                        real_preds = discriminator(images)
                        real_targets = torch.ones_like(real_preds)
                        loss_d_real = F.binary_cross_entropy_with_logits(real_preds, real_targets)
                        
                        fake_preds = discriminator(reconstructions)
                        fake_targets = torch.zeros_like(fake_preds)
                        loss_d_fake = F.binary_cross_entropy_with_logits(fake_preds, fake_targets)
                        
                        loss_d = 0.5 * (loss_d_real + loss_d_fake)
                        
                        # Generator adversarial loss
                        adv_loss = F.binary_cross_entropy_with_logits(fake_preds, real_targets)
                        
                        # Combined generator loss
                        loss_g = recon_loss + vq_loss + perceptual_weight * p_loss + adv_weight * adv_loss
                else:
                    # Forward pass without mixed precision
                    reconstructions, quantized, vq_loss, perceptual = model(images, return_perceptual=True)
                    
                    # Calculate losses
                    recon_loss = F.mse_loss(reconstructions, images)
                    p_loss = perceptual_loss(reconstructions, images).mean() if perceptual_weight > 0 else torch.tensor(0.0, device=device)
                    
                    # Discriminator evaluation
                    real_preds = discriminator(images)
                    real_targets = torch.ones_like(real_preds)
                    loss_d_real = F.binary_cross_entropy_with_logits(real_preds, real_targets)
                    
                    fake_preds = discriminator(reconstructions)
                    fake_targets = torch.zeros_like(fake_preds)
                    loss_d_fake = F.binary_cross_entropy_with_logits(fake_preds, fake_targets)
                    
                    loss_d = 0.5 * (loss_d_real + loss_d_fake)
                    
                    # Generator adversarial loss
                    adv_loss = F.binary_cross_entropy_with_logits(fake_preds, real_targets)
                    
                    # Combined generator loss
                    loss_g = recon_loss + vq_loss + perceptual_weight * p_loss + adv_weight * adv_loss
                
                # Update val losses
                val_losses['total_g'] += loss_g.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['vq'] += vq_loss.item()
                val_losses['perceptual'] += p_loss.item() if perceptual_weight > 0 else 0
                val_losses['adv_g'] += adv_loss.item()
                val_losses['adv_d'] += loss_d.item()
                
                # Save some validation reconstructions (only for first batch)
                if i == 0:
                    # Save a limited number of images to save memory
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
                
                # Close any matplotlib figures and clear memory
                plt.close('all')
                torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Calculate average val losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)
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
            checkpoint_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}_best_model_gan.pt')
            torch.save(model.state_dict(), checkpoint_path)
            
            # Also save to the main best model path for testing
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model_gan.pt"))
            
            # Save discriminator
            discriminator_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}_best_discriminator.pt')
            torch.save(discriminator.state_dict(), discriminator_path)
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, "best_discriminator.pt"))
            
            print(f"  New best model saved at {checkpoint_path}")
            
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}_checkpoint_gan_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            discriminator_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}_checkpoint_disc_epoch_{epoch+1}.pt')
            torch.save(discriminator.state_dict(), discriminator_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}_final_checkpoint_gan.pt')
    torch.save(model.state_dict(), final_checkpoint_path)
    final_discriminator_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}_final_discriminator.pt')
    torch.save(discriminator.state_dict(), final_discriminator_path)
    
    # Ensure we clean up before returning
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    plt.close('all')
    
    return model, discriminator, best_val_loss

def evaluate_on_test_set(model, discriminator, test_loader, device, results_folder, 
                         perceptual_weight=0.1, adv_weight=0.1):
    """Evaluate the VQGAN on the test set"""
    model.eval()
    discriminator.eval()
    
    # Create results folder
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(os.path.join(results_folder, 'test_reconstructions'), exist_ok=True)
    os.makedirs(os.path.join(results_folder, 'codebook'), exist_ok=True)
    
    # Setup loss functions
    perceptual_loss = PerceptualLoss().to(device)
    
    # Initialize test metrics
    test_losses = {
        'total_g': 0.0,
        'recon': 0.0,
        'vq': 0.0,
        'perceptual': 0.0,
        'adv_g': 0.0
    }
    all_codebook_indices = []
    
    # Use mixed precision for evaluation if available
    with torch.no_grad():
        for i, (images, metadata) in enumerate(tqdm(test_loader, desc="Evaluating on test set")):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Use mixed precision for memory efficiency
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    # Forward pass
                    reconstructions, quantized, vq_loss, perceptual, indices = model(
                        images, return_perceptual=True, return_indices=True
                    )
                    
                    # Calculate losses
                    recon_loss = F.mse_loss(reconstructions, images)
                    p_loss = perceptual_loss(reconstructions, images).mean() if perceptual_weight > 0 else torch.tensor(0.0, device=device)
                    
                    # Adversarial evaluation
                    fake_preds = discriminator(reconstructions)
                    real_targets = torch.ones_like(fake_preds)
                    adv_loss = F.binary_cross_entropy_with_logits(fake_preds, real_targets)
                    
                    # Combined loss
                    loss_g = recon_loss + vq_loss + perceptual_weight * p_loss + adv_weight * adv_loss
            else:
                # Forward pass without mixed precision
                reconstructions, quantized, vq_loss, perceptual, indices = model(
                    images, return_perceptual=True, return_indices=True
                )
                
                # Calculate losses
                recon_loss = F.mse_loss(reconstructions, images)
                p_loss = perceptual_loss(reconstructions, images).mean() if perceptual_weight > 0 else torch.tensor(0.0, device=device)
                
                # Adversarial evaluation
                fake_preds = discriminator(reconstructions)
                real_targets = torch.ones_like(fake_preds)
                adv_loss = F.binary_cross_entropy_with_logits(fake_preds, real_targets)
                
                # Combined loss
                loss_g = recon_loss + vq_loss + perceptual_weight * p_loss + adv_weight * adv_loss
            
            # Update test losses
            test_losses['total_g'] += loss_g.item() * batch_size
            test_losses['recon'] += recon_loss.item() * batch_size
            test_losses['vq'] += vq_loss.item() * batch_size
            test_losses['perceptual'] += p_loss.item() * batch_size if perceptual_weight > 0 else 0
            test_losses['adv_g'] += adv_loss.item() * batch_size
            
            # Collect codebook indices for analysis
            # Only store if small enough to avoid memory issues
            if i < 5:
                all_codebook_indices.append(indices.cpu())
            
            # Save reconstructions for visualization (limited to 5 batches to save memory)
            if i < 5:
                # Only use a small subset of images
                n_samples = min(4, images.shape[0])
                
                reconstructions_path = os.path.join(
                    results_folder, 
                    'test_reconstructions', 
                    f'batch_test_reconstruction_epoch_{i}.png'
                )
                
                # Combine original and reconstructed images
                comparison = torch.cat([images[:n_samples], reconstructions[:n_samples]], dim=0)
                save_image_grid(comparison, reconstructions_path, nrow=n_samples, value_range=(0, 1))
            
            # Clean up to prevent memory issues
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            plt.close('all')
    
    # Calculate average test losses
    num_samples = len(test_loader.dataset)
    for key in test_losses:
        test_losses[key] /= num_samples
    
    # Print test results
    print(f"Test results:")
    print(f"  Total G loss: {test_losses['total_g']:.4f}")
    print(f"  Recon loss: {test_losses['recon']:.4f}")
    print(f"  VQ loss: {test_losses['vq']:.4f}")
    print(f"  Perceptual loss: {test_losses['perceptual']:.4f}")
    print(f"  Adversarial G loss: {test_losses['adv_g']:.4f}")
    
    # Get and visualize codebook usage
    if hasattr(model, 'get_codebook_usage'):
        usage = model.get_codebook_usage().cpu().numpy()
        codebook_path = os.path.join(results_folder, 'codebook', 'batch_test_codebook_usage_epoch_0.png')
        visualize_codebook_usage(usage, codebook_path, model.num_embeddings)
        
        # Calculate statistics
        num_active = (usage > 0).sum()
        usage_entropy = -(usage / usage.sum() * np.log(usage / usage.sum() + 1e-10)).sum()
        
        print(f"  Codebook statistics:")
        print(f"    Active entries: {num_active}/{model.num_embeddings} ({num_active/model.num_embeddings*100:.1f}%)")
        print(f"    Usage entropy: {usage_entropy:.4f}")
    
    # Clean up before returning
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    plt.close('all')
    
    return test_losses['total_g'], test_losses['recon'], test_losses['vq'], usage