import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision.transforms import functional as TF
from PIL import Image
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqgan import VQGAN, Discriminator
from utils.data_utils import get_data_loaders
from utils.training_utils import save_image_grid, normalize_images

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize VQGAN on UCF101 frames')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/data/UCFData',
                        help='Directory containing the UCF101 numpy files')
    parser.add_argument('--batch_idx', type=int, default=0,
                        help='Batch index to use for visualization')
    
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
                        help='Number of embeddings in the codebook')
    parser.add_argument('--downsample_factor', type=int, default=4,
                        help='Downsampling factor in the encoder/decoder')
    
    # Visualization parameters
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='results/visualization',
                        help='Directory for saving visualizations')
    parser.add_argument('--video_idx', type=int, default=0,
                        help='Video index to visualize (for sequential frames)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Starting frame index for video visualization')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames to visualize in sequence')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_gan.pt',
                        help='Path to the checkpoint to load')
    
    # CUDA settings
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    return parser.parse_args()

def visualize_latent_space(model, samples, device, output_file):
    """Visualize latent space and codebook usage"""
    # Get latent representations
    model.eval()
    with torch.no_grad():
        z_list = []
        indices_list = []
        for x in samples:
            x = x.unsqueeze(0).to(device)
            z, quantized, _, indices = model.encode(x)
            z_list.append(z.cpu())
            indices_list.append(indices.cpu())
        
        # Combine all latent representations
        z_all = torch.cat(z_list, dim=0)
        indices_all = torch.cat(indices_list, dim=0)
    
    # Create visualization
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: First two dimensions of latent space
    axs[0, 0].scatter(z_all[:, 0, :, :].flatten().numpy(), z_all[:, 1, :, :].flatten().numpy(),
                      alpha=0.1, s=1)
    axs[0, 0].set_title('Latent Space Dimensions 0 vs 1')
    axs[0, 0].set_xlabel('Dimension 0')
    axs[0, 0].set_ylabel('Dimension 1')
    
    # Plot 2: Histogram of latent values
    axs[0, 1].hist(z_all.flatten().numpy(), bins=50)
    axs[0, 1].set_title('Latent Values Distribution')
    axs[0, 1].set_xlabel('Latent Value')
    axs[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Codebook usage
    if hasattr(model, 'get_codebook_usage'):
        usage = model.get_codebook_usage().cpu().numpy()
        
        # Sort usage to see most used vectors
        sorted_indices = np.argsort(usage)[::-1]
        sorted_usage = usage[sorted_indices]
        
        # Plot top N vectors
        top_n = min(50, len(sorted_usage))
        axs[1, 0].bar(range(top_n), sorted_usage[:top_n])
        axs[1, 0].set_title(f'Top {top_n} Codebook Vector Usage')
        axs[1, 0].set_xlabel('Sorted Codebook Index')
        axs[1, 0].set_ylabel('Usage Count')
        
        # Calculate active codebook percentage
        active_count = (usage > 0).sum()
        active_percentage = active_count / len(usage) * 100
        axs[1, 0].text(0.5, 0.9, f'Active: {active_count}/{len(usage)} ({active_percentage:.1f}%)',
                      transform=axs[1, 0].transAxes, ha='center')
    
    # Plot 4: 2D histogram of indices
    flattened_indices = indices_all.flatten().numpy()
    axs[1, 1].hist2d(
        np.repeat(range(len(flattened_indices)), 1), 
        flattened_indices,
        bins=[min(100, len(flattened_indices)), min(100, model.num_embeddings)],
        cmap='viridis'
    )
    axs[1, 1].set_title('Codebook Index Distribution')
    axs[1, 1].set_xlabel('Sample Position')
    axs[1, 1].set_ylabel('Codebook Index')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def visualize_reconstructions(model, loader, device, output_dir, num_samples=8):
    """Generate reconstruction visualizations"""
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get random samples from the loader
    all_samples = []
    all_metadata = []
    
    # Get a few batches and select random samples
    for i, (x, metadata) in enumerate(loader):
        if len(all_samples) < num_samples:
            # Randomly select samples from this batch
            indices = torch.randperm(len(x))[:min(num_samples - len(all_samples), len(x))]
            all_samples.extend([x[idx] for idx in indices])
            all_metadata.extend([{k: v[idx].item() if torch.is_tensor(v) else v[idx] for k, v in metadata.items()} for idx in indices])
        else:
            break
    
    # Ensure we have exactly num_samples
    all_samples = all_samples[:num_samples]
    all_metadata = all_metadata[:num_samples]
    
    # Process samples
    with torch.no_grad():
        # Create the figure for original vs reconstructed
        fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        
        for i, (sample, metadata) in enumerate(zip(all_samples, all_metadata)):
            # Move to device
            x = sample.unsqueeze(0).to(device)
            
            # Forward pass
            reconstructed, quantized, vq_loss = model(x)
            
            # Convert to images
            original_img = TF.to_pil_image(sample)
            recon_img = TF.to_pil_image(reconstructed.squeeze(0).cpu())
            
            # Add to the figure
            axs[0, i].imshow(np.array(original_img))
            axs[0, i].set_title(f"Original\nVideo {metadata['video_idx']}, Frame {metadata['frame_idx']}")
            axs[0, i].axis('off')
            
            axs[1, i].imshow(np.array(recon_img))
            axs[1, i].set_title(f"Reconstructed")
            axs[1, i].axis('off')
            
            # Save individual reconstructions
            fig_single, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
            ax1.imshow(np.array(original_img))
            ax1.set_title(f"Original")
            ax1.axis('off')
            
            ax2.imshow(np.array(recon_img))
            ax2.set_title(f"Reconstructed")
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i}_recon.png'), dpi=150)
            plt.close(fig_single)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reconstructions_grid.png'), dpi=200)
        plt.close(fig)
        
        # Create a combined grid using torchvision
        all_recons = []
        for sample in all_samples:
            x = sample.unsqueeze(0).to(device)
            recon, _, _ = model(x)
            all_recons.append(recon.cpu())
        
        # Create comparison grid
        originals = torch.stack(all_samples)
        recons = torch.cat(all_recons)
        
        # Make comparison grid
        comparison = torch.cat([originals, recons], dim=0)
        grid = save_image_grid(comparison, 
                              os.path.join(output_dir, 'comparison_grid.png'), 
                              nrow=num_samples, value_range=(0, 1))
        
        # Return the samples and their metadata for further visualizations
        return all_samples, all_metadata

def visualize_video_sequence(model, loader, device, output_dir, video_idx, start_frame, num_frames):
    """Visualize a sequence of frames from a video to show temporal coherence"""
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
    
    # Get frames from the specified video
    all_frames = []
    frame_data = []
    
    print(f"Finding frames from video {video_idx}, starting at frame {start_frame}")
    
    # Iterate through the dataset to find matching frames
    for i, (x, metadata) in enumerate(tqdm(loader, desc="Looking for frames")):
        for j in range(len(x)):
            current_video = metadata['video_idx'][j].item()
            current_frame = metadata['frame_idx'][j].item()
            
            if current_video == video_idx and current_frame >= start_frame and current_frame < start_frame + num_frames:
                # Found a frame from our target sequence
                frame_data.append({
                    'video_idx': current_video,
                    'frame_idx': current_frame,
                    'patch_h': metadata['patch_h'][j].item() if 'patch_h' in metadata else 0,
                    'patch_w': metadata['patch_w'][j].item() if 'patch_w' in metadata else 0,
                    'tensor': x[j]
                })
    
    # Sort frames by frame index and patch position
    frame_data.sort(key=lambda x: (x['frame_idx'], x['patch_h'], x['patch_w']))
    
    # Group frames by frame index
    frame_groups = {}
    for data in frame_data:
        frame_idx = data['frame_idx']
        if frame_idx not in frame_groups:
            frame_groups[frame_idx] = []
        frame_groups[frame_idx].append(data)
    
    # For each frame index, get one patch (e.g., the center patch)
    selected_frames = []
    for idx in sorted(frame_groups.keys()):
        # Find a center-ish patch if possible
        patches = frame_groups[idx]
        
        # Pick one patch from each frame (preferably a center patch if available)
        center_h, center_w = 1, 1  # For 3x4 grid (assuming 0-indexed)
        best_patch = None
        best_distance = float('inf')
        
        for patch in patches:
            dist = abs(patch['patch_h'] - center_h) + abs(patch['patch_w'] - center_w)
            if dist < best_distance:
                best_distance = dist
                best_patch = patch
        
        if best_patch:
            selected_frames.append(best_patch)
    
    # Make sure frames are in order
    selected_frames.sort(key=lambda x: x['frame_idx'])
    
    # Get reconstructions for each frame
    with torch.no_grad():
        for i, frame_data in enumerate(selected_frames):
            # Get the original frame
            original = frame_data['tensor'].unsqueeze(0).to(device)
            
            # Get reconstruction
            reconstructed, _, _ = model(original)
            
            # Save the original and reconstruction
            original_img = TF.to_pil_image(frame_data['tensor'])
            recon_img = TF.to_pil_image(reconstructed.squeeze(0).cpu())
            
            # Save individual frames
            original_img.save(os.path.join(output_dir, 'frames', f'frame_{i:02d}_original.png'))
            recon_img.save(os.path.join(output_dir, 'frames', f'frame_{i:02d}_recon.png'))
            
            # Create comparison image
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
            ax1.imshow(np.array(original_img))
            ax1.set_title(f"Original Frame {frame_data['frame_idx']}")
            ax1.axis('off')
            
            ax2.imshow(np.array(recon_img))
            ax2.set_title(f"Reconstructed")
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'frame_{i:02d}_comparison.png'), dpi=150)
            plt.close()
        
        # Create grid of original frames
        original_frames = [frame_data['tensor'] for frame_data in selected_frames]
        original_grid = vutils.make_grid(original_frames, nrow=4, padding=2)
        vutils.save_image(original_grid, os.path.join(output_dir, 'original_sequence.png'))
        
        # Create grid of reconstructed frames
        recon_frames = []
        for frame_data in selected_frames:
            original = frame_data['tensor'].unsqueeze(0).to(device)
            reconstructed, _, _ = model(original)
            recon_frames.append(reconstructed.squeeze(0).cpu())
        
        recon_grid = vutils.make_grid(recon_frames, nrow=4, padding=2)
        vutils.save_image(recon_grid, os.path.join(output_dir, 'reconstructed_sequence.png'))
        
        # Create a side-by-side comparison grid
        num_frames = len(original_frames)
        rows = math.ceil(num_frames / 4)
        fig, axs = plt.subplots(rows, 8, figsize=(16, 2 * rows))
        
        for i in range(num_frames):
            row = i // 4
            col = (i % 4) * 2
            
            if rows > 1:
                axs[row, col].imshow(TF.to_pil_image(original_frames[i]))
                axs[row, col].set_title(f"Original {selected_frames[i]['frame_idx']}")
                axs[row, col].axis('off')
                
                axs[row, col+1].imshow(TF.to_pil_image(recon_frames[i]))
                axs[row, col+1].set_title(f"Reconstructed")
                axs[row, col+1].axis('off')
            else:
                axs[col].imshow(TF.to_pil_image(original_frames[i]))
                axs[col].set_title(f"Original {selected_frames[i]['frame_idx']}")
                axs[col].axis('off')
                
                axs[col+1].imshow(TF.to_pil_image(recon_frames[i]))
                axs[col+1].set_title(f"Reconstructed")
                axs[col+1].axis('off')
        
        # Hide unused subplots
        for i in range(num_frames, rows * 4):
            row = i // 4
            col = (i % 4) * 2
            if rows > 1:
                axs[row, col].axis('off')
                axs[row, col+1].axis('off')
            else:
                axs[col].axis('off')
                axs[col+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sequence_comparison.png'), dpi=200)
        plt.close()
        
        print(f"Sequence visualization saved to {output_dir}")

def visualize_interpolation(model, samples, device, output_dir):
    """Visualize latent space interpolation between samples"""
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get encodings for at least 2 samples
    if len(samples) < 2:
        print("Need at least 2 samples for interpolation")
        return
    
    # Choose 2 random samples for interpolation
    idx1, idx2 = 0, 1  # Could be randomized
    sample1, sample2 = samples[idx1], samples[idx2]
    
    # Encode the samples
    with torch.no_grad():
        x1 = sample1.unsqueeze(0).to(device)
        x2 = sample2.unsqueeze(0).to(device)
        
        # Get latent representations
        z1, q1, _, _ = model.encode(x1)
        z2, q2, _, _ = model.encode(x2)
        
        # Create interpolation steps
        steps = 8
        interpolated_images = []
        
        for alpha in np.linspace(0, 1, steps):
            # Interpolate in the pre-quantized space
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Quantize the interpolated latent
            q_interp, _, _ = model.vector_quantizer(z_interp)
            
            # Decode
            decoded = model.decode(q_interp)
            interpolated_images.append(decoded.squeeze(0).cpu())
        
        # Create a visualization grid
        grid = vutils.make_grid(interpolated_images, nrow=steps, padding=2)
        vutils.save_image(grid, os.path.join(output_dir, 'latent_interpolation.png'))
        
        # Create a figure with original images and interpolation
        fig, axs = plt.subplots(1, steps+2, figsize=(2*(steps+2), 2))
        
        # Display original images
        axs[0].imshow(TF.to_pil_image(sample1))
        axs[0].set_title("Original 1")
        axs[0].axis('off')
        
        # Display interpolations
        for i, img in enumerate(interpolated_images):
            axs[i+1].imshow(TF.to_pil_image(img))
            axs[i+1].set_title(f"α={i/(steps-1):.2f}")
            axs[i+1].axis('off')
        
        # Display second original
        axs[-1].imshow(TF.to_pil_image(sample2))
        axs[-1].set_title("Original 2")
        axs[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'interpolation_with_originals.png'), dpi=200)
        plt.close()
        
        print(f"Interpolation visualization saved to {output_dir}")

def visualize_codebook(model, device, output_dir):
    """Visualize codebook entries directly"""
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the codebook entries
    if hasattr(model, 'vector_quantizer') and hasattr(model.vector_quantizer, 'embeddings'):
        # Get the embeddings
        embeddings = model.vector_quantizer.embeddings
        
        # Get usage statistics
        usage = model.vector_quantizer.get_codebook_usage().cpu().numpy()
        
        # Sort by usage
        sorted_indices = np.argsort(usage)[::-1]
        
        # Visualize the most used embeddings
        top_n = min(64, model.num_embeddings)
        
        # Reshape embeddings to a form suitable for decoding
        latent_dim = model.latent_dim
        
        # Create a grid of visualizations for the top used codes
        rows = 8
        cols = 8
        fig, axs = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
        
        with torch.no_grad():
            for i in range(min(top_n, rows*cols)):
                row = i // cols
                col = i % cols
                
                # Get the embedding
                embed_idx = sorted_indices[i]
                embedding = embeddings[embed_idx].clone().unsqueeze(0)
                
                # Create a latent representation filled with this embedding
                latent_shape = (1, latent_dim, 10, 10)  # Small latent shape for decoding
                z = embedding.view(1, latent_dim, 1, 1).expand(latent_shape)
                
                # Decode
                decoded = model.decode(z)
                
                # Display
                axs[row, col].imshow(TF.to_pil_image(decoded.squeeze(0).cpu()))
                axs[row, col].set_title(f"Code {embed_idx}\nUsage: {usage[embed_idx]:.0f}", fontsize=8)
                axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'codebook_visualization.png'), dpi=200)
        plt.close()
        
        # Create a usage histogram
        plt.figure(figsize=(10, 6))
        active_codes = (usage > 0).sum()
        plt.bar(range(len(usage)), usage[sorted_indices])
        plt.title(f"Codebook Usage (Active: {active_codes}/{len(usage)} = {active_codes/len(usage)*100:.1f}%)")
        plt.xlabel("Codebook Entry (sorted by usage)")
        plt.ylabel("Usage Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'codebook_usage.png'), dpi=200)
        plt.close()
        
        print(f"Codebook visualization saved to {output_dir}")

def main():
    args = parse_args()
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the VQGAN model
    model = VQGAN(
        in_channels=3,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.latent_dim,  # Must be same as latent_dim
        downsample_factor=args.downsample_factor,
    ).to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading model from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Model loaded successfully")
    else:
        print(f"Checkpoint {args.checkpoint} not found, using random weights")
    
    # Get data loader
    test_loader = get_data_loaders(
        data_path=args.data_path,
        batch_idx=args.batch_idx,
        batch_size=args.num_samples,
        patch_size=args.patch_size,
        num_patches_h=args.num_patches_h,
        num_patches_w=args.num_patches_w,
        num_workers=4,
        is_test_set=True
    )
    
    # Visualize reconstructions
    print("Generating reconstruction visualizations...")
    samples, metadata = visualize_reconstructions(
        model, test_loader, device, 
        os.path.join(args.output_dir, 'reconstructions'),
        args.num_samples
    )
    
    # Visualize latent space
    print("Generating latent space visualizations...")
    visualize_latent_space(
        model, samples, device,
        os.path.join(args.output_dir, 'latent_space.png')
    )
    
    # Visualize interpolation
    print("Generating latent interpolation visualizations...")
    visualize_interpolation(
        model, samples, device,
        os.path.join(args.output_dir, 'interpolation')
    )
    
    # Visualize codebook
    print("Generating codebook visualizations...")
    visualize_codebook(
        model, device,
        os.path.join(args.output_dir, 'codebook')
    )
    
    # Visualize video sequence
    print("Generating video sequence visualizations...")
    visualize_video_sequence(
        model, test_loader, device,
        os.path.join(args.output_dir, 'sequence'),
        args.video_idx, args.start_frame, args.num_frames
    )
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()