import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import glob
import random
from matplotlib.animation import FuncAnimation
import imageio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqgan import VQGAN
from utils.data_utils import get_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='VQGAN Demo for UCF101 dataset')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/data/UCFData',
                        help='Directory containing the UCF101 numpy files')
    parser.add_argument('--batch_idx', type=int, default=0,
                        help='Batch index to use for demo')
    
    # Model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128, 256, 512],
                        help='Hidden dimensions of the encoder/decoder')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of the latent space')
    parser.add_argument('--num_embeddings', type=int, default=512,
                        help='Number of embeddings in the codebook')
    parser.add_argument('--downsample_factor', type=int, default=4,
                        help='Downsampling factor in the encoder/decoder')
    
    # Demo parameters
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_gan.pt',
                        help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/demo',
                        help='Directory for demo outputs')
    parser.add_argument('--video_idx', type=int, default=0,
                        help='Video index for demo')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames for animation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # CUDA settings
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    return parser.parse_args()

def load_model(args, device):
    """Load the VQGAN model from checkpoint"""
    model = VQGAN(
        in_channels=3,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.latent_dim,
        downsample_factor=args.downsample_factor
    ).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading model from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Model loaded successfully")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using random weights.")
    
    model.eval()
    return model

def find_video_frames(loader, video_idx, max_frames=30):
    """Find a sequence of frames from the specified video"""
    frame_data = []
    
    print(f"Looking for frames from video {video_idx}...")
    for batch, metadata in loader:
        for i in range(len(batch)):
            if metadata['video_idx'][i].item() == video_idx:
                frame_data.append({
                    'frame_idx': metadata['frame_idx'][i].item(),
                    'patch_h': metadata['patch_h'][i].item(),
                    'patch_w': metadata['patch_w'][i].item(),
                    'tensor': batch[i]
                })
                
                if len(frame_data) >= max_frames * 12:  # 12 patches per frame (3x4 grid)
                    break
        
        if len(frame_data) >= max_frames * 12:
            break
    
    # Sort by frame index and patch position
    frame_data.sort(key=lambda x: (x['frame_idx'], x['patch_h'], x['patch_w']))
    
    # Group by frame index
    frames_by_idx = {}
    for data in frame_data:
        idx = data['frame_idx']
        if idx not in frames_by_idx:
            frames_by_idx[idx] = []
        frames_by_idx[idx].append(data)
    
    return frames_by_idx

def create_animation(model, frames_by_idx, device, output_dir, num_frames=10):
    """Create an animation showing original and reconstructed frames"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    
    # Get frames in order
    frame_indices = sorted(frames_by_idx.keys())[:num_frames]
    
    # For each frame, select the center patch
    center_patches = []
    for idx in frame_indices:
        patches = frames_by_idx[idx]
        
        # Find a center-ish patch (for a 3x4 patch grid)
        center_h, center_w = 1, 1
        best_patch = None
        best_distance = float('inf')
        
        for patch in patches:
            dist = abs(patch['patch_h'] - center_h) + abs(patch['patch_w'] - center_w)
            if dist < best_distance:
                best_distance = dist
                best_patch = patch
        
        if best_patch:
            center_patches.append(best_patch)
    
    # Generate reconstructions
    with torch.no_grad():
        animation_frames = []
        
        for i, patch_data in enumerate(center_patches):
            # Get original patch
            original = patch_data['tensor'].unsqueeze(0).to(device)
            
            # Get reconstruction
            reconstructed, _, _ = model(original)
            
            # Convert tensors to images
            original_img = TF.to_pil_image(patch_data['tensor'])
            recon_img = TF.to_pil_image(reconstructed.squeeze(0).cpu())
            
            # Create side-by-side comparison
            comparison = Image.new('RGB', (original_img.width * 2, original_img.height))
            comparison.paste(original_img, (0, 0))
            comparison.paste(recon_img, (original_img.width, 0))
            
            # Add text overlay
            frame_idx = patch_data['frame_idx']
            comparison_path = os.path.join(output_dir, "frames", f"frame_{i:02d}.png")
            comparison.save(comparison_path)
            
            # Create matplotlib figure for nicer visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            ax1.imshow(original_img)
            ax1.set_title(f"Original (Frame {frame_idx})")
            ax1.axis('off')
            
            ax2.imshow(recon_img)
            ax2.set_title("VQGAN Reconstruction")
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Save figure for animation
            animation_path = os.path.join(output_dir, "frames", f"animation_{i:02d}.png")
            plt.savefig(animation_path, dpi=100)
            plt.close(fig)
            
            animation_frames.append(imageio.imread(animation_path))
        
        # Create GIF animation
        print("Creating animation...")
        imageio.mimsave(
            os.path.join(output_dir, "reconstruction_animation.gif"),
            animation_frames,
            duration=0.5  # 500ms per frame
        )
        
        # Also create MP4 video if possible
        try:
            imageio.mimsave(
                os.path.join(output_dir, "reconstruction_animation.mp4"),
                animation_frames,
                fps=2
            )
        except Exception as e:
            print(f"Could not create MP4 video: {e}")
        
        print(f"Animation saved to {output_dir}")

def create_latent_manipulation(model, sample, device, output_dir, n_steps=8):
    """Create visualizations by manipulating latent vectors"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode the sample
    with torch.no_grad():
        x = sample['tensor'].unsqueeze(0).to(device)
        z, quantized, _, indices = model.encode(x)
        
        # Generate variations by manipulating the latent
        variations = []
        
        # 1. Add noise to the latent
        for scale in np.linspace(0, 0.5, n_steps):
            noise = torch.randn_like(z) * scale
            z_noisy = z + noise
            
            # Quantize and decode
            q_noisy, _, _ = model.vector_quantizer(z_noisy)
            decoded = model.decode(q_noisy)
            variations.append(decoded.squeeze(0).cpu())
        
        # Create a grid
        fig, axs = plt.subplots(2, n_steps // 2, figsize=(n_steps, 4))
        axs = axs.flatten()
        
        # Original image
        axs[0].imshow(TF.to_pil_image(sample['tensor']))
        axs[0].set_title("Original")
        axs[0].axis('off')
        
        # Variations
        for i, img in enumerate(variations):
            if i < len(axs) - 1:  # Skip the first one which is the original
                axs[i+1].imshow(TF.to_pil_image(img))
                axs[i+1].set_title(f"Noise {(i+1)/(n_steps-1):.2f}")
                axs[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latent_variations.png"), dpi=150)
        plt.close()
        
        print(f"Latent manipulations saved to {output_dir}")

def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args, device)
    
    # Get data loader
    loader = get_data_loaders(
        data_path=args.data_path,
        batch_idx=args.batch_idx,
        batch_size=32,  # Larger batch size to get more frames at once
        num_workers=4,
        is_test_set=True
    )
    
    # Find frames from the specified video
    frames_by_idx = find_video_frames(loader, args.video_idx, args.num_frames)
    
    if not frames_by_idx:
        print(f"No frames found for video {args.video_idx}. Try a different video index.")
        return
    
    print(f"Found {len(frames_by_idx)} frames from video {args.video_idx}")
    
    # Create animation
    create_animation(
        model, frames_by_idx, device, 
        os.path.join(args.output_dir, "animation"),
        args.num_frames
    )
    
    # Get a sample for latent manipulation
    sample_frame_idx = list(frames_by_idx.keys())[0]
    sample = frames_by_idx[sample_frame_idx][0]  # Get first patch of first frame
    
    # Create latent manipulation visualizations
    create_latent_manipulation(
        model, sample, device,
        os.path.join(args.output_dir, "latent_manipulation")
    )
    
    print("Demo complete! Check the output directory for results.")

if __name__ == "__main__":
    main()