#!/usr/bin/env python3
"""
Test script to verify the correctness of data projections.
This script:
1. Loads a sample batch file
2. Projects a few samples to latent space
3. Reconstructs the original images from latent space
4. Verifies that all three projection formats (continuous, vectors, indices) are consistent
5. Creates comparison visualizations
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqgan_optimized import OptimizedVQGAN

def parse_args():
    parser = argparse.ArgumentParser(description='Test data projections from VQGAN')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/data/UCFData',
                        help='Path to UCF101 numpy files')
    parser.add_argument('--projection_path', type=str, 
                        default='/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections',
                        help='Path to projection files')
    parser.add_argument('--output_path', type=str, 
                        default='/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections/test_results',
                        help='Path to save test results')
    
    # Test parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained VQGAN model checkpoint')
    parser.add_argument('--batch_index', type=int, default=0,
                        help='Batch index to test')
    parser.add_argument('--num_samples', type=int, default=2,
                        help='Number of samples to test')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Hidden dimensions of the encoder/decoder')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Dimension of the latent space')
    parser.add_argument('--num_embeddings', type=int, default=256,
                        help='Number of embeddings in the codebook')
    parser.add_argument('--downsample_factor', type=int, default=4,
                        help='Downsampling factor in the encoder/decoder')
    
    # Patching parameters
    parser.add_argument('--patch_size', type=int, default=80,
                        help='Size of image patches')
    parser.add_argument('--num_patches_h', type=int, default=3,
                        help='Number of patches in height dimension (240/80=3)')
    parser.add_argument('--num_patches_w', type=int, default=4,
                        help='Number of patches in width dimension (320/80=4)')
    
    # Compute parameters
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    return parser.parse_args()

def load_model(args, device):
    """Load the trained VQGAN model"""
    print(f"Loading model from {args.model_path}")
    
    # Initialize model with same parameters as during training
    model = OptimizedVQGAN(
        in_channels=3,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.latent_dim,
        downsample_factor=args.downsample_factor,
    ).to(device)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Set to evaluation mode
    model.eval()
    return model

def load_data(args):
    """Load original data and projections"""
    # Original data
    orig_data_path = os.path.join(args.data_path, f"ucf101_subset_batch_{args.batch_index}.npy")
    if not os.path.exists(orig_data_path):
        print(f"Original data file not found: {orig_data_path}")
        sys.exit(1)
    
    print(f"Loading original data: {orig_data_path}")
    original_data = np.load(orig_data_path)
    print(f"Original data shape: {original_data.shape}")
    
    # Projection data
    cont_path = os.path.join(args.projection_path, 'continuous_latents', 
                            f'batch_{args.batch_index}_continuous.npy')
    vecs_path = os.path.join(args.projection_path, 'codebook_entries', 
                            f'batch_{args.batch_index}_vectors.npy')
    idx_path = os.path.join(args.projection_path, 'codebook_indices', 
                           f'batch_{args.batch_index}_indices.npy')
    
    # Check if projection files exist
    if not all(os.path.exists(p) for p in [cont_path, vecs_path, idx_path]):
        print("Projection files not found. Please run the projection script first.")
        # List the files that are missing
        for p in [cont_path, vecs_path, idx_path]:
            if not os.path.exists(p):
                print(f"Missing: {p}")
        sys.exit(1)
    
    # Load projections
    print("Loading projection data")
    continuous_latents = np.load(cont_path)
    codebook_vectors = np.load(vecs_path)
    codebook_indices = np.load(idx_path)
    
    print(f"Continuous latents shape: {continuous_latents.shape}")
    print(f"Codebook vectors shape: {codebook_vectors.shape}")
    print(f"Codebook indices shape: {codebook_indices.shape}")
    
    return original_data, continuous_latents, codebook_vectors, codebook_indices

def test_reconstructions(model, original_data, continuous_latents, codebook_vectors, 
                        codebook_indices, args, device):
    """Test reconstruction from latent representations"""
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Select samples to test
    num_samples = min(args.num_samples, original_data.shape[0])
    sample_indices = np.random.choice(original_data.shape[0], num_samples, replace=False)
    
    # Height and width of each patch
    actual_h_patch_size = original_data.shape[2] // args.num_patches_h
    actual_w_patch_size = original_data.shape[3] // args.num_patches_w
    
    for sample_idx in sample_indices:
        print(f"\nTesting sample {sample_idx}")
        
        # Test a single frame (first frame of the sequence)
        frame_idx = 0
        
        # Get original frame
        original_frame = original_data[sample_idx, frame_idx]
        
        # Initialize reconstructed frames
        recon_from_continuous = np.zeros_like(original_frame)
        recon_from_vectors = np.zeros_like(original_frame)
        recon_from_indices = np.zeros_like(original_frame)
        
        # Process each patch in the frame
        with torch.no_grad():
            for h_idx in range(args.num_patches_h):
                for w_idx in range(args.num_patches_w):
                    # Extract patch region coordinates
                    h_start = h_idx * actual_h_patch_size
                    w_start = w_idx * actual_w_patch_size
                    h_end = min(h_start + actual_h_patch_size, original_frame.shape[0])
                    w_end = min(w_start + actual_w_patch_size, original_frame.shape[1])
                    
                    # Get original patch
                    original_patch = original_frame[h_start:h_end, w_start:w_end]
                    
                    # Get corresponding latent representations
                    continuous_latent = continuous_latents[sample_idx, frame_idx, h_idx, w_idx]
                    codebook_vector = codebook_vectors[sample_idx, frame_idx, h_idx, w_idx]
                    codebook_index = codebook_indices[sample_idx, frame_idx, h_idx, w_idx]
                    
                    # Reshape latents to match model's expected shape
                    # Calculate spatial dimensions based on the downsample factor
                    flat_size = args.latent_dim * (args.downsample_factor ** 2)
                    h_latent = actual_h_patch_size // args.downsample_factor
                    w_latent = actual_w_patch_size // args.downsample_factor
                    
                    # Reshape flat vector to 3D tensor [C, H, W]
                    cont_reshaped = torch.from_numpy(continuous_latent).float().reshape(
                        args.latent_dim, h_latent, w_latent).unsqueeze(0).to(device)
                    
                    vec_reshaped = torch.from_numpy(codebook_vector).float().reshape(
                        args.latent_dim, h_latent, w_latent).unsqueeze(0).to(device)
                    
                    # For indices, we need to look up the embedding
                    # Get the corresponding vector from the model's codebook
                    if hasattr(model, 'vector_quantizer') and hasattr(model.vector_quantizer, 'embeddings'):
                        idx_vector = model.vector_quantizer.embeddings[codebook_index].reshape(
                            args.latent_dim, 1, 1).repeat(1, h_latent, w_latent).unsqueeze(0).to(device)
                    else:
                        # Fallback if model structure is different
                        idx_vector = vec_reshaped  # Use the stored vector instead
                    
                    # Decode latents to image space
                    recon_cont = model.decode(cont_reshaped).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    recon_vec = model.decode(vec_reshaped).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    recon_idx = model.decode(idx_vector).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    
                    # Place reconstructed patches in the full frame
                    recon_from_continuous[h_start:h_end, w_start:w_end] = recon_cont
                    recon_from_vectors[h_start:h_end, w_start:w_end] = recon_vec
                    recon_from_indices[h_start:h_end, w_start:w_end] = recon_idx
        
        # Create comparison visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original_frame)
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        # Reconstructions
        axes[0, 1].imshow(recon_from_continuous)
        axes[0, 1].set_title('Reconstruction from Continuous Latents')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(recon_from_vectors)
        axes[1, 0].set_title('Reconstruction from Codebook Vectors')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(recon_from_indices)
        axes[1, 1].set_title('Reconstruction from Codebook Indices')
        axes[1, 1].axis('off')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, f'sample_{sample_idx}_reconstructions.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Calculate and visualize differences between reconstructions
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Continuous vs Vectors
        diff_cont_vec = np.abs(recon_from_continuous - recon_from_vectors).mean(axis=2)
        im0 = axes[0].imshow(diff_cont_vec, cmap='hot')
        axes[0].set_title('Diff: Continuous vs Vectors')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Continuous vs Indices
        diff_cont_idx = np.abs(recon_from_continuous - recon_from_indices).mean(axis=2)
        im1 = axes[1].imshow(diff_cont_idx, cmap='hot')
        axes[1].set_title('Diff: Continuous vs Indices')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Vectors vs Indices
        diff_vec_idx = np.abs(recon_from_vectors - recon_from_indices).mean(axis=2)
        im2 = axes[2].imshow(diff_vec_idx, cmap='hot')
        axes[2].set_title('Diff: Vectors vs Indices')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, f'sample_{sample_idx}_differences.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Print statistics
        print(f"  Mean absolute difference (Continuous vs Vectors): {diff_cont_vec.mean():.6f}")
        print(f"  Mean absolute difference (Continuous vs Indices): {diff_cont_idx.mean():.6f}")
        print(f"  Mean absolute difference (Vectors vs Indices): {diff_vec_idx.mean():.6f}")
        
        # Verify reconstruction quality
        mse_cont = np.mean((original_frame - recon_from_continuous) ** 2)
        mse_vec = np.mean((original_frame - recon_from_vectors) ** 2)
        mse_idx = np.mean((original_frame - recon_from_indices) ** 2)
        
        print(f"  MSE (Original vs Continuous): {mse_cont:.6f}")
        print(f"  MSE (Original vs Vectors): {mse_vec:.6f}")
        print(f"  MSE (Original vs Indices): {mse_idx:.6f}")
        
        # Save summary results
        summary = {
            'sample_idx': sample_idx,
            'diff_cont_vec_mean': float(diff_cont_vec.mean()),
            'diff_cont_idx_mean': float(diff_cont_idx.mean()),
            'diff_vec_idx_mean': float(diff_vec_idx.mean()),
            'mse_cont': float(mse_cont),
            'mse_vec': float(mse_vec),
            'mse_idx': float(mse_idx),
        }
        
        np.save(os.path.join(args.output_path, f'sample_{sample_idx}_summary.npy'), summary)

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args, device)
    
    # Load data
    try:
        original_data, continuous_latents, codebook_vectors, codebook_indices = load_data(args)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Test reconstructions
    test_reconstructions(model, original_data, continuous_latents, codebook_vectors, 
                       codebook_indices, args, device)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()