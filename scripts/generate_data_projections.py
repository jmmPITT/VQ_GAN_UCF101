#!/usr/bin/env python3
"""
Script to project UCF101 video data to latent space using a trained VQGAN model.
Generates three types of projections:
1. Continuous latent embeddings
2. Codebook vectors
3. Codebook indices
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.amp  # Import torch.amp for autocast
from tqdm import tqdm
import time
import gc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import optimized VQGAN model
from models.vqgan_optimized import OptimizedVQGAN

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data projections from trained VQGAN')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/data/UCFData',
                        help='Path to UCF101 numpy files')
    parser.add_argument('--output_path', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections',
                        help='Directory to save projection outputs')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained VQGAN model checkpoint')
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
                        help='Number of patches in height (240/80=3)')
    parser.add_argument('--num_patches_w', type=int, default=4,
                        help='Number of patches in width (320/80=4)')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--batch_indices', type=int, nargs='+', default=None,
                        help='Specific batch indices to process (default: all)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision for inference')
    
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

def get_batch_files(data_path, batch_indices=None):
    """Get list of batch files to process"""
    import glob
    
    batch_files = sorted(glob.glob(os.path.join(data_path, "ucf101_subset_batch_*.npy")))
    
    if not batch_files:
        print(f"No batch files found in {data_path}")
        sys.exit(1)
    
    if batch_indices is not None:
        # Filter to specific batch indices
        filtered_files = []
        for idx in batch_indices:
            matching_files = [f for f in batch_files if f.endswith(f"batch_{idx}.npy")]
            if matching_files:
                filtered_files.extend(matching_files)
            else:
                print(f"Warning: No file found for batch index {idx}")
        
        if not filtered_files:
            print("No matching batch files found")
            sys.exit(1)
        
        return filtered_files
    
    return batch_files

def process_video_batch(model, video_batch, args, device):
    """Process a batch of videos to extract latent projections"""
    # Expected input: (batch_size, seq_length, H, W, C)
    batch_size, seq_length, height, width, channels = video_batch.shape
    patch_h, patch_w = args.num_patches_h, args.num_patches_w
    
    # Calculate patch dimensions
    actual_h_patch_size = height // patch_h
    actual_w_patch_size = width // patch_w
    
    # Initialize output containers
    # Shape: (batch_size, seq_length, num_patches_h, num_patches_w, latent_dim)
    continuous_latents = np.zeros((batch_size, seq_length, patch_h, patch_w, 
                                   args.latent_dim * (args.downsample_factor**2)), 
                                  dtype=np.float32)
    
    codebook_vectors = np.zeros((batch_size, seq_length, patch_h, patch_w, 
                                args.latent_dim * (args.downsample_factor**2)), 
                               dtype=np.float32)
    
    codebook_indices = np.zeros((batch_size, seq_length, patch_h, patch_w), 
                               dtype=np.int32)
    
    # Process each video frame in the batch
    with torch.no_grad():
        # Use mixed precision for inference if specified
        with torch.amp.autocast('cuda') if args.fp16 and device.type == 'cuda' else nullcontext():
            # Process in smaller batches to avoid OOM
            processing_batch_size = args.batch_size
            
            # Create progress bar for the total workload
            total_patches = batch_size * seq_length * patch_h * patch_w
            pbar = tqdm(total=total_patches, desc="Processing patches")
            
            for b_idx in range(batch_size):
                for s_idx in range(seq_length):
                    patch_batch_data = []
                    patch_positions = []
                    
                    # Collect all patches from this frame
                    for h_idx in range(patch_h):
                        for w_idx in range(patch_w):
                            # Extract patch
                            h_start = h_idx * actual_h_patch_size
                            w_start = w_idx * actual_w_patch_size
                            h_end = min(h_start + actual_h_patch_size, height)
                            w_end = min(w_start + actual_w_patch_size, width)
                            
                            patch = video_batch[b_idx, s_idx, h_start:h_end, w_start:w_end]
                            
                            # Normalize patch to [0, 1]
                            patch = patch.astype(np.float32) / 255.0
                            
                            # Convert to tensor with shape [C, H, W]
                            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
                            
                            patch_batch_data.append(patch_tensor)
                            patch_positions.append((h_idx, w_idx))
                            
                            # Process in mini-batches
                            if len(patch_batch_data) == processing_batch_size or (
                                    h_idx == patch_h - 1 and w_idx == patch_w - 1):
                                
                                # Stack patches and move to device
                                patch_batch = torch.stack(patch_batch_data).to(device)
                                
                                # Get latent projections
                                z, quantized, _, indices = model.encode(patch_batch)
                                
                                # Store results for each patch
                                for i, (ph, pw) in enumerate(patch_positions):
                                    # Flatten spatial dimensions of latent
                                    flat_z = z[i].reshape(-1).cpu().numpy()
                                    flat_q = quantized[i].reshape(-1).cpu().numpy()
                                    
                                    # Handle indices properly - they might be tensors not scalars
                                    if hasattr(indices[i], 'item'):
                                        if indices[i].numel() == 1:
                                            idx = indices[i].item()
                                        else:
                                            # If it's a tensor with multiple values, just use the first one or flatten
                                            idx = indices[i].flatten()[0].item()
                                    else:
                                        # Fallback if item() method is not available
                                        idx = int(indices[i].cpu().numpy().flatten()[0])
                                    
                                    continuous_latents[b_idx, s_idx, ph, pw] = flat_z
                                    codebook_vectors[b_idx, s_idx, ph, pw] = flat_q
                                    codebook_indices[b_idx, s_idx, ph, pw] = idx
                                
                                # Update progress bar
                                pbar.update(len(patch_batch_data))
                                
                                # Clear patch batch
                                patch_batch_data = []
                                patch_positions = []
                                
                                # Clean up GPU memory
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
            
            pbar.close()
    
    return continuous_latents, codebook_vectors, codebook_indices

def save_projections(batch_file, continuous_latents, codebook_vectors, codebook_indices, args):
    """Save projection data to designated folders"""
    # Extract batch number from filename
    import re
    batch_match = re.search(r'batch_(\d+)', os.path.basename(batch_file))
    if batch_match:
        batch_num = batch_match.group(1)
    else:
        # Fallback to index in filename
        batch_num = os.path.basename(batch_file).split('_')[-1].split('.')[0]
    
    print(f"Saving projections for batch {batch_num}")
    
    # Create output paths
    cont_path = os.path.join(args.output_path, 'continuous_latents', f'batch_{batch_num}_continuous.npy')
    vecs_path = os.path.join(args.output_path, 'codebook_entries', f'batch_{batch_num}_vectors.npy')
    idx_path = os.path.join(args.output_path, 'codebook_indices', f'batch_{batch_num}_indices.npy')
    meta_path = os.path.join(args.output_path, 'metadata', f'batch_{batch_num}_metadata.npy')
    
    # Save data
    np.save(cont_path, continuous_latents)
    np.save(vecs_path, codebook_vectors)
    np.save(idx_path, codebook_indices)
    
    # Save metadata
    metadata = {
        'batch_file': os.path.basename(batch_file),
        'shape': continuous_latents.shape,
        'latent_dim': args.latent_dim,
        'downsample_factor': args.downsample_factor,
        'num_embeddings': args.num_embeddings,
        'num_patches_h': args.num_patches_h,
        'num_patches_w': args.num_patches_w,
        'model_path': args.model_path,
    }
    np.save(meta_path, metadata)
    
    print(f"Saved projections for batch {batch_num}:")
    print(f"  Continuous latents: {cont_path}, shape: {continuous_latents.shape}")
    print(f"  Codebook vectors: {vecs_path}, shape: {codebook_vectors.shape}")
    print(f"  Codebook indices: {idx_path}, shape: {codebook_indices.shape}")
    print(f"  Metadata: {meta_path}")

class nullcontext:
    """A context manager that does nothing (for Python < 3.7)"""
    def __enter__(self):
        return None
    def __exit__(self, *excinfo):
        pass

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = load_model(args, device)
    
    # Get batch files to process
    batch_files = get_batch_files(args.data_path, args.batch_indices)
    print(f"Found {len(batch_files)} batch files to process")
    
    # Process each batch file
    for batch_file in batch_files:
        try:
            print(f"\nProcessing {batch_file}")
            
            # Load video data
            start_time = time.time()
            video_data = np.load(batch_file)
            load_time = time.time() - start_time
            print(f"Loaded data with shape {video_data.shape} in {load_time:.2f}s")
            
            # Process the data
            continuous_latents, codebook_vectors, codebook_indices = process_video_batch(
                model, video_data, args, device
            )
            
            # Save projections
            save_projections(batch_file, continuous_latents, codebook_vectors, 
                           codebook_indices, args)
            
            # Clean up memory
            del video_data, continuous_latents, codebook_vectors, codebook_indices
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {batch_file}: {e}")
            continue
    
    print("\nProjection generation complete!")

if __name__ == "__main__":
    main()