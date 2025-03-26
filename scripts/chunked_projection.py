#!/usr/bin/env python3
"""
Memory-efficient projection script that processes UCF101 batch data in chunks.
Loads and processes 1/4 of the batch at a time to avoid memory crashes.
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import gc
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model
from models.vqgan_optimized import OptimizedVQGAN

def parse_args():
    parser = argparse.ArgumentParser(description='Process UCF101 data in chunks')
    parser.add_argument('--data_path', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/data/UCFData',
                        help='Path to UCF101 numpy files')
    parser.add_argument('--output_path', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections',
                        help='Directory to save projection outputs')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained VQGAN model checkpoint')
    parser.add_argument('--batch_index', type=int, required=True,
                        help='Batch index to process')
    parser.add_argument('--num_chunks', type=int, default=4,
                        help='Number of chunks to split the batch into')
    parser.add_argument('--chunk_index', type=int, default=None,
                        help='Specific chunk to process (if None, process all chunks)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Hidden dimensions of the encoder')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Dimension of the latent space')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--encoder_only', action='store_true',
                        help='Only run the encoder (no quantization)')
    return parser.parse_args()

def process_chunk(model, chunk_data, chunk_idx, args, device):
    """Process a single chunk of the data"""
    print(f"Processing chunk {chunk_idx}, shape: {chunk_data.shape}")
    
    # Get dimensions
    chunk_size, seq_length, height, width, channels = chunk_data.shape
    
    # Define patch dimensions
    patch_size_h = 80
    patch_size_w = 80
    n_patches_h = height // patch_size_h
    n_patches_w = width // patch_size_w
    
    # First, determine output shapes with a test patch
    test_patch = chunk_data[0, 0, :patch_size_h, :patch_size_w].astype(np.float32) / 255.0
    test_tensor = torch.from_numpy(test_patch).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # Get encoder output for dimension check
    with torch.no_grad():
        # Get encoder output shape
        test_encoder_output = model.encoder(test_tensor)
        encoder_output_shape = test_encoder_output.shape
        encoder_flat_size = int(np.prod(encoder_output_shape[1:]))
        print(f"Encoder output shape: {encoder_output_shape}, flat size: {encoder_flat_size}")
        
        # Check if we should try quantization
        if not args.encoder_only:
            try:
                # Fix dtype issue
                if hasattr(model, 'vector_quantizer'):
                    model.vector_quantizer.embeddings = model.vector_quantizer.embeddings.float()
                    model.vector_quantizer.ema_weight = model.vector_quantizer.ema_weight.float()
                    model.vector_quantizer.ema_count = model.vector_quantizer.ema_count.float()
                    model.vector_quantizer.usage = model.vector_quantizer.usage.float()
                
                # Test full forward pass
                test_recon, test_quantized, test_loss = model(test_tensor)
                print("Full model forward pass successful!")
                
                # Test encode method
                test_z, test_quantized, test_loss, test_indices = model.encode(test_tensor)
                quantized_shape = test_quantized.shape
                quantized_flat_size = int(np.prod(quantized_shape[1:]))
                print(f"Quantized output shape: {quantized_shape}, flat size: {quantized_flat_size}")
                
                use_full_model = True
            except Exception as e:
                print(f"Warning: Could not run full model: {e}")
                print("Using encoder-only mode")
                use_full_model = False
        else:
            print("Encoder-only mode requested")
            use_full_model = False
    
    # Initialize output arrays
    continuous_latents = np.zeros((chunk_size, seq_length, n_patches_h, n_patches_w, encoder_flat_size), 
                                dtype=np.float32)
    
    if use_full_model:
        codebook_vectors = np.zeros((chunk_size, seq_length, n_patches_h, n_patches_w, quantized_flat_size), 
                                  dtype=np.float32)
        codebook_indices = np.zeros((chunk_size, seq_length, n_patches_h, n_patches_w), 
                                 dtype=np.int32)
    
    # Process all patches
    with torch.no_grad():
        total_patches = chunk_size * seq_length * n_patches_h * n_patches_w
        pbar = tqdm(total=total_patches, desc=f"Processing chunk {chunk_idx}")
        
        for b_idx in range(chunk_size):
            for t_idx in range(seq_length):
                # Process each patch
                for h_idx in range(n_patches_h):
                    for w_idx in range(n_patches_w):
                        # Extract patch coordinates
                        h_start = h_idx * patch_size_h
                        w_start = w_idx * patch_size_w
                        h_end = min(h_start + patch_size_h, height)
                        w_end = min(w_start + patch_size_w, width)
                        
                        # Extract patch
                        patch = chunk_data[b_idx, t_idx, h_start:h_end, w_start:w_end].astype(np.float32) / 255.0
                        
                        # Handle size mismatch at boundaries
                        if patch.shape[0] != patch_size_h or patch.shape[1] != patch_size_w:
                            temp_patch = np.zeros((patch_size_h, patch_size_w, channels), dtype=np.float32)
                            temp_patch[:patch.shape[0], :patch.shape[1]] = patch
                            patch = temp_patch
                        
                        # Convert to tensor
                        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(device)
                        
                        try:
                            # Get encoder output
                            encoder_output = model.encoder(patch_tensor)
                            
                            # Store continuous latent
                            flat_encoder = encoder_output.cpu().flatten().numpy()
                            continuous_latents[b_idx, t_idx, h_idx, w_idx] = flat_encoder
                            
                            # Get quantized representations if using full model
                            if use_full_model:
                                try:
                                    # Get quantized representations
                                    _, quantized, _, indices = model.encode(patch_tensor)
                                    
                                    # Store quantized vector
                                    flat_quantized = quantized.cpu().flatten().numpy()
                                    codebook_vectors[b_idx, t_idx, h_idx, w_idx] = flat_quantized
                                    
                                    # Store index - handle different formats
                                    if hasattr(indices, 'shape') and len(indices.shape) > 0:
                                        # Get first element if multi-dimensional
                                        idx_val = indices.flatten()[0].item()
                                    else:
                                        # Handle scalar
                                        idx_val = int(indices.item())
                                    
                                    codebook_indices[b_idx, t_idx, h_idx, w_idx] = idx_val
                                    
                                except Exception as e:
                                    print(f"Warning: Quantization failed for patch at position ({b_idx}, {t_idx}, {h_idx}, {w_idx}): {e}")
                                    # Use zeros as fallback
                                    codebook_vectors[b_idx, t_idx, h_idx, w_idx] = np.zeros(quantized_flat_size)
                                    codebook_indices[b_idx, t_idx, h_idx, w_idx] = 0
                            
                        except Exception as e:
                            print(f"Error processing patch at position ({b_idx}, {t_idx}, {h_idx}, {w_idx}): {e}")
                            # Use zeros as fallback
                            continuous_latents[b_idx, t_idx, h_idx, w_idx] = np.zeros(encoder_flat_size)
                            if use_full_model:
                                codebook_vectors[b_idx, t_idx, h_idx, w_idx] = np.zeros(quantized_flat_size)
                                codebook_indices[b_idx, t_idx, h_idx, w_idx] = 0
                        
                        # Update progress
                        pbar.update(1)
                
                # Clear GPU memory periodically
                if (b_idx % 2 == 0) and device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        pbar.close()
    
    # Save results
    cont_path = os.path.join(args.output_path, 'continuous_latents', 
                           f'batch_{args.batch_index}_chunk_{chunk_idx}_continuous.npy')
    np.save(cont_path, continuous_latents)
    print(f"Saved continuous latents: {cont_path}")
    
    if use_full_model:
        vecs_path = os.path.join(args.output_path, 'codebook_entries', 
                               f'batch_{args.batch_index}_chunk_{chunk_idx}_vectors.npy')
        idx_path = os.path.join(args.output_path, 'codebook_indices', 
                              f'batch_{args.batch_index}_chunk_{chunk_idx}_indices.npy')
        
        np.save(vecs_path, codebook_vectors)
        np.save(idx_path, codebook_indices)
        
        print(f"Saved codebook vectors: {vecs_path}")
        print(f"Saved codebook indices: {idx_path}")
    
    # Save metadata
    metadata = {
        'batch_index': args.batch_index,
        'chunk_index': chunk_idx,
        'chunk_size': chunk_size,
        'continuous_shape': continuous_latents.shape,
        'model_path': args.model_path,
        'latent_dim': args.latent_dim,
        'use_full_model': use_full_model,
        'original_shape': chunk_data.shape
    }
    
    if use_full_model:
        metadata.update({
            'vectors_shape': codebook_vectors.shape,
            'indices_shape': codebook_indices.shape,
        })
    
    meta_path = os.path.join(args.output_path, 'metadata', 
                           f'batch_{args.batch_index}_chunk_{chunk_idx}_metadata.npy')
    np.save(meta_path, metadata)
    print(f"Saved metadata: {meta_path}")
    
    # Clean up
    del continuous_latents
    if use_full_model:
        del codebook_vectors
        del codebook_indices
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return use_full_model

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(os.path.join(args.output_path, 'continuous_latents'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'codebook_entries'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'codebook_indices'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'metadata'), exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    model = OptimizedVQGAN(
        in_channels=3,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_embeddings=256,  # Use default
        embedding_dim=args.latent_dim,
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully")
        model.eval()
        
        # Convert model to float32
        for param in model.parameters():
            param.data = param.data.float()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load the full batch data
    batch_file = os.path.join(args.data_path, f"ucf101_subset_batch_{args.batch_index}.npy")
    if not os.path.exists(batch_file):
        print(f"Batch file not found: {batch_file}")
        sys.exit(1)
    
    print(f"Loading data from {batch_file}")
    try:
        # Load data but don't keep it all in memory
        full_data = np.load(batch_file, mmap_mode='r')
        batch_size = full_data.shape[0]
        print(f"Full data shape: {full_data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Calculate chunk sizes
    chunk_size = math.ceil(batch_size / args.num_chunks)
    print(f"Splitting into {args.num_chunks} chunks of size {chunk_size}")
    
    # Create summary files to track progress
    summary_path = os.path.join(args.output_path, f'batch_{args.batch_index}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Batch {args.batch_index} Processing Summary\n")
        f.write(f"Total samples: {batch_size}\n")
        f.write(f"Number of chunks: {args.num_chunks}\n")
        f.write(f"Chunk size: {chunk_size}\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Device: {args.device}\n")
        f.write("-" * 50 + "\n\n")
    
    # Process specified chunk or all chunks
    chunks_to_process = [args.chunk_index] if args.chunk_index is not None else range(args.num_chunks)
    
    for chunk_idx in chunks_to_process:
        # Calculate chunk range
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, batch_size)
        actual_chunk_size = end_idx - start_idx
        
        print(f"\nProcessing chunk {chunk_idx}/{args.num_chunks-1} (samples {start_idx}:{end_idx})")
        
        if start_idx >= batch_size:
            print(f"Chunk {chunk_idx} is beyond batch size {batch_size}, skipping")
            continue
        
        # Load only this chunk into memory
        try:
            chunk_data = full_data[start_idx:end_idx].copy()  # Make a copy to avoid mmap issues
            print(f"Loaded chunk {chunk_idx} with shape {chunk_data.shape}")
        except Exception as e:
            print(f"Error loading chunk {chunk_idx}: {e}")
            continue
        
        # Process the chunk
        success = False
        try:
            use_full_model = process_chunk(model, chunk_data, chunk_idx, args, device)
            success = True
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {e}")
        
        # Update summary file
        with open(summary_path, 'a') as f:
            f.write(f"Chunk {chunk_idx} ({start_idx}:{end_idx}):\n")
            f.write(f"  Status: {'Success' if success else 'Failed'}\n")
            if success:
                f.write(f"  Mode: {'Full model' if use_full_model else 'Encoder only'}\n")
            f.write("\n")
        
        # Clean up chunk data
        del chunk_data
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"Completed chunk {chunk_idx}")
    
    print("\nAll specified chunks processed!")

if __name__ == "__main__":
    main()