#!/usr/bin/env python3
"""
Project UCF101 data to latent space using the trained VQGAN model.
This script correctly handles the VQGAN architecture and saves both continuous 
latents and quantized vectors with indices.
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import gc
import contextlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the real model
from models.vqgan_optimized import OptimizedVQGAN

def parse_args():
    parser = argparse.ArgumentParser(description='Project UCF101 data using trained VQGAN model')
    parser.add_argument('--data_path', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/data/UCFData',
                        help='Path to UCF101 numpy files')
    parser.add_argument('--output_path', type=str, default='/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections',
                        help='Directory to save projection outputs')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained VQGAN model checkpoint')
    parser.add_argument('--batch_index', type=int, required=True,
                        help='Batch index to process')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Hidden dimensions of the encoder')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Dimension of the latent space')
    parser.add_argument('--num_embeddings', type=int, default=256,
                        help='Number of embeddings in the codebook')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--process_batch_size', type=int, default=4,
                        help='Number of patches to process at once')
    return parser.parse_args()

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
        num_embeddings=args.num_embeddings,
        embedding_dim=args.latent_dim,
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully")
        model.eval()  # Set to evaluation mode
        
        # Force FP32 by converting model parameters
        for param in model.parameters():
            param.data = param.data.float()
            
        # Ensure vector quantizer buffers are float
        if hasattr(model, 'vector_quantizer'):
            model.vector_quantizer.embeddings = model.vector_quantizer.embeddings.float()
            model.vector_quantizer.ema_weight = model.vector_quantizer.ema_weight.float()
            model.vector_quantizer.ema_count = model.vector_quantizer.ema_count.float() 
            model.vector_quantizer.usage = model.vector_quantizer.usage.float()
            print("Converted vector quantizer buffers to float")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load the data
    batch_file = os.path.join(args.data_path, f"ucf101_subset_batch_{args.batch_index}.npy")
    if not os.path.exists(batch_file):
        print(f"Batch file not found: {batch_file}")
        sys.exit(1)
    
    print(f"Loading data from {batch_file}")
    try:
        video_data = np.load(batch_file)
        print(f"Data loaded with shape {video_data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Get dimensions
    batch_size, seq_length, height, width, channels = video_data.shape
    
    # Define patch dimensions
    patch_size_h = 80  # Height of each patch
    patch_size_w = 80  # Width of each patch
    
    # Calculate number of patches
    n_patches_h = height // patch_size_h
    n_patches_w = width // patch_size_w
    
    print(f"Video will be split into {n_patches_h}x{n_patches_w} patches of size {patch_size_h}x{patch_size_w}")
    
    # First, determine output shapes with a test patch
    print("Running test to determine output dimensions...")
    
    # Get a test patch
    test_patch = video_data[0, 0, :patch_size_h, :patch_size_w].astype(np.float32) / 255.0
    test_tensor = torch.from_numpy(test_patch).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # Get encoder output for dimension check
    with torch.no_grad():
        # Start with encoder only as the default, safest option
        use_full_model = False
        
        # First try encoder only - this should always work
        try:
            # Use model.encoder directly - this avoids quantization
            test_encoder_output = model.encoder(test_tensor)
            encoder_output_shape = test_encoder_output.shape
            encoder_flat_size = int(np.prod(encoder_output_shape[1:]))
            print(f"Encoder output shape: {encoder_output_shape}, flat size: {encoder_flat_size}")
            
            # Now try with quantization if encoder worked
            try:
                # Fix dtype issue - ensure VQ model embeddings are the same type as encoder output
                if hasattr(model, 'vector_quantizer') and hasattr(model.vector_quantizer, 'embeddings'):
                    # Get current dtype of encoder output
                    encoder_dtype = test_encoder_output.dtype
                    # Convert embeddings to same dtype if needed
                    if model.vector_quantizer.embeddings.dtype != encoder_dtype:
                        print(f"Converting embeddings from {model.vector_quantizer.embeddings.dtype} to {encoder_dtype}")
                        model.vector_quantizer.embeddings = model.vector_quantizer.embeddings.to(encoder_dtype)
                        if hasattr(model.vector_quantizer, 'ema_weight'):
                            model.vector_quantizer.ema_weight = model.vector_quantizer.ema_weight.to(encoder_dtype)
                
                # Try to get full encoder-quantizer-decoder path
                test_reconstructions, test_quantized, test_loss = model(test_tensor)
                print(f"Full model forward pass successful!")
                
                # Try to get quantized representations directly
                test_z, test_quantized, test_loss, test_indices = model.encode(test_tensor)
                quantized_shape = test_quantized.shape
                quantized_flat_size = int(np.prod(quantized_shape[1:]))
                print(f"Quantized output shape: {quantized_shape}, flat size: {quantized_flat_size}")
                
                # Check if indices is scalar or tensor
                if hasattr(test_indices, 'shape'):
                    print(f"Indices shape: {test_indices.shape}")
                else:
                    print(f"Indices is scalar")
                
                # If we got here, we can use the full model
                use_full_model = True
                
            except Exception as e:
                print(f"Warning: Could not run full VQ model encoding: {e}")
                print("Will use encoder-only projection without quantization")
                use_full_model = False
        except Exception as e:
            print(f"Critical error running even the encoder: {e}")
            sys.exit(1)
    
    # Initialize output arrays
    continuous_latents = np.zeros((batch_size, seq_length, n_patches_h, n_patches_w, encoder_flat_size), 
                               dtype=np.float32)
    
    if use_full_model:
        codebook_vectors = np.zeros((batch_size, seq_length, n_patches_h, n_patches_w, quantized_flat_size), 
                                 dtype=np.float32)
        codebook_indices = np.zeros((batch_size, seq_length, n_patches_h, n_patches_w), 
                                dtype=np.int32)
    
    # Process the data
    process_batch_size = args.process_batch_size
    total_patches = batch_size * seq_length * n_patches_h * n_patches_w
    
    # Disable autocast - create a context manager that does nothing
    class disable_autocast:
        def __enter__(self):
            if hasattr(torch.cuda.amp, 'autocast'):
                self.old_autocast = torch.cuda.amp.autocast
                torch.cuda.amp.autocast = lambda *args, **kwargs: contextlib.nullcontext()
            return self
        
        def __exit__(self, *args):
            if hasattr(self, 'old_autocast'):
                torch.cuda.amp.autocast = self.old_autocast
    
    with torch.no_grad(), disable_autocast():
        
        pbar = tqdm(total=total_patches, desc="Processing patches")
        
        # Process each batch of patches
        for b_idx in range(batch_size):
            for t_idx in range(seq_length):
                
                # Collect patches for batch processing
                patch_batch = []
                positions = []
                
                for h_idx in range(n_patches_h):
                    for w_idx in range(n_patches_w):
                        # Extract patch coordinates
                        h_start = h_idx * patch_size_h
                        w_start = w_idx * patch_size_w
                        h_end = min(h_start + patch_size_h, height)
                        w_end = min(w_start + patch_size_w, width)
                        
                        # Extract patch
                        patch = video_data[b_idx, t_idx, h_start:h_end, w_start:w_end].astype(np.float32) / 255.0
                        
                        # Handle size mismatch at boundaries
                        if patch.shape[0] != patch_size_h or patch.shape[1] != patch_size_w:
                            temp_patch = np.zeros((patch_size_h, patch_size_w, channels), dtype=np.float32)
                            temp_patch[:patch.shape[0], :patch.shape[1]] = patch
                            patch = temp_patch
                        
                        # Convert to tensor
                        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
                        
                        # Add to batch
                        patch_batch.append(patch_tensor)
                        positions.append((h_idx, w_idx))
                        
                        # Process batch when full or at end
                        if len(patch_batch) == process_batch_size or (h_idx == n_patches_h-1 and w_idx == n_patches_w-1):
                            if not patch_batch:
                                continue
                                
                            # Stack patches and move to device
                            batch_tensor = torch.stack(patch_batch).to(device)
                            
                            try:
                                # Get encoder outputs
                                encoder_outputs = model.encoder(batch_tensor)
                                
                                # Process quantization if possible
                                if use_full_model:
                                    try:
                                        # Check and fix dtype if necessary
                                        if hasattr(model, 'vector_quantizer') and hasattr(model.vector_quantizer, 'embeddings'):
                                            encoder_dtype = encoder_outputs.dtype
                                            if model.vector_quantizer.embeddings.dtype != encoder_dtype:
                                                # Only log first time
                                                if b_idx == 0 and t_idx == 0 and len(patch_batch) == process_batch_size:
                                                    print(f"Converting embeddings from {model.vector_quantizer.embeddings.dtype} to {encoder_dtype}")
                                                model.vector_quantizer.embeddings = model.vector_quantizer.embeddings.to(encoder_dtype)
                                                if hasattr(model.vector_quantizer, 'ema_weight'):
                                                    model.vector_quantizer.ema_weight = model.vector_quantizer.ema_weight.to(encoder_dtype)
                                                    
                                        # Try to get quantized representations
                                        z, quantized, _, indices = model.encode(batch_tensor)
                                        have_quantized = True
                                    except Exception as e:
                                        print(f"Warning: Quantization failed: {e}")
                                        have_quantized = False
                                else:
                                    have_quantized = False
                                
                                # Store results for each patch
                                for idx, (h_pos, w_pos) in enumerate(positions):
                                    # Store continuous latents
                                    flat_encoder = encoder_outputs[idx].flatten().cpu().numpy()
                                    if len(flat_encoder) != encoder_flat_size:
                                        print(f"Warning: Encoder output size mismatch: {len(flat_encoder)} vs {encoder_flat_size}")
                                        flat_encoder = np.resize(flat_encoder, encoder_flat_size)
                                    continuous_latents[b_idx, t_idx, h_pos, w_pos] = flat_encoder
                                    
                                    # Store quantized results if available
                                    if have_quantized:
                                        # Store quantized vectors
                                        flat_quantized = quantized[idx].flatten().cpu().numpy()
                                        if len(flat_quantized) != quantized_flat_size:
                                            flat_quantized = np.resize(flat_quantized, quantized_flat_size)
                                        codebook_vectors[b_idx, t_idx, h_pos, w_pos] = flat_quantized
                                        
                                        # Store indices - handle various formats
                                        if hasattr(indices, 'shape') and len(indices.shape) > 1:
                                            # Handle multi-dimensional indices
                                            idx_val = indices[idx].flatten()[0].item()
                                        elif hasattr(indices, 'shape') and len(indices.shape) == 1:
                                            # Handle 1D tensor of indices
                                            idx_val = indices[idx].item()
                                        else:
                                            # Handle scalar or other
                                            idx_val = int(indices[idx]) if hasattr(indices, '__getitem__') else int(indices)
                                        
                                        codebook_indices[b_idx, t_idx, h_pos, w_pos] = idx_val
                                
                                # Update progress
                                pbar.update(len(positions))
                                
                            except Exception as e:
                                print(f"Error processing batch at b={b_idx}, t={t_idx}: {e}")
                                pbar.update(len(positions))  # Still update progress
                            
                            # Clear batch
                            patch_batch = []
                            positions = []
                            
                            # Clean GPU memory periodically
                            if (b_idx * seq_length + t_idx) % 5 == 0:
                                torch.cuda.empty_cache()
        
        pbar.close()
    
    # Save results
    print("Saving projection results...")
    
    # Save continuous latents
    out_cont = os.path.join(args.output_path, 'continuous_latents', f'batch_{args.batch_index}_continuous.npy')
    np.save(out_cont, continuous_latents)
    print(f"Saved continuous latents with shape {continuous_latents.shape}")
    
    # Save quantized results if available
    if use_full_model:
        out_vecs = os.path.join(args.output_path, 'codebook_entries', f'batch_{args.batch_index}_vectors.npy')
        out_idx = os.path.join(args.output_path, 'codebook_indices', f'batch_{args.batch_index}_indices.npy')
        
        np.save(out_vecs, codebook_vectors)
        np.save(out_idx, codebook_indices)
        
        print(f"Saved codebook vectors with shape {codebook_vectors.shape}")
        print(f"Saved codebook indices with shape {codebook_indices.shape}")
    else:
        print("Note: Quantized representations were not saved (encoder-only mode)")
    
    # Save metadata
    metadata = {
        'batch_index': args.batch_index,
        'continuous_shape': continuous_latents.shape,
        'quantized_available': use_full_model,
        'model_path': args.model_path,
        'latent_dim': args.latent_dim,
        'patch_size': (patch_size_h, patch_size_w),
        'original_shape': video_data.shape
    }
    
    if use_full_model:
        metadata.update({
            'vectors_shape': codebook_vectors.shape,
            'indices_shape': codebook_indices.shape,
        })
    
    metadata_file = os.path.join(args.output_path, 'metadata', f'batch_{args.batch_index}_metadata.npy')
    np.save(metadata_file, metadata)
    print(f"Saved metadata to {metadata_file}")
    
    print("Projection complete!")

if __name__ == "__main__":
    main()