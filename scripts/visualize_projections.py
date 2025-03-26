#!/usr/bin/env python3
"""
Utility script to visualize and analyze data projections.
This script:
1. Creates visualizations of latent space distributions
2. Generates t-SNE/PCA plots of latent representations
3. Analyzes codebook usage statistics
4. Creates motion analysis visualizations across frames
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize and analyze data projections')
    
    # Data parameters
    parser.add_argument('--projection_path', type=str, 
                        default='/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections',
                        help='Path to projection files')
    parser.add_argument('--output_path', type=str, 
                        default='/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections/visualizations',
                        help='Path to save visualizations')
    
    # Analysis parameters
    parser.add_argument('--batch_index', type=int, default=0,
                        help='Batch index to analyze')
    parser.add_argument('--analysis_type', type=str, choices=['all', 'distribution', 'embedding', 'temporal', 'codebook'],
                        default='all', help='Type of analysis to perform')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                        help='Specific sample indices to analyze (default: first 5)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of random samples to analyze (if sample_indices not specified)')
    
    return parser.parse_args()

def load_projection_data(args):
    """Load projection data for analysis"""
    # Projection data paths
    cont_path = os.path.join(args.projection_path, 'continuous_latents', 
                            f'batch_{args.batch_index}_continuous.npy')
    vecs_path = os.path.join(args.projection_path, 'codebook_entries', 
                            f'batch_{args.batch_index}_vectors.npy')
    idx_path = os.path.join(args.projection_path, 'codebook_indices', 
                           f'batch_{args.batch_index}_indices.npy')
    meta_path = os.path.join(args.projection_path, 'metadata', 
                            f'batch_{args.batch_index}_metadata.npy')
    
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
    
    # Load metadata if available
    metadata = None
    if os.path.exists(meta_path):
        try:
            metadata = np.load(meta_path, allow_pickle=True).item()
            print("Metadata loaded successfully")
        except:
            print("Warning: Could not load metadata")
    
    print(f"Continuous latents shape: {continuous_latents.shape}")
    print(f"Codebook vectors shape: {codebook_vectors.shape}")
    print(f"Codebook indices shape: {codebook_indices.shape}")
    
    return continuous_latents, codebook_vectors, codebook_indices, metadata

def analyze_latent_distributions(continuous_latents, codebook_vectors, args):
    """Visualize the distribution of latent representations"""
    print("Analyzing latent distributions...")
    output_dir = os.path.join(args.output_path, 'distributions')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions
    batch_size, seq_length, patch_h, patch_w, latent_size = continuous_latents.shape
    
    # Flatten the latent vectors for distribution analysis
    flat_continuous = continuous_latents.reshape(-1, latent_size)
    flat_vectors = codebook_vectors.reshape(-1, latent_size)
    
    # Select random subset for visualization if too large
    max_samples = 10000
    if flat_continuous.shape[0] > max_samples:
        indices = np.random.choice(flat_continuous.shape[0], max_samples, replace=False)
        flat_continuous = flat_continuous[indices]
        flat_vectors = flat_vectors[indices]
    
    # Generate overall statistics
    means_cont = np.mean(flat_continuous, axis=0)
    stds_cont = np.std(flat_continuous, axis=0)
    means_vec = np.mean(flat_vectors, axis=0)
    stds_vec = np.std(flat_vectors, axis=0)
    
    # Plot mean and std for each dimension
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(means_cont)), means_cont, yerr=stds_cont, alpha=0.7)
    plt.title('Continuous Latents: Mean ± Std by Dimension')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Value')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(means_vec)), means_vec, yerr=stds_vec, alpha=0.7)
    plt.title('Codebook Vectors: Mean ± Std by Dimension')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Value')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_dimension_stats.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot distribution of values for a few selected dimensions
    num_dims_to_plot = min(5, latent_size)
    selected_dims = np.linspace(0, latent_size-1, num_dims_to_plot, dtype=int)
    
    plt.figure(figsize=(15, 10))
    for i, dim in enumerate(selected_dims):
        plt.subplot(2, num_dims_to_plot, i+1)
        plt.hist(flat_continuous[:, dim], bins=50, alpha=0.7, density=True)
        plt.title(f'Continuous Dim {dim}')
        plt.grid(alpha=0.3)
        
        plt.subplot(2, num_dims_to_plot, i+1+num_dims_to_plot)
        plt.hist(flat_vectors[:, dim], bins=50, alpha=0.7, density=True)
        plt.title(f'Codebook Dim {dim}')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_dimension_distributions.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot correlation matrices
    plt.figure(figsize=(16, 7))
    
    # Continuous latents correlation
    plt.subplot(1, 2, 1)
    corr_cont = np.corrcoef(flat_continuous, rowvar=False)
    plt.imshow(corr_cont, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Continuous Latents Correlation Matrix')
    plt.xticks([])
    plt.yticks([])
    
    # Codebook vectors correlation
    plt.subplot(1, 2, 2)
    corr_vec = np.corrcoef(flat_vectors, rowvar=False)
    plt.imshow(corr_vec, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Codebook Vectors Correlation Matrix')
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_correlation_matrices.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution visualizations saved to {output_dir}")

def generate_embedding_visualizations(continuous_latents, codebook_indices, args):
    """Generate t-SNE and PCA visualizations of the latent space"""
    print("Generating embedding visualizations...")
    output_dir = os.path.join(args.output_path, 'embeddings')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions
    batch_size, seq_length, patch_h, patch_w, latent_size = continuous_latents.shape
    
    # Select a subset of data for visualization
    max_samples = 2000  # Limit to prevent excessive computation time
    
    # Flatten spatial and temporal dimensions, keeping batch information
    reshaped_latents = continuous_latents.reshape(batch_size, seq_length * patch_h * patch_w, latent_size)
    reshaped_indices = codebook_indices.reshape(batch_size, seq_length * patch_h * patch_w)
    
    # Sample random latent vectors from each batch
    sampled_latents = []
    sampled_indices = []
    sampled_batch_ids = []
    samples_per_batch = min(max_samples // batch_size, reshaped_latents.shape[1])
    
    for b in range(batch_size):
        idx = np.random.choice(reshaped_latents.shape[1], samples_per_batch, replace=False)
        sampled_latents.append(reshaped_latents[b, idx])
        sampled_indices.append(reshaped_indices[b, idx])
        sampled_batch_ids.append(np.ones(samples_per_batch) * b)
    
    # Combine samples
    sampled_latents = np.vstack(sampled_latents)
    sampled_indices = np.concatenate(sampled_indices)
    sampled_batch_ids = np.concatenate(sampled_batch_ids)
    
    # Generate t-SNE visualization
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(sampled_latents)
    
    # Generate PCA visualization
    print("Computing PCA projection...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(sampled_latents)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Create t-SNE visualization colored by batch ID
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                         c=sampled_batch_ids, cmap='tab10', alpha=0.7, s=5)
    plt.colorbar(scatter, label='Batch ID')
    plt.title('t-SNE Visualization of Latent Space (Colored by Batch ID)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_tsne_by_batch.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create t-SNE visualization colored by codebook index
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                         c=sampled_indices, cmap='viridis', alpha=0.7, s=5)
    plt.colorbar(scatter, label='Codebook Index')
    plt.title('t-SNE Visualization of Latent Space (Colored by Codebook Index)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_tsne_by_index.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create PCA visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=sampled_indices, cmap='viridis', alpha=0.7, s=5)
    plt.colorbar(scatter, label='Codebook Index')
    plt.title('PCA Visualization of Latent Space (Colored by Codebook Index)')
    plt.xlabel(f'PCA Dimension 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PCA Dimension 2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_pca.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Embedding visualizations saved to {output_dir}")

def analyze_temporal_patterns(continuous_latents, codebook_indices, args):
    """Analyze temporal patterns in the latent space across frames"""
    print("Analyzing temporal patterns...")
    output_dir = os.path.join(args.output_path, 'temporal')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions
    batch_size, seq_length, patch_h, patch_w, latent_size = continuous_latents.shape
    
    # Determine which samples to analyze
    if args.sample_indices is not None:
        sample_indices = [idx for idx in args.sample_indices if idx < batch_size]
        if not sample_indices:
            print("Warning: No valid sample indices provided, using first sample")
            sample_indices = [0]
    else:
        num_samples = min(args.num_samples, batch_size)
        sample_indices = list(range(num_samples))
    
    # For each sample, analyze temporal patterns
    for sample_idx in sample_indices:
        print(f"Analyzing temporal patterns for sample {sample_idx}")
        
        # Extract latent representations for this sample
        sample_latents = continuous_latents[sample_idx]  # Shape: (seq_length, patch_h, patch_w, latent_size)
        sample_indices = codebook_indices[sample_idx]   # Shape: (seq_length, patch_h, patch_w)
        
        # 1. Analyze frame-to-frame similarity
        # Flatten spatial dimensions
        flat_frames = sample_latents.reshape(seq_length, -1)
        
        # Calculate cosine similarity between consecutive frames
        frame_similarities = np.zeros(seq_length - 1)
        for t in range(seq_length - 1):
            # Normalized dot product for cosine similarity
            norm1 = np.linalg.norm(flat_frames[t])
            norm2 = np.linalg.norm(flat_frames[t+1])
            if norm1 > 0 and norm2 > 0:
                frame_similarities[t] = np.dot(flat_frames[t], flat_frames[t+1]) / (norm1 * norm2)
        
        # Plot frame-to-frame similarity
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, seq_length), frame_similarities, 'o-', markersize=8)
        plt.title(f'Frame-to-Frame Similarity (Sample {sample_idx})')
        plt.xlabel('Frame Number')
        plt.ylabel('Cosine Similarity')
        plt.grid(alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_sample_{sample_idx}_frame_similarity.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Compute variance over time for each patch
        # Reshape to (seq_length, patch_h*patch_w, latent_size)
        patches_over_time = sample_latents.reshape(seq_length, patch_h * patch_w, latent_size)
        
        # Compute variance for each patch
        patch_variance = np.var(patches_over_time, axis=0).mean(axis=1)  # Mean variance across latent dimensions
        
        # Reshape to spatial grid for visualization
        patch_variance_grid = patch_variance.reshape(patch_h, patch_w)
        
        # Plot patch variance
        plt.figure(figsize=(8, 6))
        plt.imshow(patch_variance_grid, cmap='viridis')
        plt.colorbar(label='Mean Variance')
        plt.title(f'Patch Variance Over Time (Sample {sample_idx})')
        for i in range(patch_h):
            for j in range(patch_w):
                plt.text(j, i, f'{patch_variance_grid[i, j]:.2f}', 
                        ha='center', va='center', color='white', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_sample_{sample_idx}_patch_variance.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Track codebook index changes over time for a few selected patches
        # Select a few patches to track (e.g., highest variance patches)
        flat_indices = np.argsort(patch_variance)[::-1]  # Sort by descending variance
        num_patches_to_track = min(5, patch_h * patch_w)
        patches_to_track = flat_indices[:num_patches_to_track]
        
        # Get (h, w) coordinates for selected patches
        tracked_patches = [(p // patch_w, p % patch_w) for p in patches_to_track]
        
        # Extract codebook indices for selected patches over time
        tracked_indices = np.zeros((seq_length, num_patches_to_track))
        for t in range(seq_length):
            for i, (h, w) in enumerate(tracked_patches):
                tracked_indices[t, i] = sample_indices[t, h, w]
        
        # Plot index changes over time
        plt.figure(figsize=(12, 6))
        for i in range(num_patches_to_track):
            h, w = tracked_patches[i]
            plt.plot(tracked_indices[:, i], 'o-', label=f'Patch ({h},{w})')
        
        plt.title(f'Codebook Index Changes Over Time (Sample {sample_idx})')
        plt.xlabel('Frame Number')
        plt.ylabel('Codebook Index')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_sample_{sample_idx}_index_changes.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Temporal analysis visualizations saved to {output_dir}")

def analyze_codebook_usage(codebook_indices, args, metadata=None):
    """Analyze codebook usage statistics"""
    print("Analyzing codebook usage...")
    output_dir = os.path.join(args.output_path, 'codebook')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get codebook size from metadata if available
    codebook_size = metadata.get('num_embeddings', 256) if metadata else 256
    
    # Calculate overall codebook usage
    flat_indices = codebook_indices.flatten()
    index_counts = np.bincount(flat_indices.astype(np.int32), minlength=codebook_size)
    
    # Plot overall usage histogram
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(index_counts)), index_counts)
    plt.title(f'Codebook Usage Distribution (Batch {args.batch_index})')
    plt.xlabel('Codebook Index')
    plt.ylabel('Usage Count')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_codebook_usage.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Analyze codebook usage distribution
    used_indices = np.where(index_counts > 0)[0]
    usage_percentage = len(used_indices) / codebook_size * 100
    
    # Plot usage statistics
    plt.figure(figsize=(12, 6))
    
    # Create a sorted usage plot
    sorted_counts = np.sort(index_counts)[::-1]  # Sort in descending order
    plt.bar(range(len(sorted_counts)), sorted_counts)
    plt.title(f'Sorted Codebook Usage (Batch {args.batch_index})')
    plt.xlabel('Rank')
    plt.ylabel('Usage Count')
    plt.grid(alpha=0.3)
    
    # Add annotations
    plt.axhline(y=np.mean(index_counts), color='r', linestyle='-', label=f'Mean: {np.mean(index_counts):.1f}')
    plt.axhline(y=np.median(sorted_counts), color='g', linestyle='--', label=f'Median: {np.median(sorted_counts):.1f}')
    top_10_percent = np.percentile(sorted_counts, 90)
    plt.axhline(y=top_10_percent, color='b', linestyle=':', label=f'90th Percentile: {top_10_percent:.1f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_codebook_usage_sorted.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create heatmap version for visualization of usage distribution
    side_length = int(np.sqrt(codebook_size))
    if side_length**2 == codebook_size:  # Perfect square
        usage_map = index_counts.reshape(side_length, side_length)
        plt.figure(figsize=(10, 10))
        plt.imshow(usage_map, cmap='viridis')
        plt.colorbar(label='Usage Count')
        plt.title(f'Codebook Usage Heatmap (Batch {args.batch_index})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'batch_{args.batch_index}_codebook_usage_heatmap.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save usage statistics as text file
    stats = {
        'total_codebook_size': codebook_size,
        'used_entries': len(used_indices),
        'usage_percentage': usage_percentage,
        'min_usage': np.min(index_counts),
        'max_usage': np.max(index_counts),
        'mean_usage': np.mean(index_counts),
        'median_usage': np.median(index_counts),
        'std_usage': np.std(index_counts),
    }
    
    with open(os.path.join(output_dir, f'batch_{args.batch_index}_codebook_stats.txt'), 'w') as f:
        f.write(f"Codebook Usage Statistics for Batch {args.batch_index}\n")
        f.write(f"----------------------------------------\n")
        f.write(f"Total codebook size: {stats['total_codebook_size']}\n")
        f.write(f"Used entries: {stats['used_entries']} ({stats['usage_percentage']:.2f}%)\n")
        f.write(f"Unused entries: {codebook_size - stats['used_entries']}\n")
        f.write(f"Min usage: {stats['min_usage']}\n")
        f.write(f"Max usage: {stats['max_usage']}\n")
        f.write(f"Mean usage: {stats['mean_usage']:.2f}\n")
        f.write(f"Median usage: {stats['median_usage']:.2f}\n")
        f.write(f"Standard deviation: {stats['std_usage']:.2f}\n")
    
    print(f"Codebook usage statistics saved to {output_dir}")
    print(f"Codebook utilization: {usage_percentage:.2f}% ({len(used_indices)}/{codebook_size} entries used)")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load projection data
    continuous_latents, codebook_vectors, codebook_indices, metadata = load_projection_data(args)
    
    # Run analyses based on user selection
    if args.analysis_type in ['all', 'distribution']:
        analyze_latent_distributions(continuous_latents, codebook_vectors, args)
    
    if args.analysis_type in ['all', 'embedding']:
        generate_embedding_visualizations(continuous_latents, codebook_indices, args)
    
    if args.analysis_type in ['all', 'temporal']:
        analyze_temporal_patterns(continuous_latents, codebook_indices, args)
    
    if args.analysis_type in ['all', 'codebook']:
        analyze_codebook_usage(codebook_indices, args, metadata)
    
    print("\nAnalysis complete! Visualizations saved to", args.output_path)

if __name__ == "__main__":
    main()