# Data Projections for VQGAN_UCF101

This folder contains latent space projections derived from the UCF101 dataset using the trained VQGAN model. These projections transform the original video data `(num_samples, seq_length, 240, 320, 3)` into more efficient latent representations.

## Directory Structure

```
Data_Projections/
├── continuous_latents/      # Raw encoder outputs before quantization
│   └── batch_X_continuous.npy
├── codebook_entries/        # Quantized vectors from the codebook
│   └── batch_X_vectors.npy
├── codebook_indices/        # Indices into the codebook
│   └── batch_X_indices.npy
├── metadata/                # Information about each batch
│   └── batch_X_metadata.npy
├── test_results/            # Validation of projection quality
│   └── sample_X_*.png
└── visualizations/          # Analysis visualizations
    ├── distributions/
    ├── embeddings/
    ├── temporal/
    └── codebook/
```

## Data Formats

For each batch of the UCF101 dataset:

1. **Continuous Latents** `(num_samples, seq_length, num_patches_h, num_patches_w, latent_dim)`
   - Raw outputs from the encoder before vector quantization
   - Floating-point values representing the continuous latent space

2. **Codebook Entries** `(num_samples, seq_length, num_patches_h, num_patches_w, latent_dim)`
   - Quantized vectors from the codebook
   - Each vector is one of the learned codebook entries

3. **Codebook Indices** `(num_samples, seq_length, num_patches_h, num_patches_w)`
   - Integer indices into the codebook
   - Each index identifies which codebook entry was selected

4. **Metadata** (Python dictionary stored as .npy)
   - Information about the batch and model configuration
   - Contains shape information, latent dimensions, etc.

## Usage

### Generating Projections

To generate projections for a specific batch:

```bash
./run_projections.sh --batch_indices X
```

Where `X` is the batch index (0-12 for UCF101 dataset).

### Testing Projections

To validate the quality of the projections:

```bash
./test_projections.sh --batch_index X
```

This will:
1. Generate reconstructions from each latent format
2. Compare the reconstructions to the original data
3. Save visualizations of the results

### Analyzing Projections

To analyze the latent space properties:

```bash
./visualize_projections.sh --batch_index X --analysis_type TYPE
```

Analysis types include:
- `distribution`: Statistical properties of latent dimensions
- `embedding`: t-SNE and PCA visualizations
- `temporal`: Analysis of how latents change over time
- `codebook`: Codebook usage statistics
- `all`: All of the above

## Notes on Dimensionality

The original data shape is `(num_samples, seq_length, 240, 320, 3)` where:
- `num_samples`: Number of video clips in the batch
- `seq_length`: Number of frames per video (varies by batch)
- `240×320`: Resolution of each frame
- `3`: RGB channels

The projection shape is `(num_samples, seq_length, num_patches_h, num_patches_w, latent_dim)` where:
- `num_patches_h = 3`: Number of patches in height (240/80)
- `num_patches_w = 4`: Number of patches in width (320/80)
- `latent_dim`: Dimension of the latent space (default: 16×16 = 256)

The compression ratio depends on the latent dimension but is typically 8-16× smaller than the original data.

## Examples

Code example to load and use the projections:

```python
import numpy as np
import torch
from models.vqgan_optimized import OptimizedVQGAN

# Load projections
batch_idx = 0
continuous_latents = np.load(f'Data_Projections/continuous_latents/batch_{batch_idx}_continuous.npy')
codebook_indices = np.load(f'Data_Projections/codebook_indices/batch_{batch_idx}_indices.npy')

# Load model for reconstruction
model = OptimizedVQGAN(
    in_channels=3,
    hidden_dims=[32, 64, 128, 256],
    latent_dim=16,
    num_embeddings=256
).to(device)
model.load_state_dict(torch.load('checkpoints_optimized/best_model.pt'))

# Reconstruct from indices
sample_idx = 0
frame_idx = 0
patch_idx_h, patch_idx_w = 1, 2
codebook_idx = codebook_indices[sample_idx, frame_idx, patch_idx_h, patch_idx_w]

# Get the corresponding codebook vector
codebook_vector = model.vector_quantizer.embeddings[codebook_idx]

# Reshape to match expected input shape
latent_h, latent_w = 80 // 4, 80 // 4  # Assuming downsample_factor=4
reshaped_vector = codebook_vector.reshape(1, -1, latent_h, latent_w)

# Decode to image space
with torch.no_grad():
    reconstruction = model.decode(reshaped_vector)
```