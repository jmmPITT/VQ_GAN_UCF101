# VQGAN for UCF101 Dataset

An implementation of VQ-GAN (Vector Quantized Generative Adversarial Network) for the UCF101 video dataset. This project focuses on creating high-fidelity reconstructions with a compressed latent space.

## Features

- VQ-GAN architecture with compressed latent representation
- Adversarial training for improved visual quality
- Perceptual loss (LPIPS) for better reconstruction fidelity
- Patch-based processing for handling video frames
- Visualization tools for latent space and reconstructions
- Advanced codebook usage tracking and "dead" codebook revival
- Attention mechanisms for improved spatial coherence

## Overview

This project improves upon a previous VQ-VAE implementation by:
1. Compressing the latent space dimension (more efficient encoding)
2. Adding a GAN discriminator for higher visual quality
3. Using perceptual losses for better reconstruction fidelity
4. Improved architecture with residual and attention blocks

The VQGAN combines the benefits of vector quantization with adversarial training. The vector quantizer maps continuous latent representations to discrete codebook entries, providing a compressed representation. The GAN component then helps produce more realistic reconstructions by training a discriminator to distinguish real images from reconstructions.

## Architecture

The model architecture consists of the following components:

1. **Encoder**: Converts input images to continuous latent representations
   - Uses strided convolutions for downsampling
   - Residual blocks for improved gradient flow
   - Optional attention blocks for long-range dependencies

2. **Vector Quantizer**: Maps continuous latent vectors to discrete codebook entries
   - Maintains a codebook of embeddings
   - Uses EMA updates for stable training
   - Includes "dead" codebook revival mechanism

3. **Decoder**: Reconstructs images from quantized latent representations
   - Uses transposed convolutions for upsampling
   - Residual blocks for improved gradient flow
   - Optional attention blocks for spatial coherence

4. **Discriminator**: PatchGAN discriminator for adversarial training
   - Uses spectral normalization for stability
   - Operates on patches rather than entire images

## Directory Structure

- `data/` - UCF101 dataset (symlinked to avoid duplication)
- `models/` - PyTorch model implementations
  - `vqgan.py` - VQGAN model implementation
- `scripts/` - Training and visualization scripts
  - `train.py` - Main training script
  - `visualize.py` - Visualization tools
  - `run_testing.sh` - Quick test script
- `utils/` - Utility functions and data processing
  - `data_utils.py` - Data loading and processing
  - `training_utils.py` - Training functions and utilities
- `checkpoints/` - Saved model weights
- `results/` - Generated outputs and visualizations
- `logs/` - Training logs for tensorboard

## Training Process

The training process consists of two main stages:

### Stage 1: VQ-VAE with Reconstruction + Perceptual Loss
- Train the model with MSE reconstruction loss
- Add perceptual loss (LPIPS) for improved visual quality
- Focus on learning good latent representations

### Stage 2: GAN Training
- Introduce a discriminator for adversarial training
- Continue optimizing reconstruction and perceptual losses
- Balance generator and discriminator training for stability

## Implementation Details

### Vector Quantizer
- Uses Exponential Moving Average (EMA) for codebook updates
- Implements codebook usage tracking to identify "dead" codes
- Automatically restarts unused embeddings for better codebook utilization

### Attention Mechanisms
- Self-attention blocks in deeper layers of both encoder and decoder
- Helps maintain global coherence and capture long-range dependencies

### Perceptual Loss
- Uses LPIPS (Learned Perceptual Image Patch Similarity) for perceptual quality
- Better captures human perception of image similarity than MSE alone

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd VQGAN_UCF101

# Install dependencies
pip install -r requirements.txt
```

### Quick Test

Run a quick test to ensure everything is working correctly:

```bash
# Run the testing script
./scripts/run_testing.sh
```

This will train a small model on a subset of the data and generate visualizations.

### Full Training

To train the full model:

```bash
# Train the model with default parameters
python scripts/train.py

# Or with custom parameters
python scripts/train.py \
    --data_path data/UCFData \
    --test_batch 13 \
    --batch_size 64 \
    --hidden_dims 64 128 256 512 \
    --latent_dim 32 \
    --num_embeddings 512 \
    --num_epoch 3 \
    --epochs_per_batch 10 \
    --gan_epochs_per_batch 5 \
    --perceptual_weight 0.1 \
    --adv_weight 0.1
```

### Visualization

Generate visualizations from a trained model:

```bash
python scripts/visualize.py \
    --data_path data/UCFData \
    --batch_idx 0 \
    --checkpoint checkpoints/best_model_gan.pt \
    --output_dir results/visualization
```

## Key Parameters

- `hidden_dims`: Dimensions of hidden layers in encoder/decoder
- `latent_dim`: Dimension of latent space (compressed representation)
- `num_embeddings`: Number of codebook entries
- `downsample_factor`: Downsampling factor in the encoder/decoder
- `perceptual_weight`: Weight for perceptual loss
- `adv_weight`: Weight for adversarial loss

## Results

The VQGAN model achieves better reconstruction quality than the original VQ-VAE implementation, with:
- More efficient compression (smaller latent space)
- Higher visual fidelity due to adversarial and perceptual losses
- Better spatial coherence from attention mechanisms

## Visualizations

The visualization script generates several types of visualizations:
1. Original vs. reconstructed images
2. Latent space projections and statistics
3. Codebook usage and visualization
4. Latent space interpolation between samples
5. Video sequence reconstructions

## Future Work

- Implement conditional generation based on class labels
- Add temporal modeling for video generation
- Explore hierarchical VQ-GAN for multi-scale representations
- Incorporate text-to-image generation capabilities

## Acknowledgments

This project builds upon the original VQGAN paper:
- *Taming Transformers for High-Resolution Image Synthesis* by Esser et al. (2021)

And draws inspiration from:
- *Neural Discrete Representation Learning* by van den Oord et al. (2017)
- *High-Fidelity Generative Image Compression* by Mentzer et al. (2020)