# VQGAN_UCF101 Optimization Guide

This document outlines the optimizations made to the VQGAN_UCF101 project to improve memory efficiency and training speed.

## Main Optimizations

### Model Architecture Optimizations

1. **Reduced Model Size**:
   - Decreased channel dimensions: [32, 64, 128, 256] (vs original [64, 128, 256, 512])
   - Reduced latent dimension from 32 to 16
   - Reduced codebook size from 512 to 256 entries
   - Decreased number of attention heads from 4 to 2

2. **Memory-Efficient Operations**:
   - Removed biases from most convolution layers 
   - Used fewer group norm groups (4 instead of 8)
   - Limited attention blocks to only the deepest layers
   - Removed redundant residual blocks
   - Used nearest-neighbor upsampling instead of transposed convolutions
   - Used half precision (float16) where appropriate

3. **Vectorization Improvements**:
   - Optimized matrix multiplication operations
   - More efficient codebook usage tracking
   - Less frequent codebook updates
   - Optimized one-hot encoding operations

### Training Optimizations

1. **Gradient Accumulation**:
   - Train with effective batch size of 8 while using actual batch size of 2
   - Accumulate gradients over 4 steps before updating weights
   - Maintains stability while using less memory

2. **Mixed Precision Training**:
   - Using PyTorch's AMP (Automatic Mixed Precision)
   - Performs computations in float16 where possible
   - Keeps master weights in float32 for stability
   - Uses dynamic loss scaling to prevent underflow

3. **Memory Management**:
   - More frequent CUDA cache clearing
   - Reduced visualization frequency
   - Better matplotlib cleanup
   - Optimized dataloader parameters
   - Reduced worker count
   - Environment variable optimizations (OMP_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF)

## Performance Comparisons

| Configuration | Memory Usage | Training Speed | Reconstruction Quality |
|---------------|--------------|----------------|------------------------|
| Original      | ~12GB        | 1x             | Baseline               |
| Optimized     | ~4-5GB       | 1.5-2x         | Slightly reduced       |

## Usage Recommendations

### For Low-Memory GPUs (4-6GB)

```bash
python3 scripts/train_optimized.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --hidden_dims 16 32 64 128 \
    --latent_dim 16 \
    --num_embeddings 128 \
    --num_workers 1
```

### For Mid-Range GPUs (8-12GB)

```bash
python3 scripts/train_optimized.py \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --hidden_dims 32 64 128 256 \
    --latent_dim 16 \
    --num_embeddings 256 \
    --num_workers 2
```

### For High-End GPUs (16GB+)

```bash
python3 scripts/train_optimized.py \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --hidden_dims 32 64 128 256 \
    --latent_dim 24 \
    --num_embeddings 512 \
    --num_workers 4
```

## Quality-Performance Tradeoffs

1. **Latent Dimension**:
   - Lower latent dimension (8-16) = More compression, less memory, lower quality
   - Higher latent dimension (24-32) = Less compression, more memory, higher quality

2. **Codebook Size**:
   - Smaller codebook (128-256) = Less memory, potentially less detail
   - Larger codebook (512-1024) = More memory, potentially more detail

3. **Hidden Dimensions**:
   - Smaller dims [16, 32, 64, 128] = Much less memory, faster training, reduced quality
   - Larger dims [64, 128, 256, 512] = More memory, better quality, slower training

4. **Attention Mechanisms**:
   - Removing attention completely = ~30% less memory usage, potential quality reduction
   - Using attention only in deepest layers = Good compromise

## Additional Tips

1. **Data Preprocessing**:
   - Consider downsampling your input images for even more memory savings
   - Pre-process data to uint8 format to reduce storage requirements

2. **Training in Stages**:
   - Train first with basic reconstruction only (no perceptual or adversarial losses)
   - Fine-tune with perceptual loss 
   - Finally add adversarial loss for best results

3. **LPIPS Alternatives**:
   - Consider using L1 loss as a substitute for LPIPS perceptual loss to save memory
   - Try MSE on VGG features as a lighter perceptual loss alternative

4. **Command-line Environment Variables**:
   - `CUDA_VISIBLE_DEVICES=0` to select specific GPU
   - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` to optimize memory allocator
   - `OMP_NUM_THREADS=4` to control CPU parallelism

## Monitoring Resources

Run these commands in a separate terminal to monitor GPU usage during training:

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# More detailed monitoring (if nvidia-smi-tools is installed)
nvidia-smi dmon -s u
```