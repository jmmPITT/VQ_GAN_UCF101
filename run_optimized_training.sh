#!/bin/bash

# Create a clean environment for training
export PYTHONPATH=`pwd`:$PYTHONPATH

# Enforce non-interactive Agg backend for matplotlib 
export MPLBACKEND=Agg 

# Limit OpenMP threads
export OMP_NUM_THREADS=4

# Set memory growth and optimizations for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

echo "Starting optimized VQ-GAN training..."

# Run optimized training script with memory-efficient parameters
python3 scripts/train_optimized.py \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --hidden_dims 32 64 128 256 \
    --latent_dim 16 \
    --num_embeddings 256 \
    --num_workers 2 \
    --epochs_per_batch 5 \
    --gan_epochs_per_batch 3 \
    --num_epoch 2 \
    --fp16 \
    --checkpoint_freq 5 \
    --vis_freq 5 \
    --gan_mode \
    "$@"

echo "Training completed!"