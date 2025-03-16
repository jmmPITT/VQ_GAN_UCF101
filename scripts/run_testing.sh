#!/bin/bash

# Script to test the VQGAN model on a small subset of UCF101 data

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Set to your GPU ID

# Create a small test model
echo "Creating test model..."
python scripts/train.py \
    --data_path data/UCFData \
    --test_batch 13 \
    --batch_size 16 \
    --hidden_dims 32 64 128 256 \
    --latent_dim 16 \
    --num_embeddings 256 \
    --num_epoch 1 \
    --epochs_per_batch 2 \
    --gan_epochs_per_batch 2 \
    --downsample_factor 4 \
    --perceptual_weight 0.1 \
    --adv_weight 0.1 \
    --log_dir logs/test \
    --results_dir results/test \
    --checkpoint_dir checkpoints/test

# Run visualization
echo "Running visualization..."
python scripts/visualize.py \
    --data_path data/UCFData \
    --batch_idx 0 \
    --hidden_dims 32 64 128 256 \
    --latent_dim 16 \
    --num_embeddings 256 \
    --downsample_factor 4 \
    --num_samples 4 \
    --output_dir results/test_viz \
    --video_idx 0 \
    --start_frame 0 \
    --num_frames 4 \
    --checkpoint checkpoints/test/best_model_gan.pt

echo "Test complete! Check the results in results/test_viz directory."