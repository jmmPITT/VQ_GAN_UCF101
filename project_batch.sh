#!/bin/bash

# Set environment variables 
export PYTHONPATH=`pwd`:$PYTHONPATH
export OMP_NUM_THREADS=4

# Find the best model to use for projections
if [ -f "checkpoints_optimized/best_model_gan.pt" ]; then
    MODEL_PATH="checkpoints_optimized/best_model_gan.pt"
    echo "Using GAN model for projections"
elif [ -f "checkpoints_optimized/best_model_stage1.pt" ]; then
    MODEL_PATH="checkpoints_optimized/best_model_stage1.pt"
    echo "Using stage 1 model for projections"
elif [ -f "checkpoints/best_model_gan.pt" ]; then
    MODEL_PATH="checkpoints/best_model_gan.pt"
    echo "Using original GAN model for projections"
elif [ -f "checkpoints/best_model_stage1.pt" ]; then
    MODEL_PATH="checkpoints/best_model_stage1.pt"
    echo "Using original stage 1 model for projections"
else
    echo "No trained model found. Please specify a model path with --model."
    echo "Usage: $0 --batch_index INDEX [--model MODEL_PATH] [--device cuda|cpu]"
    exit 1
fi

# Default settings
BATCH_INDEX=0
DEVICE="cuda"
PROCESS_BATCH_SIZE=4
FORCE_ENCODER_ONLY=false

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --batch_index)
            BATCH_INDEX="$2"
            shift
            shift
            ;;
        --model)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --process_batch_size)
            PROCESS_BATCH_SIZE="$2"
            shift
            shift
            ;;
        --encoder-only)
            FORCE_ENCODER_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --batch_index INDEX [--model MODEL_PATH] [--device cuda|cpu] [--process_batch_size SIZE] [--encoder-only]"
            exit 1
            ;;
    esac
done

echo "Running projection for batch $BATCH_INDEX"
echo "Using model: $MODEL_PATH"
echo "Device: $DEVICE"
echo "Process batch size: $PROCESS_BATCH_SIZE"

# Create output directories
mkdir -p Data_Projections/continuous_latents
mkdir -p Data_Projections/codebook_entries
mkdir -p Data_Projections/codebook_indices
mkdir -p Data_Projections/metadata

# Make script executable
chmod +x scripts/project_latents.py

# Print settings
echo "Device: $DEVICE"
echo "Process batch size: $PROCESS_BATCH_SIZE"
if [ "$FORCE_ENCODER_ONLY" = true ]; then
    echo "Forcing encoder-only mode (safer, less memory usage)"
    
    # Create a minimal script to run encoder-only
    cat > scripts/temp_encoder_only.py << EOF
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load model but only use encoder
from models.vqgan_optimized import OptimizedVQGAN

# Set device
device = torch.device("${DEVICE}")
print(f"Using device: {device}")

# Load data
data_path = '/home/jonathan/claude_projects/VQGAN_UCF101/data/UCFData'
batch_file = f"ucf101_subset_batch_${BATCH_INDEX}.npy"
full_path = os.path.join(data_path, batch_file)
print(f"Loading data from {full_path}")
video_data = np.load(full_path)
print(f"Data loaded with shape {video_data.shape}")

# Get dimensions
batch_size, seq_length, height, width, channels = video_data.shape
patch_size_h = 80
patch_size_w = 80
n_patches_h = height // patch_size_h
n_patches_w = width // patch_size_w
print(f"Video will be split into {n_patches_h}x{n_patches_w} patches")

# Initialize model
model = OptimizedVQGAN(
    in_channels=3,
    hidden_dims=[32, 64, 128, 256],
    latent_dim=16
).to(device)

# Load weights
checkpoint = torch.load("${MODEL_PATH}", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Set all parameters to float
for param in model.parameters():
    param.data = param.data.float()

# Get test patch to determine encoder output size
test_patch = video_data[0, 0, :patch_size_h, :patch_size_w].astype(np.float32) / 255.0
test_tensor = torch.from_numpy(test_patch).permute(2, 0, 1).float().unsqueeze(0).to(device)

with torch.no_grad():
    # Get encoder output shape
    encoded = model.encoder(test_tensor)
    print(f"Encoder output shape: {encoded.shape}")
    flat_size = encoded.numel()
    print(f"Flattened size: {flat_size}")
    
    # Create output array
    continuous_latents = np.zeros((batch_size, seq_length, n_patches_h, n_patches_w, flat_size), dtype=np.float32)
    
    # Process all patches
    total_patches = batch_size * seq_length * n_patches_h * n_patches_w
    pbar = tqdm(total=total_patches, desc="Processing patches")
    
    for b in range(batch_size):
        for t in range(seq_length):
            for h in range(n_patches_h):
                for w in range(n_patches_w):
                    # Extract patch
                    h_start = h * patch_size_h
                    w_start = w * patch_size_w
                    h_end = min(h_start + patch_size_h, height)
                    w_end = min(w_start + patch_size_w, width)
                    
                    patch = video_data[b, t, h_start:h_end, w_start:w_end].astype(np.float32) / 255.0
                    
                    # Handle size mismatch at boundaries
                    if patch.shape[0] != patch_size_h or patch.shape[1] != patch_size_w:
                        temp_patch = np.zeros((patch_size_h, patch_size_w, channels), dtype=np.float32)
                        temp_patch[:patch.shape[0], :patch.shape[1]] = patch
                        patch = temp_patch
                    
                    # Convert to tensor
                    patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(device)
                    
                    try:
                        # Get encoder output
                        encoded = model.encoder(patch_tensor)
                        
                        # Store the flattened latent
                        flat_latent = encoded.cpu().view(-1).numpy()
                        continuous_latents[b, t, h, w] = flat_latent
                    except Exception as e:
                        print(f"Error processing patch at b={b}, t={t}, h={h}, w={w}: {e}")
                        # Use zeros as fallback
                        continuous_latents[b, t, h, w] = np.zeros(flat_size, dtype=np.float32)
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Clear memory periodically
                    if b % 2 == 0 and t % 5 == 0:
                        torch.cuda.empty_cache()

# Save the results
os.makedirs('/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections/continuous_latents', exist_ok=True)
os.makedirs('/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections/metadata', exist_ok=True)

# Save continuous latents
out_path = f'/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections/continuous_latents/batch_${BATCH_INDEX}_continuous.npy'
np.save(out_path, continuous_latents)
print(f"Saved continuous latents to {out_path}")

# Save metadata
metadata = {
    'batch_index': ${BATCH_INDEX},
    'shape': continuous_latents.shape,
    'model_path': "${MODEL_PATH}",
    'encoder_only': True,
    'original_shape': video_data.shape
}
meta_path = f'/home/jonathan/claude_projects/VQGAN_UCF101/Data_Projections/metadata/batch_${BATCH_INDEX}_metadata.npy'
np.save(meta_path, metadata)
print(f"Saved metadata to {meta_path}")
EOF
    
    # Run encoder-only version
    python3 scripts/temp_encoder_only.py
    
else
    # Run the full projection script
    python3 scripts/project_latents.py \
        --model_path "$MODEL_PATH" \
        --batch_index "$BATCH_INDEX" \
        --device "$DEVICE" \
        --process_batch_size "$PROCESS_BATCH_SIZE"
fi

RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo "Projection completed successfully for batch $BATCH_INDEX!"
else
    echo "Error: Projection failed for batch $BATCH_INDEX with exit code $RESULT"
fi