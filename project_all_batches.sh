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
    echo "Usage: $0 [--model MODEL_PATH] [--device cuda|cpu] [--start INDEX] [--end INDEX]"
    exit 1
fi

# Default settings
DEVICE="cuda"
PROCESS_BATCH_SIZE=4
START_BATCH=0
END_BATCH=13

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
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
        --start)
            START_BATCH="$2"
            shift
            shift
            ;;
        --end)
            END_BATCH="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL_PATH] [--device cuda|cpu] [--process_batch_size SIZE] [--start INDEX] [--end INDEX]"
            exit 1
            ;;
    esac
done

echo "Running projections for batches $START_BATCH to $END_BATCH"
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

# Track success and failure
TOTAL=0
SUCCESS=0
FAILED=0

# Process each batch
for ((batch_idx=START_BATCH; batch_idx<=END_BATCH; batch_idx++)); do
    echo ""
    echo "==============================================="
    echo "Processing batch $batch_idx"
    echo "==============================================="
    
    # Check if batch file exists
    if [ ! -f "/home/jonathan/claude_projects/VQGAN_UCF101/data/UCFData/ucf101_subset_batch_${batch_idx}.npy" ]; then
        echo "Warning: Batch file for index $batch_idx not found. Skipping."
        continue
    fi
    
    TOTAL=$((TOTAL+1))
    
    # Run the projection script for this batch
    python3 scripts/project_latents.py \
        --model_path "$MODEL_PATH" \
        --batch_index "$batch_idx" \
        --device "$DEVICE" \
        --process_batch_size "$PROCESS_BATCH_SIZE"
    
    # Check if projection was successful
    if [ $? -eq 0 ] && [ -f "Data_Projections/continuous_latents/batch_${batch_idx}_continuous.npy" ]; then
        echo "Projection successful for batch $batch_idx"
        SUCCESS=$((SUCCESS+1))
    else
        echo "Error: Projection failed for batch $batch_idx"
        FAILED=$((FAILED+1))
    fi
    
    # Sleep briefly to allow the system to recover
    sleep 2
    
    # Clean CUDA cache if using GPU
    if [ "$DEVICE" = "cuda" ]; then
        python3 -c "import torch; torch.cuda.empty_cache()" || true
    fi
done

echo ""
echo "All projections complete!"
echo "Results: $SUCCESS/$TOTAL successful, $FAILED failed"
echo "Files have been saved to the Data_Projections directory."