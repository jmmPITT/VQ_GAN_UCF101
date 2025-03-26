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
    echo "Usage: $0 --batch_index INDEX [--model MODEL_PATH] [--device cuda|cpu] [--num_chunks NUM]"
    exit 1
fi

# Default settings
BATCH_INDEX=0
DEVICE="cuda"
NUM_CHUNKS=4
CHUNK_INDEX=""
ENCODER_ONLY=false

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
        --num_chunks)
            NUM_CHUNKS="$2"
            shift
            shift
            ;;
        --chunk_index)
            CHUNK_INDEX="--chunk_index $2"
            shift
            shift
            ;;
        --encoder-only)
            ENCODER_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --batch_index INDEX [--model MODEL_PATH] [--device cuda|cpu] [--num_chunks NUM] [--chunk_index INDEX] [--encoder-only]"
            exit 1
            ;;
    esac
done

echo "Running chunked projection for batch $BATCH_INDEX"
echo "Using model: $MODEL_PATH"
echo "Device: $DEVICE"
echo "Number of chunks: $NUM_CHUNKS"
if [ -n "$CHUNK_INDEX" ]; then
    echo "Processing specific chunk: $CHUNK_INDEX"
else
    echo "Processing all chunks"
fi

# Create output directories
mkdir -p Data_Projections/continuous_latents
mkdir -p Data_Projections/codebook_entries
mkdir -p Data_Projections/codebook_indices
mkdir -p Data_Projections/metadata

# Make script executable
chmod +x scripts/chunked_projection.py

# Run the projection script
ENCODER_ONLY_FLAG=""
if [ "$ENCODER_ONLY" = true ]; then
    ENCODER_ONLY_FLAG="--encoder_only"
    echo "Using encoder-only mode (no quantization)"
fi

python3 scripts/chunked_projection.py \
    --model_path "$MODEL_PATH" \
    --batch_index "$BATCH_INDEX" \
    --device "$DEVICE" \
    --num_chunks "$NUM_CHUNKS" \
    $CHUNK_INDEX \
    $ENCODER_ONLY_FLAG

RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo "Chunked projection completed successfully for batch $BATCH_INDEX!"
else
    echo "Error: Projection failed for batch $BATCH_INDEX with exit code $RESULT"
fi