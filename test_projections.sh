#!/bin/bash

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
export PYTHONPATH=`pwd`:$PYTHONPATH
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4

# Find the best model to use for testing
if [ -f "checkpoints_optimized/best_model_gan.pt" ]; then
    MODEL_PATH="checkpoints_optimized/best_model_gan.pt"
    echo "Using GAN model for testing"
elif [ -f "checkpoints_optimized/best_model_stage1.pt" ]; then
    MODEL_PATH="checkpoints_optimized/best_model_stage1.pt"
    echo "Using stage 1 model for testing"
elif [ -f "checkpoints/best_model_gan.pt" ]; then
    MODEL_PATH="checkpoints/best_model_gan.pt"
    echo "Using original GAN model for testing"
elif [ -f "checkpoints/best_model_stage1.pt" ]; then
    MODEL_PATH="checkpoints/best_model_stage1.pt"
    echo "Using original stage 1 model for testing"
else
    echo "No trained model found. Please train a model first or specify a model path."
    echo "Usage: $0 [--model MODEL_PATH] [--batch_index INDEX]"
    exit 1
fi

# Initialize variables with defaults
BATCH_INDEX=0

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --batch_index)
            BATCH_INDEX="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL_PATH] [--batch_index INDEX]"
            exit 1
            ;;
    esac
done

# First, check if projections exist
CONT_PATH="Data_Projections/continuous_latents/batch_${BATCH_INDEX}_continuous.npy"
if [ ! -f "$CONT_PATH" ]; then
    echo "Projections not found for batch $BATCH_INDEX. Generating them first..."
    
    # Generate projections
    ./run_projections.sh --model "$MODEL_PATH" --batch_indices "$BATCH_INDEX"
    
    # Check if the projections were successfully generated
    if [ ! -f "$CONT_PATH" ]; then
        echo "Failed to generate projections. Aborting test."
        exit 1
    fi
    
    echo "Projections generated successfully."
fi

# Create output directory for test results
mkdir -p Data_Projections/test_results

# Run the test script
echo "Testing projections using model: $MODEL_PATH"
python3 scripts/test_projections.py \
    --model_path "$MODEL_PATH" \
    --batch_index "$BATCH_INDEX" \
    --hidden_dims 32 64 128 256 \
    --latent_dim 16 \
    --num_embeddings 256 \
    --num_samples 2

echo "Test completed. Results saved to Data_Projections/test_results/"