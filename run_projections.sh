#!/bin/bash

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
export PYTHONPATH=`pwd`:$PYTHONPATH
export MPLBACKEND=Agg
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
    echo "No trained model found. Please train a model first or specify a model path."
    echo "Usage: $0 [--model MODEL_PATH] [--batch_indices INDICES]"
    exit 1
fi

# Process command-line arguments
BATCH_INDICES=""
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --batch_indices)
            BATCH_INDICES="--batch_indices $2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL_PATH] [--batch_indices INDICES]"
            exit 1
            ;;
    esac
done

# Run the projection script
echo "Generating projections using model: $MODEL_PATH"
python3 scripts/generate_data_projections.py \
    --model_path "$MODEL_PATH" \
    --hidden_dims 32 64 128 256 \
    --latent_dim 16 \
    --num_embeddings 256 \
    --batch_size 8 \
    --fp16 \
    $BATCH_INDICES

echo "Projection generation complete!"