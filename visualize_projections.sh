#!/bin/bash

# Set environment variables for better performance
export PYTHONPATH=`pwd`:$PYTHONPATH
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4

# Initialize variables with defaults
BATCH_INDEX=0
ANALYSIS_TYPE="all"

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --batch_index)
            BATCH_INDEX="$2"
            shift
            shift
            ;;
        --analysis_type)
            ANALYSIS_TYPE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--batch_index INDEX] [--analysis_type TYPE]"
            echo "Analysis types: all, distribution, embedding, temporal, codebook"
            exit 1
            ;;
    esac
done

# Check if projections exist
CONT_PATH="Data_Projections/continuous_latents/batch_${BATCH_INDEX}_continuous.npy"
if [ ! -f "$CONT_PATH" ]; then
    echo "Projections not found for batch $BATCH_INDEX. Please generate them first."
    echo "Run ./run_projections.sh --batch_indices $BATCH_INDEX"
    exit 1
fi

# Create output directory for visualizations
mkdir -p Data_Projections/visualizations

# Run the visualization script
echo "Generating visualizations for batch $BATCH_INDEX (analysis type: $ANALYSIS_TYPE)"
python3 scripts/visualize_projections.py \
    --batch_index "$BATCH_INDEX" \
    --analysis_type "$ANALYSIS_TYPE"

echo "Visualization complete. Results saved to Data_Projections/visualizations/"