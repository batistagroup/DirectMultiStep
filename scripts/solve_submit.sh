#!/bin/bash

# Change values
MODEL_NAME="wide"
TARGET_NAME="chembl"
NUM_PART=10
USE_FP16=true

# Convert USE_FP16 to "true" or "false" as a string
FP16_STR="false"
if [ "$USE_FP16" = true ]; then
    FP16_STR="true"
fi

# Loop through all parts
for i in $(seq 1 ${NUM_PART}); do
    CMD="solve_compounds.py --part ${i} --model_name ${MODEL_NAME} --target_name ${TARGET_NAME} --num_part ${NUM_PART}"
    if [ "$USE_FP16" = true ]; then
        CMD="${CMD} --use_fp16"
    fi
    
    # Submit the job to SLURM
    sbatch --job-name="${TARGET_NAME}_${MODEL_NAME}_${FP16_STR}_${i}" \
           --export="PYTHON_CMD=${CMD}" \
           batch_sub.sub
    echo "Will execute command: python scripts/$CMD"
done
