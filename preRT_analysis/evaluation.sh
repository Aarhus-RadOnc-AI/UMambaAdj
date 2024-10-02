#!/bin/bash

# Set the base directory where the folds are located
FOLDS_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset501_preRT/nnUNetTrainerUmamba__nnUNetResEncUNetMPlans__3d_fullres_bs4"
VAL_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset501_preRT/nnUNetTrainerUmamba__nnUNetResEncUNetMPlans__3d_fullres_bs4/aggregated_validation"

GT_DIR="/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset501_preRT/labelsTr"
# List of folds
folds=("fold_0" "fold_1" "fold_2" "fold_3" "fold_4")

# Iterate over each fold
for fold in "${folds[@]}"; do
    # Extract the integer part of the fold (e.g., 0, 1, 2, 3, 4)
    fold_num="${fold##*_}"

    # Define the input and output directories
    OUTPUT_FOLDER="${FOLDS_DIR}/${fold}/validation"

    # Print a message indicating which fold and margin are being processed
    echo "evaluating ${fold} margin..."

    # Run the Evaluation
    python evaluation.py "${GT_DIR}" "${OUTPUT_FOLDER}" 

    # Print a message when done
    echo "Evaluation for ${fold}  completed!"

done
#python evaluation.py "${GT_DIR}" "${VAL_DIR}" 

echo "All Evaluation completed!"