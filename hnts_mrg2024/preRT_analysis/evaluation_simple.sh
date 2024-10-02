#!/bin/bash

# Set the base directory where the folds are located
#PRED_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset504_midRT_geodist/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres_bs8/fold_0/margins/4voxels/docker_new"
#PRED_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset504_midRT_geodist/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres_bs8/fold_0/margins/4voxels/docker_preds"
GT_DIR="/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset501_preRT/labelsTr"

# # Run the Evaluation for folds 0 to 4
# for i in {0..4}; do
#   PRED_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset501_preRT/nnUNetTrainerUmamba__nnUNetResEncUNetMPlans__3d_fullres_bs4/fold_${i}/validation"
  
#   # Execute the evaluation
#   python evaluation.py "${GT_DIR}" "${PRED_DIR}"
# done


#PRED_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset501_preRT/nnUNetTrainerResenc__nnUNetResEncUNetMPlans__3d_fullres_bs4/fold_0/validation"
#PRED_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset501_preRT/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres_bs4/fold_0/validation"
PRED_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset501_preRT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"

python evaluation.py "${GT_DIR}" "${PRED_DIR}"
