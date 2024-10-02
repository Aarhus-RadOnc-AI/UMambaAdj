#!/bin/bash
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_train --c 501 3d_fullres_bs4 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerUmamba
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 501 3d_fullres_bs4 0 -p nnUNetPlansUmamba -tr nnUNetTrainerUMambaEnc


