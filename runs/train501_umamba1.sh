#!/bin/bash
#CUDA_VISIBLE_DEVICES=3 nnUNetv2_train --c 501 3d_fullres_bs4 3 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerUmamba
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train --c 501 3d_fullres_bs4 4 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerUmamba #350
