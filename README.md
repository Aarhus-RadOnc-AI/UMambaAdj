# UMambaAdj: Advancing GTV Segmentation for Head and Neck Cancer in MRI-Guided RT with UMamba and nnU-Net ResEnc Planner
This repository contains the code and model for UMambaAdj, a hybrid network combining UMamba and nnU-Net Residual Encoder (ResEnc) designed for T2-weighted MRI head and neck tumor segmentation. UMambaAdj leverages the long-range dependency capabilities of the Mamba block and the feature extraction strength of the residual encoder to improve segmentation performance for both GTVp and GTVn. The model was tested on the HNTS-MRG 2024 dataset, achieving enhanced accuracy in boundary delineation and volumetric overlap metrics.

## UMamba Adjustment (UMambaAdj): Enhanced UMamba for Head and Neck Tumor Segmentation
Folder ```nnUNet``` presents a customized version of nnUNet, optimized for head and neck tumor segmentation (HNTS). We enhance the original UMamba architecture by introducing significant optimizations to improve computational efficiency and segmentation accuracy.

### Key Modifications
- **Removal of the Mamba Layer and Residual Blocks**: We optimize UMamba by removing the Mamba layer in the first block and the residual blocks in the decoder, significantly enhancing computational efficiency while preserving the model's ability to capture long-range dependencies.
- **Integration with nnU-Net ResEnc**: By combining UMamba’s long-range dependency modeling with nnU-Net ResEnc’s enhanced residual encoding, we achieve improved accuracy in Gross Tumor Volume (GTV) delineation, especially in the complex anatomical structures of head and neck cancer.

