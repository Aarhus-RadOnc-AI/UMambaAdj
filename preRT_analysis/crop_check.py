import os
import SimpleITK as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings

# Set folder paths
imageTr_folder = "/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset501_preRT/imagesTr"
labelTr_folder = "/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset501_preRT/labelsTr"
imageTr_crop_folder = "/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset501_preRT/imagesTr_crop"
labelTr_crop_folder = "/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset501_preRT/labelsTr_crop"

# Create output directories if they don't exist
os.makedirs(imageTr_crop_folder, exist_ok=True)
os.makedirs(labelTr_crop_folder, exist_ok=True)

def crop_image_and_label(patient_id, margin=50):
    try:
        # File paths
        image_path = os.path.join(imageTr_folder, f"{patient_id}_0000.nii.gz")
        label_path = os.path.join(labelTr_folder, f"{patient_id}.nii.gz")
        image_out_path = os.path.join(imageTr_crop_folder, f"{patient_id}_0000.nii.gz")
        label_out_path = os.path.join(labelTr_crop_folder, f"{patient_id}.nii.gz")
    
        
        # Read images
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        

        
        # Convert images to numpy arrays for manipulation
        image_array = sitk.GetArrayFromImage(image)
        label_array = sitk.GetArrayFromImage(label)
        
        # Get image shape (z, y, x) -> axial, coronal, sagittal
        img_shape = image_array.shape
        
        if 550 <img_shape[-1] < 600:
            margin = 50
        elif img_shape[-1] >= 600:
            margin = 110
        else:
            margin = 30
            
            # Crop the images (keep all slices along the axial axis)
        cropped_image_array = image_array[:, margin:-margin, margin:-margin]
        cropped_label_array = label_array[:, margin:-margin, margin:-margin]
        
        # Check if any label volume was removed
        original_label_volume = np.sum(label_array > 0)
        cropped_label_volume = np.sum(cropped_label_array > 0)
        
        if cropped_label_volume < original_label_volume:
            warnings.warn(f"@@Warning: Cropping may have removed some label volume for patient {patient_id}.")
        
        # Adjust origin after cropping (since margin is removed)
        original_origin = image.GetOrigin()
        original_spacing = image.GetSpacing()

        new_origin = (
            original_origin[0] + margin * original_spacing[0],
            original_origin[1] + margin * original_spacing[1],
            original_origin[2]
        )
        
        # Convert cropped numpy arrays back to SimpleITK images
        cropped_image = sitk.GetImageFromArray(cropped_image_array)
        cropped_label = sitk.GetImageFromArray(cropped_label_array)
        
        # Update metadata (spacing, direction, new origin)
        cropped_image.SetSpacing(image.GetSpacing())
        cropped_image.SetDirection(image.GetDirection())
        cropped_image.SetOrigin(new_origin)
        
        cropped_label.SetSpacing(label.GetSpacing())
        cropped_label.SetDirection(label.GetDirection())
        cropped_label.SetOrigin(new_origin)
        #print(f"{patient_id}, before shape {img_shape} after shape {cropped_image_array.shape}")
        
        # Write the cropped images
        sitk.WriteImage(cropped_image, image_out_path)
        sitk.WriteImage(cropped_label, label_out_path)
        
        print(f"Successfully processed patient {patient_id}.")
        
        # Return necessary information for restoring the original image after prediction
        return {
            'patient_id': patient_id,
            'crop_margin': margin,
            'original_origin': original_origin,
            'original_spacing': original_spacing,
            'new_origin': new_origin,
        }
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        return None
    
def main():
    # Get list of patient IDs based on label file names
    patient_ids = [f.split(".nii.gz")[0] for f in os.listdir(labelTr_folder) if f.endswith(".nii.gz")]

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        executor.map(crop_image_and_label, patient_ids)

if __name__ == "__main__":
    main()