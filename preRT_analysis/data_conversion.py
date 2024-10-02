import os
import shutil
import numpy as np
import SimpleITK as sitk
import logging
import logging.handlers
import queue
from concurrent.futures import ThreadPoolExecutor
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndimage

import time

# Set up logging to file and stdout (thread-safe with queue)
log_queue = queue.Queue()
queue_handler = logging.handlers.QueueHandler(log_queue)
stream_handler = logging.StreamHandler()  # Console output
file_handler = logging.FileHandler("data_conversion.log")  # File output

# Define logging formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Queue listener to process logs from multiple threads
queue_listener = logging.handlers.QueueListener(log_queue, stream_handler, file_handler)

logging.basicConfig(level=logging.INFO, handlers=[queue_handler])

# Start the queue listener
queue_listener.start()



# Define paths
base_dir = '/data/jintao/nnUNet/HNTSMRG24_train/'
dataset_preRT = 'Dataset501_preRT'
dataset_midRT = 'Dataset502_midRT'
dataset_bbox = 'Dataset503_midRT_bbox'
dataset_geodist = 'Dataset504_midRT_geodist'

# Create dataset folders
os.makedirs(os.path.join(nnUNet_raw, dataset_preRT, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(nnUNet_raw, dataset_preRT, 'labelsTr'), exist_ok=True)
os.makedirs(os.path.join(nnUNet_raw, dataset_midRT, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(nnUNet_raw, dataset_midRT, 'labelsTr'), exist_ok=True)
os.makedirs(os.path.join(nnUNet_raw, dataset_bbox, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(nnUNet_raw, dataset_bbox, 'labelsTr'), exist_ok=True)
os.makedirs(os.path.join(nnUNet_raw, dataset_geodist, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(nnUNet_raw, dataset_geodist, 'labelsTr'), exist_ok=True)

# Function to generate bounding box masks for multiple instances of a given label
def create_bounding_box_mask(mask_data):
    bbox_mask = np.zeros_like(mask_data)

    for value in [1, 2]:  # Process each mask value separately
        label_mask = mask_data == value

        if np.sum(label_mask) ==0:
            continue
        
        connected_components = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(label_mask.astype(np.uint8))))

        for component_label in np.unique(connected_components):
            if component_label == 0:
                continue  # Skip background

            coords = np.argwhere(connected_components == component_label)
            if coords.size == 0:
                continue  # Skip if no voxels are found

            # Perturbation in x and y directions (first two dimensions)
            min_coords = coords.min(axis=0) - np.array([np.random.randint(4, 8), np.random.randint(4, 8), np.random.randint(4, 8)])   
            max_coords = coords.max(axis=0) + np.array([np.random.randint(4, 8),np.random.randint(4, 8), np.random.randint(4, 8)])  

            min_coords = np.maximum(min_coords, 0)  # Ensure coordinates are within bounds
            max_coords = np.minimum(max_coords, mask_data.shape)
            bbox_slice = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
            bbox_mask[bbox_slice] = value

    return bbox_mask

def normalize_image(image):
    """Normalize the image based on the full image's min and max values."""
    normalized_image = (image - image.min()) / (image.max() - image.min())
    return normalized_image

def shrink_mask_to_fraction(mask, fraction):
    """Shrink the mask proportionally by a given fraction using iterative erosion.
    If the volume erodes to zero, return a 2x2x2 voxel centered on the mass center of the original mask."""
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask)

    # Calculate the target volume (number of voxels) after shrinking
    original_volume = mask.sum()
    target_volume = int(original_volume * fraction)
    
    # Perform iterative erosion until the desired volume is reached
    current_volume = original_volume
    iteration = 0
    origina_mask = mask.copy()
    while current_volume > target_volume and current_volume > 0:
        # Erode slightly in each iteration
        mask = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
        current_volume = mask.sum()
        iteration += 1
        logging.info(f"Iteration {iteration}: current mask volume = {current_volume}")
    
    if current_volume == 0:
        logging.warning("Erosion resulted in an empty mask. Returning 2x2x2 voxel centered on mass center.")
        
        # Calculate the mass center of the original mask
        center_of_mass = ndimage.center_of_mass(origina_mask)
        center_of_mass = np.round(center_of_mass).astype(int)
        
        # Create a 2x2x2 voxel centered on the mass center
        small_mask = np.zeros_like(mask)
        x, y, z = center_of_mass
        x_slice = slice(max(x-1, 0), min(x+1+1, mask.shape[0]))
        y_slice = slice(max(y-1, 0), min(y+1+1, mask.shape[1]))
        z_slice = slice(max(z-1, 0), min(z+1+1, mask.shape[2]))
        
        small_mask[x_slice, y_slice, z_slice] = 1
        
        return small_mask
    
    return mask

def compute_gradient_magnitude(image, mask, bbox_slice):
    """Compute the normalized gradient magnitude inside the bounding box."""
    logging.info("Starting compute_gradient_magnitude...")

    # Normalize the image within the bounding box
    normalized_image_in_bbox = normalize_image(image[bbox_slice])
    logging.info(f"Normalized image in bbox min: {normalized_image_in_bbox.min()}, max: {normalized_image_in_bbox.max()}")

    # Calculate the gradient magnitude of the normalized image inside the bounding box
    gradient_magnitude_in_bbox = ndimage.gaussian_gradient_magnitude(normalized_image_in_bbox, sigma=1)
    logging.info(f"Gradient magnitude in bbox min: {gradient_magnitude_in_bbox.min()}, max: {gradient_magnitude_in_bbox.max()}")

    # Normalize the gradient magnitude map to the range [0, 1]
    min_value = gradient_magnitude_in_bbox.min()
    max_value = gradient_magnitude_in_bbox.max()
    
    # Ensure the denominator is not zero to avoid division by zero
    if max_value - min_value > 0:
        gradient_magnitude_in_bbox = (gradient_magnitude_in_bbox - min_value) / (max_value - min_value)
    else:
        gradient_magnitude_in_bbox = np.ones_like(gradient_magnitude_in_bbox)  # Set to 1, would convert to 0 later

    logging.info(f"Gradient magnitude in bbox after normalization min: {gradient_magnitude_in_bbox.min()}, max: {gradient_magnitude_in_bbox.max()}")

    # Apply scaling and clipping
    gradient_magnitude_in_bbox = gradient_magnitude_in_bbox * 2
    gradient_magnitude_in_bbox[gradient_magnitude_in_bbox > 1] = 1

    return gradient_magnitude_in_bbox

def check_bbox_coverage(midRT_mask, bbox_mask,  threshold=0.05):
    """
    Compares the volume of the tumor inside the bounding box (bbox_mask)
    with the total tumor volume in the midRT mask.

    Parameters:
    - midRT_mask: the actual midRT tumor mask
    - bbox_mask: the mask defined by the bounding box
    - threshold: percentage threshold for reporting missing volume
    """
    midRT_mask=midRT_mask.astype(int)
    bbox_mask=bbox_mask.astype(int)
    total_bbox = bbox_mask>0
    total_tumor_volume = np.sum(midRT_mask > 0)  # Total volume of the tumor in midRT
    tumor_in_bbox_volume = np.sum(np.logical_and(midRT_mask > 0, total_bbox))  # Tumor volume inside bbox
    
    # midRT_mask = sitk.GetImageFromArray(midRT_mask)
    # midRT_mask.CopyInformation(midRT_mask_img) 
    # bbox_mask = sitk.GetImageFromArray(bbox_mask)
    # bbox_mask.CopyInformation(midRT_mask_img) 
    
    # sitk.WriteImage(midRT_mask_img, 'mask.nii.gz')
    # sitk.WriteImage(bbox_mask_img, 'bbox.nii.gz')

    # If total tumor volume is zero, no tumor is present in midRT
    if total_tumor_volume == 0:
        return False, 0.0  # No tumor to check against

    # Calculate the missing volume rate
    missing_volume_rate = (total_tumor_volume - tumor_in_bbox_volume) / total_tumor_volume

    # Check if the missing volume exceeds the threshold
    if missing_volume_rate > threshold:
        logging.warning(f"Missing volume rate is {missing_volume_rate * 100:.2f}% (Threshold: {threshold * 100}%).")
        logging.warning(f"total_tumor_volume is {total_tumor_volume} tumor_in_bbox_volume is {tumor_in_bbox_volume}, bbox volume is {np.sum(total_bbox)}.")
        return True, missing_volume_rate
    else:
        return False, missing_volume_rate
    
def calculate_gradient_map_within_bbox(image, preRT_mask, midRT_mask, patient_id, threshold=0.05):
    """
    Calculate the gradient magnitude map for each component within the bounding box
    and check if any portion of the midRT tumor is missing from the bounding box.
    
    Parameters:
    - image: the T2 MRI image (midRT)
    - preRT_mask: pre-treatment mask (used to define the bounding box)
    - midRT_mask: mid-treatment mask (used for volume comparison)
    - patient_id: ID of the patient being processed
    - threshold: missing volume threshold for warning (default: 5%)
    """
    gradient_map = np.zeros_like(image)  # Initialize the gradient map with zeros for the background

    for value in [1, 2]:  # GTVp and GTVn tumor instances
        start_time = time.time()  # Start timing this block
        instance_mask = preRT_mask == value
        if midRT_mask is None:
            midRT_instance_mask = instance_mask
        else:
            midRT_instance_mask = midRT_mask == value

        logging.info(f"Processing Patient ID: {patient_id}, Tumor Type: {value}.")

        if np.sum(instance_mask) == 0:
            logging.info(f"Patient ID: {patient_id}, Tumor Type: {value} has no preRT tumor.")
            continue
        
        connected_components = sitk.GetArrayFromImage(
            sitk.ConnectedComponent(sitk.GetImageFromArray(instance_mask.astype(np.uint8))))
        
        for component_label in np.unique(connected_components):
            if component_label == 0:
                continue  # Skip background

            component_coords = np.argwhere(connected_components == component_label)
            if component_coords.size == 0:
                logging.info(f"Patient ID: {patient_id}, Tumor Type: {value}, Component {component_label} - No voxels found.")
                continue

            # Define the bounding box from the preRT mask (with perturbations)
            min_coords = component_coords.min(axis=0) - np.array([np.random.randint(4, 8), np.random.randint(4, 8), np.random.randint(4, 8)])
            max_coords = component_coords.max(axis=0) + np.array([np.random.randint(4, 8), np.random.randint(4, 8), np.random.randint(4, 8)])

            # Ensure the bounding box coordinates are within the bounds of the image
            min_coords = np.maximum(min_coords, 0)
            max_coords = np.minimum(max_coords, image.shape)

            logging.info(f"Component {component_label} bounding box: min={min_coords}, max={max_coords}")

            # Extract the bounding box slice
            bbox_slice = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
            extracted_preRT_mask = instance_mask[bbox_slice]
            extracted_midRT_mask = midRT_instance_mask[bbox_slice]  # Extract the midRT tumor within the bounding box

            if extracted_preRT_mask.sum() == 0:
                logging.info(f"Patient ID: {patient_id}, Tumor Type: {value} - Bounding box contains no preRT tumor.")
                continue

            logging.info(f"Bbox slice: {bbox_slice}")

            # Compute the gradient magnitude map inside the bounding box
            gradient_start_time = time.time()  # Start timing the gradient computation
            gradient_map_in_bbox = compute_gradient_magnitude(image, extracted_preRT_mask, bbox_slice)
            gradient_end_time = time.time()  # End timing the gradient computation

            # Log the runtime for gradient computation if it took longer than 0 seconds
            gradient_runtime = gradient_end_time - gradient_start_time
            if gradient_runtime > 0:
                logging.info(f"Gradient calculation for component {component_label} took {gradient_runtime:.2f} seconds.")
                
            # Integrate the gradient magnitude map into the corresponding region of the full gradient map
            gradient_map[bbox_slice] = gradient_map_in_bbox
            
        # Step to check for missing tumor volume in the midRT mask
        bbox_mask = np.zeros_like(gradient_map)
        bbox_mask = gradient_map > 0
        
        check_start_time = time.time()  # Start timing the coverage check
        is_missing, missing_rate = check_bbox_coverage(midRT_instance_mask, bbox_mask, threshold)
        check_end_time = time.time()  # End timing the coverage check

        # Log the runtime for the coverage check if it took longer than 0 seconds
        check_runtime = check_end_time - check_start_time
        if check_runtime > 0:
            logging.info(f"Coverage check took {check_runtime:.2f} seconds.")

        if is_missing:
            logging.warning(f"Patient {patient_id}, Tumor Type {value}: Bounding box misses {missing_rate * 100:.2f}% of the tumor in midRT.")
        
        end_time = time.time()  # End timing the whole block for each tumor type
        total_runtime = end_time - start_time

        # Log the runtime for the whole block if it took longer than 0 seconds
        if total_runtime > 0:
            logging.info(f"Processing Patient ID: {patient_id}, Tumor Type: {value} took {total_runtime:.2f} seconds.")
    
    return gradient_map

# def calculate_gradient_map_within_bbox(image, preRT_mask, midRT_mask, patient_id, threshold=0.05):
#     """
#     Calculate the gradient magnitude map for each component within the bounding box
#     and check if any portion of the midRT tumor is missing from the bounding box.
    
#     Parameters:
#     - image: the T2 MRI image (midRT)
#     - preRT_mask: pre-treatment mask (used to define the bounding box)
#     - midRT_mask: mid-treatment mask (used for volume comparison)
#     - patient_id: ID of the patient being processed
#     - threshold: missing volume threshold for warning (default: 5%)
#     """
#     gradient_map = np.zeros_like(image)  # Initialize the gradient map with zeros for the background

#     for value in [1, 2]:  # GTVp and GTVn tumor instances
#         instance_mask = preRT_mask == value
#         if midRT_mask is None:
#             midRT_instance_mask = instance_mask
#         else:
#             midRT_instance_mask = midRT_mask == value

#         logging.info(f"Processing Patient ID: {patient_id}, Tumor Type: {value}.")

#         if np.sum(instance_mask) == 0:
#             logging.info(f"Patient ID: {patient_id}, Tumor Type: {value} has no preRT tumor.")
#             continue
        
#         connected_components = sitk.GetArrayFromImage(
#             sitk.ConnectedComponent(sitk.GetImageFromArray(instance_mask.astype(np.uint8))))

#         for component_label in np.unique(connected_components):
#             if component_label == 0:
#                 continue  # Skip background

#             component_coords = np.argwhere(connected_components == component_label)
#             if component_coords.size == 0:
#                 logging.info(f"Patient ID: {patient_id}, Tumor Type: {value}, Component {component_label} - No voxels found.")
#                 continue

#             # Define the bounding box from the preRT mask (with perturbations)
#             min_coords = component_coords.min(axis=0) - np.array([np.random.randint(2, 6), np.random.randint(2, 6), np.random.randint(2, 6)])
#             max_coords = component_coords.max(axis=0) + np.array([np.random.randint(2, 6), np.random.randint(2, 6), np.random.randint(2, 6)])

#             # Ensure the bounding box coordinates are within the bounds of the image
#             min_coords = np.maximum(min_coords, 0)
#             max_coords = np.minimum(max_coords, image.shape)

#             logging.info(f"Component {component_label} bounding box: min={min_coords}, max={max_coords}")

#             # Extract the bounding box slice
#             bbox_slice = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
#             extracted_preRT_mask = instance_mask[bbox_slice]
#             extracted_midRT_mask = midRT_instance_mask[bbox_slice]  # Extract the midRT tumor within the bounding box

#             if extracted_preRT_mask.sum() == 0:
#                 logging.info(f"Patient ID: {patient_id}, Tumor Type: {value} - Bounding box contains no preRT tumor.")
#                 continue

#             logging.info(f"Bbox slice: {bbox_slice}")

#             # Compute the gradient magnitude map inside the bounding box
#             gradient_map_in_bbox = compute_gradient_magnitude(image, extracted_preRT_mask, bbox_slice)

#             # Integrate the gradient magnitude map into the corresponding region of the full gradient map
#             gradient_map[bbox_slice] = gradient_map_in_bbox
            
#         # Step to check for missing tumor volume in the midRT mask
#         bbox_mask = np.zeros_like(gradient_map)
#         bbox_mask= gradient_map>0
        
#         is_missing, missing_rate = check_bbox_coverage(midRT_instance_mask, bbox_mask, threshold)
#         if is_missing:
#             logging.warning(f"Patient {patient_id}, Tumor Type {value}: Bounding box misses {missing_rate * 100:.2f}% of the tumor in midRT.")

#     return gradient_map

# Function to copy and rename files, and create bounding box dataset
def process_patient(patient_id):
    preRT_dir = os.path.join(base_dir, patient_id, 'preRT')
    midRT_dir = os.path.join(base_dir, patient_id, 'midRT')
    logging.info(f"Start processing for patient {patient_id}...")

    # Copying files to Dataset501_preRT
    shutil.copy(os.path.join(preRT_dir, f'{patient_id}_preRT_T2.nii.gz'),
                os.path.join(nnUNet_raw, dataset_preRT, 'imagesTr', f'{patient_id}_0000.nii.gz'))
    shutil.copy(os.path.join(preRT_dir, f'{patient_id}_preRT_mask.nii.gz'),
                os.path.join(nnUNet_raw, dataset_preRT, 'labelsTr', f'{patient_id}.nii.gz'))

    # Copying files to Dataset502_midRT
    shutil.copy(os.path.join(midRT_dir, f'{patient_id}_midRT_T2.nii.gz'),
                os.path.join(nnUNet_raw, dataset_midRT, 'imagesTr', f'{patient_id}_0000.nii.gz'))
    shutil.copy(os.path.join(midRT_dir, f'{patient_id}_preRT_T2_registered.nii.gz'),
                os.path.join(nnUNet_raw, dataset_midRT, 'imagesTr', f'{patient_id}_0001.nii.gz'))
    shutil.copy(os.path.join(midRT_dir, f'{patient_id}_preRT_mask_registered.nii.gz'),
                os.path.join(nnUNet_raw, dataset_midRT, 'imagesTr', f'{patient_id}_0002.nii.gz'))
    shutil.copy(os.path.join(midRT_dir, f'{patient_id}_midRT_mask.nii.gz'),
                os.path.join(nnUNet_raw, dataset_midRT, 'labelsTr', f'{patient_id}.nii.gz'))

    # Copy to Dataset503_midRT_bbox
    shutil.copy(os.path.join(nnUNet_raw, dataset_midRT, 'imagesTr', f'{patient_id}_0000.nii.gz'),
                os.path.join(nnUNet_raw, dataset_bbox, 'imagesTr', f'{patient_id}_0000.nii.gz'))
    shutil.copy(os.path.join(nnUNet_raw, dataset_midRT, 'labelsTr', f'{patient_id}.nii.gz'),
                os.path.join(nnUNet_raw, dataset_bbox, 'labelsTr', f'{patient_id}.nii.gz'))

    # Create bounding box mask for Dataset503_midRT_bbox
    preRT_mask_registered_img = sitk.ReadImage(os.path.join(midRT_dir, f'{patient_id}_preRT_mask_registered.nii.gz'))
    preRT_mask_registered_data = sitk.GetArrayFromImage(preRT_mask_registered_img)

    bbox_mask_data = create_bounding_box_mask(preRT_mask_registered_data)
    bbox_mask_img = sitk.GetImageFromArray(bbox_mask_data)
    bbox_mask_img.CopyInformation(preRT_mask_registered_img)  # Copy the metadata from the original image
    sitk.WriteImage(bbox_mask_img, os.path.join(nnUNet_raw, dataset_bbox, 'imagesTr', f'{patient_id}_0001.nii.gz'))
    

    
    # PreRT data to Dataset504 with "pre" suffix
    shutil.copy(os.path.join(preRT_dir, f'{patient_id}_preRT_T2.nii.gz'),
                os.path.join(nnUNet_raw, dataset_geodist, 'imagesTr', f'{patient_id}pre_0000.nii.gz'))
    shutil.copy(os.path.join(preRT_dir, f'{patient_id}_preRT_mask.nii.gz'),
                os.path.join(nnUNet_raw, dataset_geodist, 'labelsTr', f'{patient_id}pre.nii.gz'))
    
    # Create gradient map based on preRT T2 and preRT mask
    preRT_mask_img = sitk.ReadImage(os.path.join(preRT_dir, f'{patient_id}_preRT_mask.nii.gz'))
    preRT_mask_data = sitk.GetArrayFromImage(preRT_mask_img)
    preRT_T2_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(preRT_dir, f'{patient_id}_preRT_T2.nii.gz')))
    gradient_map_pre = calculate_gradient_map_within_bbox(preRT_T2_image, preRT_mask_data, None, f"{patient_id}pre")
    if gradient_map_pre is not None:
        gradient_map_img_pre = sitk.GetImageFromArray(gradient_map_pre.astype(np.float32))
        gradient_map_img_pre.CopyInformation(preRT_mask_img)
        sitk.WriteImage(gradient_map_img_pre, os.path.join(nnUNet_raw, dataset_geodist, 'imagesTr', f'{patient_id}pre_0001.nii.gz'))

    # Copy to Dataset504_midRT_geodist
    shutil.copy(os.path.join(nnUNet_raw, dataset_midRT, 'imagesTr', f'{patient_id}_0000.nii.gz'),
                os.path.join(nnUNet_raw, dataset_geodist, 'imagesTr', f'{patient_id}_0000.nii.gz'))
    shutil.copy(os.path.join(nnUNet_raw, dataset_midRT, 'labelsTr', f'{patient_id}.nii.gz'),
                os.path.join(nnUNet_raw, dataset_geodist, 'labelsTr', f'{patient_id}.nii.gz'))

    #Create geodesic distance map for Dataset504_midRT_geodist
    logging.info(f"Calculating geodesic distance map for patient {patient_id}...")

    preRT_mask_registered_img = sitk.ReadImage(os.path.join(midRT_dir, f'{patient_id}_preRT_mask_registered.nii.gz'))
    preRT_mask_registered_data = sitk.GetArrayFromImage(preRT_mask_registered_img)
    midRT_mask_img = sitk.ReadImage(os.path.join(midRT_dir, f'{patient_id}_midRT_mask.nii.gz'))
    midRT_mask_data = sitk.GetArrayFromImage(midRT_mask_img)
    t2_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(midRT_dir, f'{patient_id}_midRT_T2.nii.gz')))
    gradient_map = calculate_gradient_map_within_bbox(t2_image, preRT_mask_registered_data, midRT_mask_data, patient_id)
    #logging.info(f"Geodesic distance map {patient_id} shape: {geodesic_distance_map.shape}.")

    if gradient_map is not None:
        gradient_map_img = sitk.GetImageFromArray(gradient_map.astype(np.float32))
        gradient_map_img.CopyInformation(preRT_mask_registered_img)
        sitk.WriteImage(gradient_map_img, os.path.join(nnUNet_raw, dataset_geodist, 'imagesTr', f'{patient_id}_0001.nii.gz'))
        #logging.info(f"Geodesic distance map written for patient {patient_id}.")
    else:
        logging.error(f"Geodesic distance map calculation failed for patient {patient_id}.")


# Process each patient using multithreading
def main():
    patients = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # # Using ThreadPoolExecutor to run the process_patient function concurrently
    with ThreadPoolExecutor(max_workers=64) as executor:
        executor.map(process_patient, patients)
    # for patient in patients:
    #     #patient ='177'
    #     process_patient(patient)

    # Generate dataset JSON files after processing
    generate_dataset_json(os.path.join(nnUNet_raw, dataset_preRT),
                          {0: "T2"}, 
                          labels={"background": 0, "GTVp": 1, "GTVn": 2}, 
                          num_training_cases=len(patients), 
                          file_ending='.nii.gz', 
                          dataset_name=dataset_preRT, 
                          description="Dataset 501 - preRT")

    generate_dataset_json(os.path.join(nnUNet_raw, dataset_midRT),
                          {0: "T2", 1: "preRT_T2_registered", 3: "preRT_mask_registered"}, 
                          labels={"background": 0, "GTVp": 1, "GTVn": 2}, 
                          num_training_cases=len(patients), 
                          file_ending='.nii.gz', 
                          dataset_name=dataset_midRT, 
                          description="Dataset 502 - midRT")

    generate_dataset_json(os.path.join(nnUNet_raw, dataset_bbox),
                          {0: "T2", 1: "preRT_bbox_mask"}, 
                          labels={"background": 0, "GTVp": 1, "GTVn": 2}, 
                          num_training_cases=len(patients), 
                          file_ending='.nii.gz', 
                          dataset_name=dataset_bbox, 
                          description="Dataset 503 - midRT with Bounding Box")
    
    generate_dataset_json(os.path.join(nnUNet_raw, dataset_geodist),
                          {0: "T2", 1: "gradient"}, 
                          labels={"background": 0, "GTVp": 1, "GTVn": 2}, 
                          num_training_cases=len(patients), 
                          file_ending='.nii.gz', 
                          dataset_name=dataset_geodist, 
                          description="Dataset 504 - midRT with Gradient Map")
    

if __name__ == "__main__":
    main()
    queue_listener.stop()