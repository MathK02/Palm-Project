import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
from scipy.optimize import minimize
from tqdm import tqdm
import time

def load_palm_data(filename):
    """Load .mat file data"""
    try:
        data = loadmat(filename)
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def detect_local_maxima(image, threshold=0.15, neighborhood_size=5):
    """
    Efficiently detect local maxima in an image above a threshold.
    
    Parameters:
    -----------
    image : ndarray
        2D input image
    threshold : float
        Minimum intensity value for maxima detection (0 to 1)
    neighborhood_size : int
        Size of the neighborhood for local maxima detection
        
    Returns:
    --------
    coordinates : tuple of arrays
        (y_coordinates, x_coordinates) of detected maxima
    """
    # Normalize image to 0-1 range
    image_norm = (image - image.min()) / (image.max() - image.min())
    
    # Create mask for values above threshold
    above_threshold = image_norm > threshold
    
    # Apply maximum filter to find local maxima
    footprint = generate_binary_structure(2, 2)  # 8-connectivity
    local_max = maximum_filter(image_norm, size=neighborhood_size) == image_norm
    
    # Combine threshold and local maxima conditions
    maxima = local_max & above_threshold
    
    # Get coordinates of maxima
    coordinates = np.where(maxima)
    
    return coordinates


def visualize_all_maxima(image_shape, all_coordinates, save_path=None):
    """
    Visualize all detected local maxima from all frames on a single plot.
    
    Parameters:
    -----------
    image_shape : tuple
        Shape of the original image (height, width)
    all_coordinates : list of tuples
        List containing (y_coordinates, x_coordinates) tuples from each frame
    save_path : str, optional
        Path to save the visualization. If None, display only.
    """
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Create empty image to show density
    density_map = np.zeros(image_shape)
    
    # Accumulate all points
    total_points = 0
    for coords in all_coordinates:
        y_coords, x_coords = coords
        for y, x in zip(y_coords, x_coords):
            density_map[y, x] += 1
        total_points += len(y_coords)
    
    # Display density map
    plt.imshow(density_map, cmap='viridis')
    plt.colorbar(label='Number of detections')
    
    plt.title(f'All Local Maxima Across Frames\nTotal detections: {total_points}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def gaussian_psf(x, y, x0, y0, amplitude=1.0, fwhm=2.0):
    """Generate Gaussian PSF"""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def fit_psf(image_patch, initial_guess, fwhm=2.0):
    """Fit a 2D Gaussian PSF to an image patch"""
    y_grid, x_grid = np.indices(image_patch.shape)
    
    def negative_likelihood(params):
        x0, y0, amplitude = params
        model = gaussian_psf(x_grid, y_grid, x0, y0, amplitude, fwhm)
        log_likelihood = np.sum(image_patch * np.log(model + 1e-10) - model)
        return -log_likelihood
    
    bounds = ((0, image_patch.shape[1]), 
              (0, image_patch.shape[0]), 
              (0, None))
    
    result = minimize(negative_likelihood, initial_guess, 
                     method='L-BFGS-B', bounds=bounds)
    
    return result.x

def refine_positions(image, rough_coords, patch_size=6):
    """Refine fluorophore positions using ML estimation"""
    refined_positions = []
    half_patch = patch_size // 2
    
    for y, x in zip(*rough_coords):
        y_start = max(0, y - half_patch)
        y_end = min(image.shape[0], y + half_patch + 1)
        x_start = max(0, x - half_patch)
        x_end = min(image.shape[1], x + half_patch + 1)
        
        patch = image[y_start:y_end, x_start:x_end]
        
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            continue
        
        initial_guess = (half_patch, half_patch, np.max(patch))
        
        try:
            refined = fit_psf(patch, initial_guess)
            global_y = y_start + 2*refined[1]
            global_x = x_start + 2*refined[0]
            refined_positions.append([global_y, global_x, refined[2]])
        except:
            print(f"Failed to fit PSF at position ({y}, {x})")
            continue
    
    return np.array(refined_positions) if refined_positions else None

def reconstruct_super_resolution_image(positions, pixel_size=0.01, image_shape=None, sigma_factor=6):
    """
    Reconstruct a super-resolution image using local Gaussian patches.
    Efficient version that only computes Gaussian values in a small radius around each position.
    """
    if image_shape is None:
        max_y = np.ceil(np.max(positions[:, 0]))
        max_x = np.ceil(np.max(positions[:, 1]))
        image_shape = (int(max_y), int(max_x))
    
    # Calculate super-resolution grid dimensions
    sr_shape = (int(image_shape[0] / pixel_size), int(image_shape[1] / pixel_size))
    super_res_image = np.zeros(sr_shape)
    
    if len(positions) == 0:
        return super_res_image
    
    # Convert positions to high-res grid coordinates
    y_coords = (positions[:, 0] / pixel_size).astype(int)
    x_coords = (positions[:, 1] / pixel_size).astype(int)
    amplitudes = positions[:, 2]
    
    # Create small Gaussian patch once
    radius = int(3 * sigma_factor)  # 3 sigma radius
    y_patch, x_patch = np.mgrid[-radius:radius+1, -radius:radius+1]
    gaussian_patch = np.exp(-(x_patch**2 + y_patch**2) / (2 * (sigma_factor)**2))
    
    # Add patch around each position
    for y, x, amplitude in zip(y_coords, x_coords, amplitudes):
        # Calculate patch boundaries
        y_min = max(0, y - radius)
        y_max = min(sr_shape[0], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(sr_shape[1], x + radius + 1)
        
        # Calculate patch indices
        patch_y_min = max(0, radius - y)
        patch_y_max = patch_y_min + (y_max - y_min)
        patch_x_min = max(0, radius - x)
        patch_x_max = patch_x_min + (x_max - x_min)
        
        # Add Gaussian patch
        super_res_image[y_min:y_max, x_min:x_max] += amplitude * \
            gaussian_patch[patch_y_min:patch_y_max, patch_x_min:patch_x_max]
    
    return super_res_image

def process_palm_sequence(filename="ImagesPALM.mat", blurred_reference="BlurredImage.png"):
    """Process a sequence of PALM images and reconstruct the super-resolution image."""
    # Load image sequence
    print("Loading PALM sequence...")
    data = load_palm_data(filename)
    if data is None:
        return None
    
    # Get image sequence
    image_sequence = data['ImagesPALM']
    n_frames = image_sequence.shape[2]
    print(f"Processing {n_frames} frames...")
    
    # Store all detected coordinates before refinement
    all_detected_coords = []
    
    # Process each frame
    all_positions = []
    for i in tqdm(range(n_frames), desc="Processing frames"):
        frame = image_sequence[:, :, i]
        
        # Detect fluorophores
        detected_coords = detect_local_maxima(frame, threshold=0.3)
        all_detected_coords.append(detected_coords)  # Store coordinates
        
        # Refine positions
        if len(detected_coords[0]) > 0:
            refined_positions = refine_positions(frame, detected_coords)
            if refined_positions is not None and len(refined_positions) > 0:
                all_positions.extend(refined_positions)
                # print(refined_positions)
    
    # Visualize all detected maxima before refinement
    visualize_all_maxima(image_sequence.shape[:2], all_detected_coords)
    
    all_positions = np.array(all_positions)
    print(f"Total fluorophores detected: {len(all_positions)}")
    
    # Reconstruct super-resolution image
    print("Reconstructing super-resolution image...")
    super_res_image = reconstruct_super_resolution_image(
        all_positions,  # 10x super-resolution
        image_shape=image_sequence.shape[:2]
    )
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Display the super-resolution image
    im = ax.imshow(super_res_image, cmap='hot')
    ax.set_title('Reconstructed Super-Resolution Image')

    # Add a colorbar to the plot
    plt.colorbar(im, ax=ax)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
    return super_res_image

if __name__ == "__main__":
    super_res_image = process_palm_sequence()