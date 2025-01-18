import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
from scipy.optimize import minimize
from tqdm import tqdm  # For progress bar
import time  # Add this import at the top

def load_palm_data(filename):
    """Previous implementation remains the same"""
    try:
        data = loadmat(filename)
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def detect_local_maxima(image, threshold=0.3, neighborhood_size=3):
    """Previous implementation remains the same"""
    image_norm = (image - image.min()) / (image.max() - image.min())
    footprint = generate_binary_structure(2, 2)
    local_max = maximum_filter(image_norm, size=neighborhood_size) == image_norm
    background = (image_norm <= threshold)
    eroded_background = binary_erosion(background, structure=footprint, border_value=1)
    detected_maxima = local_max & ~eroded_background
    coordinates = np.where(detected_maxima)
    return coordinates

def gaussian_psf(x, y, x0, y0, amplitude=1.0, fwhm=2.0):
    """Previous implementation remains the same"""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def fit_psf(image_patch, initial_guess, fwhm=2.0):
    """
    Fit a 2D Gaussian PSF to an image patch using Maximum Likelihood estimation
    
    Parameters:
    -----------
    image_patch : 2D numpy array
        Small region of the image containing a single fluorophore
    initial_guess : tuple
        Initial guess for (x0, y0, amplitude)
    fwhm : float
        Full Width at Half Maximum of the PSF
    
    Returns:
    --------
    tuple
        Optimized (x0, y0, amplitude) parameters
    """
    y_grid, x_grid = np.indices(image_patch.shape)
    
    def negative_likelihood(params):
        x0, y0, amplitude = params
        model = gaussian_psf(x_grid, y_grid, x0, y0, amplitude, fwhm)
        # Assuming Poisson noise model
        log_likelihood = np.sum(image_patch * np.log(model + 1e-10) - model)
        return -log_likelihood
    
    # Bounds for parameters: x, y within patch size, amplitude > 0
    bounds = ((0, image_patch.shape[1]), 
              (0, image_patch.shape[0]), 
              (0, None))
    
    result = minimize(negative_likelihood, initial_guess, 
                     method='L-BFGS-B', bounds=bounds)
    
    return result.x

def refine_positions(image, rough_coords, patch_size=5):
    """
    Refine fluorophore positions using ML estimation
    
    Parameters:
    -----------
    image : 2D numpy array
        Full image
    rough_coords : tuple of arrays
        (y_coordinates, x_coordinates) from initial detection
    patch_size : int
        Size of the patch to use for fitting (should be odd)
    
    Returns:
    --------
    numpy array
        Refined positions as array of [y, x, amplitude] values
    """
    refined_positions = []
    half_patch = patch_size // 2
    
    for y, x in zip(*rough_coords):
        # Extract patch around the detected position
        y_start = max(0, y - half_patch)
        y_end = min(image.shape[0], y + half_patch + 1)
        x_start = max(0, x - half_patch)
        x_end = min(image.shape[1], x + half_patch + 1)
        
        patch = image[y_start:y_end, x_start:x_end]
        
        # Skip if patch is too small
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            continue
        
        # Initial guess: center of patch and max intensity
        initial_guess = (half_patch, half_patch, np.max(patch))
        
        try:
            # Fit PSF to patch
            refined = fit_psf(patch, initial_guess)
            
            # Convert local coordinates back to global image coordinates
            global_y = y_start + refined[1]
            global_x = x_start + refined[0]
            
            refined_positions.append([global_y, global_x, refined[2]])
        except:
            print(f"Failed to fit PSF at position ({y}, {x})")
            continue
    
    return np.array(refined_positions)

def display_detection_results(image, detected_coords, refined_positions=None, true_coords=None):
    """Updated to show both rough detection and refined positions"""
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.colorbar(label='Intensity')
    
    # Detection results
    plt.subplot(122)
    plt.imshow(image, cmap='gray')
    
    # Plot initial detected positions
    if detected_coords is not None:
        plt.plot(detected_coords[1], detected_coords[0], 'rx', 
                label='Initial Detection', markersize=8, alpha=0.5)
    
    # Plot refined positions
    if refined_positions is not None:
        plt.plot(refined_positions[:, 1], refined_positions[:, 0], 'y+',
                label='Refined Position', markersize=10)
    
    # Plot ground truth if available
    if true_coords is not None:
        x_true = true_coords['j_molecules'].flatten() - 0.5
        y_true = true_coords['i_molecules'].flatten() - 0.5
        plt.plot(x_true, y_true, 'go', fillstyle='none',
                label='Ground Truth', markersize=15)
    
    plt.title('Detection Results')
    plt.legend()
    plt.colorbar(label='Intensity')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load test data
    print("Loading data...")
    start_time = time.time()
    test_data = load_palm_data('ImageTest.mat')
    coordinates_data = load_palm_data('CoordinatesTest.mat')
    load_time = time.time() - start_time
    print(f"Data loading time: {load_time:.2f} seconds")
    
    if test_data is not None and coordinates_data is not None:
        test_image = test_data['ImageTest']
        
        # Initial detection
        print("\nPerforming initial detection...")
        start_time = time.time()
        detected_coords = detect_local_maxima(test_image, threshold=0.3)
        detection_time = time.time() - start_time
        print(f"Initial detection time: {detection_time:.2f} seconds")
        
        # Refine positions
        print("\nRefining positions...")
        start_time = time.time()
        refined_positions = refine_positions(test_image, detected_coords)
        refinement_time = time.time() - start_time
        print(f"Position refinement time: {refinement_time:.2f} seconds")
        
        # Print timing summary
        print("\nTiming Summary:")
        print(f"Data loading: {load_time:.2f} seconds")
        print(f"Initial detection: {detection_time:.2f} seconds")
        print(f"Position refinement: {refinement_time:.2f} seconds")
        print(f"Total processing time: {load_time + detection_time + refinement_time:.2f} seconds")
        
        # Print results
        print("\nResults Summary:")
        print(f"Number of initial detections: {len(detected_coords[0])}")
        print(f"Number of refined positions: {len(refined_positions)}")
        print(f"Number of true fluorophores: {len(coordinates_data['i_molecules'].flatten())}")
        
        if len(refined_positions) > 0:
            print("\nRefined positions (y, x, amplitude):")
            for pos in refined_positions:
                print(f"y: {pos[0]:.2f}, x: {pos[1]:.2f}, amplitude: {pos[2]:.2f}")
        
        # Display results
        print("\nDisplaying results...")
        display_detection_results(test_image, detected_coords, 
                                refined_positions, coordinates_data)

if __name__ == "__main__":
    main()