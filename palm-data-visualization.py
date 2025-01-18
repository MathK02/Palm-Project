import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

def load_palm_data(filename):
    """
    Load .mat file and return its contents
    """
    try:
        data = loadmat(filename)
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def display_test_image(image_data):
    """
    Display test image (70x100 double)
    """
    plt.figure(figsize=(10, 7))
    plt.imshow(image_data, cmap='gray', aspect='auto')
    plt.colorbar(label='Intensity')
    plt.title('Test Image')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.show()

def display_test_image_with_coordinates(image_data, coordinates):
    """
    Display test image with molecule coordinates overlay
    """
    plt.figure(figsize=(10, 7))
    
    # Display the image
    plt.imshow(image_data, cmap='gray', aspect='auto')
    plt.colorbar(label='Intensity')
    
    # Extract coordinates
    x_coords = coordinates['j_molecules'].flatten() -0.5
    y_coords = coordinates['i_molecules'].flatten() -0.5
    
    # Plot coordinates as scatter points
    plt.scatter(x_coords, y_coords, c='r', s=10, alpha=0.5, label='Molecule positions')
    plt.legend()
    
    plt.title('Test Image with Molecule Positions')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.show()

def display_palm_sequence(palm_data, num_samples=9):
    """
    Display a sample of images from the PALM sequence (70x100x999)
    """
    # Calculate number of rows and columns for the subplot grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle('Sample Images from PALM Sequence')
    
    # Select evenly spaced frames from the sequence
    indices = np.linspace(0, palm_data.shape[2]-1, num_samples, dtype=int)
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(palm_data[:, :, indices[i]], cmap='gray', aspect='auto')
            ax.set_title(f'Frame {indices[i]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def display_floue_image(image_floue):
    """
    Display the blurred (floue) reference image
    """
    plt.figure(figsize=(10, 7))
    plt.imshow(image_floue, cmap='gray', aspect='auto')
    plt.colorbar(label='Intensity')
    plt.title('Blurred Reference Image (Image Floue)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.show()

def main():
    # Load all data files
    test_data = load_palm_data('ImageTest.mat')
    coordinates_data = load_palm_data('CoordinatesTest.mat')
    palm_sequence_data = load_palm_data('ImagesPALM.mat')
    floue_data = load_palm_data('ImageFloue.mat')
    
    if test_data is not None:
        # Display test image
        display_test_image(test_data['ImageTest'])
        
        # Display test image with coordinates if available
        if coordinates_data is not None:
            display_test_image_with_coordinates(test_data['ImageTest'], coordinates_data)
    
    # Display PALM sequence samples
    if palm_sequence_data is not None:
        display_palm_sequence(palm_sequence_data['ImagesPALM'])
    
    # Display ImageFloue
    if floue_data is not None:
        display_floue_image(floue_data['ImageFloue'])
    
    # Print data shapes for verification
    print("\nData Shapes:")
    if test_data is not None:
        print(f"Test Image: {test_data['ImageTest'].shape}")
    if coordinates_data is not None:
        print(f"Coordinates i: {coordinates_data['i_molecules'].shape}")
        print(f"Coordinates j: {coordinates_data['j_molecules'].shape}")
    if palm_sequence_data is not None:
        print(f"PALM Sequence: {palm_sequence_data['ImagesPALM'].shape}")
    if floue_data is not None:
        print(f"Image Floue: {floue_data['ImageFloue'].shape}")

if __name__ == "__main__":
    main()