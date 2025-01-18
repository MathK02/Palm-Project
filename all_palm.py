import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_and_display_average_palm(filename="ImagesPALM.mat"):
    # Load the PALM sequence
    data = loadmat(filename)
    palm_sequence = data['ImagesPALM']
    
    # Calculate the average image
    average_image = np.mean(palm_sequence, axis=2)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create two subplots: one for normal view, one with different colormap
    plt.subplot(121)
    plt.imshow(average_image, cmap='gray')
    plt.colorbar(label='Average Intensity')
    plt.title('Average of PALM Sequence\n(Grayscale)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    
    # Show the same data with a different colormap to highlight details
    plt.subplot(122)
    plt.imshow(average_image, cmap='hot')
    plt.colorbar(label='Average Intensity')
    plt.title('Average of PALM Sequence\n(Hot colormap)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    
    plt.tight_layout()
    plt.show()
    
    return average_image

# Run the function
average_image = load_and_display_average_palm()