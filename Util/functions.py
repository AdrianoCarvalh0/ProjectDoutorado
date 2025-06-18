import os, json
import sys
from pathlib import Path
import numpy as np
#windows
sys.path.insert(0, r"C:\Users\adria\Documents\Doutorado\ProjetoDoutorado")
root_dir = Path(r"C:\Users\adria\Documents\Doutorado\ProjetoDoutorado")

#linux
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")
#root_dir = f"/home/adriano/projeto_mestrado/modules"

def read_directories(directory, img=None, exclude_json=None):
    # Get a list of filenames in the specified directory
    filenames = []
    for filename in os.listdir(directory):
        if img is not None:
            # If 'img' is provided, filter filenames containing it
            if img in filename:   
                filenames.append(filename)
        elif exclude_json is not None:
            filenames.append(filename.replace('.json',''))     
        else:
            filenames.append(filename)    
    return filenames

def write_array_to_file(array_list, filename):  
    # Convert array items to lists before writing to JSON file
    list_of_lists = [item.tolist() for item in array_list]
    json.dump(list_of_lists, open(filename, 'w'), indent=2)

def write_dict_to_file(dictionary, filename):
    array_list = dictionary["curve"]
    distance = dictionary["distance"]  
    # Convert array items to lists before writing to JSON file
    list_of_lists = [item.tolist() for item in array_list]  
    json.dump(list_of_lists, open(filename, 'w'), indent=2)

def add_gaussian_noise(image, mean=0, sigma=30):
    """
    Adds Gaussian noise to an image.

    Parameters:
    - image: input image (numpy array).
    - mean: mean of the Gaussian distribution (default: 0).
    - sigma: standard deviation of the Gaussian distribution (default: 25).

    Returns:
    - Image with Gaussian noise.
    """
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):   
    """
    Adds salt and pepper noise to an image.

    Parameters:
    - image: input image (numpy array).
    - salt_prob: probability of salt pixels (white).
    - pepper_prob: probability of pepper pixels (black).

    Returns:
    - Image with salt and pepper noise.
    """
    
    row, col = image.shape
    noisy = np.copy(image)

    # Adds salt noise
    salt_pixels = np.random.rand(row, col) < salt_prob
    noisy[salt_pixels] = 255

    # Adds pepper noise
    pepper_pixels = np.random.rand(row, col) < pepper_prob
    noisy[pepper_pixels] = 0

    return noisy.astype(np.uint8)