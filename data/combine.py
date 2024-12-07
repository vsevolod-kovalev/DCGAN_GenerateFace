import numpy as np
import os
def combine_to_single_file(folder, output_file):
    all_images = []
    for npy_file in os.listdir(folder):
        if npy_file.endswith('.npy'):
            img_array = np.load(os.path.join(folder, npy_file))
            all_images.append(img_array)
    all_images = np.stack(all_images)
    np.save(output_file, all_images)
    print(f"Dataset saved to {output_file}, shape: {all_images.shape}")

combine_to_single_file('preprocessed_celeba_50000/', "dataset.npy")