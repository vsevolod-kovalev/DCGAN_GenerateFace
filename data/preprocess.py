import os
from PIL import Image
import numpy as np

def preprocess_images(input_folder, output_folder, image_size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size)
            
                img_array = np.asarray(img, dtype=np.float32)
                img_array = (img_array / 127.5) - 1
                npy_path = os.path.join(output_folder, filename.split('.')[0] + '.npy')
                np.save(npy_path, img_array)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

input_folder = "celeba_50000/"
output_folder = "preprocessed_celeba_50000/"
preprocess_images(input_folder, output_folder)
