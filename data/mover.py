from PIL import Image
import os

MOVE = 50000
new_size = (128, 128)

source = 'img_align_celeba/'
destination = 'celeba_50000/'

img_paths = os.listdir(source)
for _ in range(MOVE):
    source_img_path = source + img_paths[_]
    with Image.open(source_img_path) as img:
        resized = img.resize(new_size)
        resized.save(destination + img_paths[_])