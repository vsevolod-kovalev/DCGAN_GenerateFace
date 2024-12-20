import numpy as np
from PIL import Image
import pickle

from Generator import Generator
from constants import *

BATCH_SIZE = 16

generator = Generator(BATCH_SIZE)

with open(f"model_saves/trained_generator.pkl", "rb") as g_state_file:
    generator.load_state_dict(pickle.load(g_state_file))

def save_image_grid(images, image_width, num_cols, output_path):
    canvas = Image.new("RGBA", (image_width * num_cols, image_width * num_cols))
    x = 0
    y = 0
    canvas_size = image_width * num_cols
    for image_idx in range(len(images)):
        image =  Image.fromarray(((images[image_idx] + 1) * 127.5).clip(0, 255).astype(np.uint8))
        canvas.paste(image, (x, y))
        x += image_width
        if x >= canvas_size:
            x = 0
            y += image_width
    print(f"Fake grid saved to {output_path}")
    canvas.save(output_path)

grids_to_generate = 10
for _ in range(grids_to_generate):
    latent_vector = np.random.normal(0, 1, (BATCH_SIZE, 1, 1, LATENT_SIZE))
    fake_batch = generator.forward(latent_vector)
    save_image_grid(fake_batch, len(fake_batch[0]), 4, f"generated_images/fake_grid_{_}.png")
