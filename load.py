import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from constants import *
import pickle

from Discriminator import Discriminator
from Generator import Generator

def train():
    real_images = np.load("data/preprocessed_augmented_characters_16.npy")
    print("Dataset shape:", real_images.shape)
    generator, discriminator = None, None
    with open("model_saves/generator_epoch_3_21500.pkl", 'rb') as f:
        generator = pickle.load(f)
    with open("model_saves/discriminator_epoch_3_21500.pkl", 'rb') as f:
        discriminator = pickle.load(f)

    epochs = 200
    D_learning_rate = 0.0001
    G_learning_rate = 0.0004


    i = 21501
    for epoch_number in range(3, epochs+1):
        print(f"Epoch:\t{epoch_number}.")
        # Stochaistic Gradient Descent:
        for real_image in real_images:
            # discriminator + real
            d_output_real = discriminator.forward(real_image)
            d_output_real = np.clip(d_output_real, 1e-7, 1 - 1e-7)
            d_output_real_loss  = -np.log(d_output_real)
            d_output_real_loss_gradient = -1.0 / d_output_real
            discriminator.backward(d_output_real_loss_gradient)
            # discriminator + fake
            latent_vector = np.random.uniform(*LATENT_RANGE, (1, 1, 100))
            fake_image = generator.forward(latent_vector)
            d_output_fake = discriminator.forward(fake_image)
            d_output_fake = np.clip(d_output_fake, 1e-7, 1 - 1e-7) 
            d_output_fake_loss = -np.log(1.0 - d_output_fake)
            d_output_fake_loss_gradient = 1.0 / (1.0 - d_output_fake)
            discriminator.backward(d_output_fake_loss_gradient)
            # apply accumulates changes + reset
            discriminator.applyDeltas(D_learning_rate)
            discriminator.resetDeltas()
            # train generator
            for _ in range(2):
                # train discriminator:generator 1:2 times
                latent_vector = np.random.uniform(*LATENT_RANGE, (1, 1, 100))
                fake_image = generator.forward(latent_vector)
                d_output_fake = discriminator.forward(fake_image)
                d_output_fake = np.clip(d_output_fake, 1e-7, 1 - 1e-7)
                d_output_fake_loss_max = -np.log(d_output_fake)
                d_output_fake_loss_max_gradient = -1.0 / d_output_fake
                d_gradient_max = discriminator.backward(d_output_fake_loss_max_gradient)
                generator.backward(d_gradient_max)
                generator.applyDeltas(G_learning_rate)
                generator.resetDeltas()
                discriminator.resetDeltas()
            print(f"{i}\tD Real Loss:\t{d_output_real_loss[0]:.5f} | D Fake Loss:\t{d_output_fake_loss[0]:.5f}\tG Loss:\t{d_output_fake_loss_max[0]:.5f}")
            if i % 200 == 0:
                print(np.min(fake_image), np.max(fake_image))
                fake_image = ((fake_image + 1) * 127.5).clip(0, 255).astype(np.uint8)
                fake_image_to_save = Image.fromarray(fake_image)
                fake_image_to_save.save(f"generated_images/fake_{i}.png")
            if i % 500 == 0:
                with open(f"model_saves/generator_epoch_{epoch_number}_{i}.pkl", "wb") as gen_file:
                    pickle.dump(generator, gen_file)
                with open(f"model_saves/discriminator_epoch_{epoch_number}_{i}.pkl", "wb") as disc_file:
                    pickle.dump(discriminator, disc_file)
            i += 1
        print("Reshuffling data...")
        np.random.shuffle(real_images)
        print("Reshuffling done.")
train()