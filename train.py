import numpy as np
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import time
from constants import *

from Discriminator import Discriminator
from Generator import Generator

import jax.numpy as jnp
from jax import random


def train():
    real_images = np.load("data/dataset_32.npy")
    noise = np.random.normal(0, 0.02, real_images.shape)
    real_images_noisy = real_images + noise
    real_images = np.array_split(real_images_noisy, np.ceil(len(real_images_noisy) / BATCH_SIZE))
    for _ in range(20, 25):
        plt.imshow(real_images[_][0])
        plt.show()

    batch_num = len(real_images)

    generator = Generator(BATCH_SIZE)
    discriminator = Discriminator(BATCH_SIZE)

    epochs = 200
    D_learning_rate = 0.002
    G_learning_rate = 0.002
    adjust_rates_every = 5
    # Default training ratio: 1:1
    train_d_times = 1
    train_g_times = 1

    window_size = 10
    d_real_losses = deque(maxlen=window_size)
    d_fake_losses = deque(maxlen=window_size)
    g_losses = deque(maxlen=window_size)

    threshold = 0.2
    decay_factor = 0.99 
    min_lr = 1e-5
    max_lr = 0.001

    def adjustLearningRates():
        nonlocal D_learning_rate, G_learning_rate, train_d_times, train_g_times

        prev_D_learning_rate = D_learning_rate
        prev_G_learning_rate = G_learning_rate

        d_real_avg = sum(d_real_losses) / len(d_real_losses) if len(d_real_losses) > 0 else 0
        d_fake_avg = sum(d_fake_losses) / len(d_fake_losses) if len(d_fake_losses) > 0 else 0
        g_avg = sum(g_losses) / len(g_losses) if len(g_losses) > 0 else 0

        d_loss_avg = d_real_avg + d_fake_avg

        # Adjust learning rates based on the difference in losses
        if abs(g_avg - d_loss_avg) > threshold:
            if g_avg > d_loss_avg:
                G_learning_rate *= 1.05
                D_learning_rate *= 0.95
            elif d_loss_avg > g_avg:
                G_learning_rate *= 0.95
                D_learning_rate *= 1.05

        # Apply decay
        G_learning_rate *= decay_factor
        D_learning_rate *= decay_factor

        G_learning_rate = max(min_lr, min(G_learning_rate, max_lr))
        D_learning_rate = max(min_lr, min(D_learning_rate, max_lr))

        print(
            f"Adjusting Learning Rates:\n"
            f"Previous G Learning Rate: {prev_G_learning_rate:.6f}, Current G Learning Rate: {G_learning_rate:.6f}\n"
            f"Previous D Learning Rate: {prev_D_learning_rate:.6f}, Current D Learning Rate: {D_learning_rate:.6f}"
        )
        prev_train_d_times = train_d_times
        prev_train_g_times = train_g_times

        if d_loss_avg > g_avg * 1.5:
            train_d_times += 1
            train_g_times = max(1, train_g_times - 1)
        elif g_avg > d_loss_avg * 1.5:
            train_g_times += 1
            train_d_times = max(1, train_d_times - 1)
        if train_g_times == train_d_times:
            train_g_times, train_d_times = 1, 1

        print(
            f"Adjusting Training Ratios:\n"
            f"Previous D Training Times: {prev_train_d_times}, Current D Training Times: {train_d_times}\n"
            f"Previous G Training Times: {prev_train_g_times}, Current G Training Times: {train_g_times}"
        )
    for epoch_number in range(1, epochs + 1):
        print(f"Epoch:\t{epoch_number}.")
        current_time = time.time()
        for batch_index, batch in enumerate(real_images):
            batch_start_time = time.time()
            d_time = 0
            g_time = 0
            # Train discriminator
            for _ in range(train_d_times):
                start_time = time.time()
                d_output_real = discriminator.forward(batch)
                d_output_real = np.clip(d_output_real, 1e-7, 1 - 1e-7)
                d_output_real_loss = -np.log(d_output_real)
                d_output_real_loss_gradient = -1.0 / d_output_real
                discriminator.backward(d_output_real_loss_gradient)

                latent_vector = np.random.uniform(*LATENT_RANGE, (BATCH_SIZE, 1, 1, 100))
                fake_image = generator.forward(latent_vector)
                d_output_fake = discriminator.forward(fake_image)
                d_output_fake = np.clip(d_output_fake, 1e-7, 1 - 1e-7)
                d_output_fake_loss = -np.log(1.0 - d_output_fake)
                d_output_fake_loss_gradient = 1.0 / (1.0 - d_output_fake)
                discriminator.backward(d_output_fake_loss_gradient)

                discriminator.applyDeltas(D_learning_rate)
                discriminator.resetDeltas()
                d_time += time.time() - start_time

            # Train generator
            for _ in range(train_g_times):
                start_time = time.time()
                latent_vector = np.random.uniform(*LATENT_RANGE, (BATCH_SIZE, 1, 1, 100))
                fake_image = generator.forward(latent_vector)
                d_output_fake = discriminator.forward(fake_image)
                d_output_fake = np.clip(d_output_fake, 1e-7, 1 - 1e-7)
                d_output_fake_loss_max = -np.log(d_output_fake)
                d_output_fake_loss_max_gradient = -1.0 / d_output_fake
                d_gradient_max = discriminator.backward(d_output_fake_loss_max_gradient)
                generator.backward(d_gradient_max)
                generator.applyDeltas(G_learning_rate)
                generator.resetDeltas()
                g_time += time.time() - start_time

            # Update sliding window losses
            d_real_losses.append(np.mean(d_output_real_loss))
            d_fake_losses.append(np.mean(d_output_fake_loss))
            g_losses.append(np.mean(d_output_fake_loss_max))

            if (batch_index + 1) % adjust_rates_every == 0 and len(d_real_losses) == window_size:
                adjustLearningRates()

            batch_time = time.time() - batch_start_time
            print(
                f"{batch_index}/{batch_num}\tD Real Loss:\t{np.mean(d_output_real_loss):.5f} | "
                f"D Fake Loss:\t{np.mean(d_output_fake_loss):.5f}\tG Loss:\t{np.mean(d_output_fake_loss_max):.5f} | "
                f"Batch Time: {batch_time:.3f}s (D (trained {train_d_times} times): {d_time:.3f}s, G (trained {train_g_times} times): {g_time:.3f}s)"
            )

            fake_image = ((fake_image + 1) * 127.5).clip(0, 255).astype(np.uint8)
            fake_image_to_save = Image.fromarray(fake_image[-1])
            fake_image_to_save.save(f"generated_images/fake_{batch_index}.png")
            if batch_index % 50 == 0:
                with open(f"model_saves/generator_epoch_{epoch_number}_{batch_index}.pkl", "wb") as gen_file:
                    pickle.dump(generator, gen_file)
                with open(f"model_saves/discriminator_epoch_{epoch_number}_{batch_index}.pkl", "wb") as disc_file:
                    pickle.dump(discriminator, disc_file)

        print(f"Epoch Time: {time.time() - current_time:.3f}s")
        print("Reshuffling data...")
        np.random.shuffle(real_images)
        print("Reshuffling done.")

train()