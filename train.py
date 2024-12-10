import numpy as np
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import time
from constants import *

from Discriminator import Discriminator
from Generator import Generator

def train():
    real_images = np.load("data/characters_32.npy")
    print(np.shape(real_images))
    noise = np.random.normal(0, 0.02, real_images.shape)
    real_images_noisy = real_images + noise
    real_images = np.array_split(real_images_noisy, np.ceil(len(real_images_noisy) / BATCH_SIZE))

    batch_num = len(real_images)

    generator = Generator(BATCH_SIZE)
    discriminator = Discriminator(BATCH_SIZE)

    # load_epoch = 0
    # load_batch = 0
    # with open(f"model_saves/discriminator_state_epoch_{load_epoch}_{load_batch}.pkl", "rb") as d_state_file:
    #     discriminator.load_state_dict(pickle.load(d_state_file))


    # with open(f"model_saves/generator_state_epoch_{load_epoch}_{load_batch}.pkl", "rb") as g_state_file:
    #     generator.load_state_dict(pickle.load(g_state_file))

    start_epoch = 0
    start_batch = 0
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
    min_lr = 0.00005
    max_lr = 0.002

    # Label Smoothing
    REAL_LABEL_MIN = 0.8
    REAL_LABEL_MAX = 1.0
    FAKE_LABEL_MIN = 0.0
    FAKE_LABEL_MAX = 0.2

    # with open(f"model_saves/hyperparams_state_epoch_{load_epoch}_{load_batch}.pkl", "rb") as hyperparams_file:
    #     hyperparams = pickle.load(hyperparams_file)
    #     D_learning_rate = hyperparams['D_learning_rate']
    #     G_learning_rate = hyperparams['G_learning_rate']
    #     train_d_times = hyperparams['train_d_times']
    #     train_g_times = hyperparams['train_g_times']
    #     start_epoch = hyperparams['epoch_number']
    #     start_batch = hyperparams['batch_index']
    #     print(f"Loaded: {hyperparams}")

    def adjustLearningRates():
        nonlocal D_learning_rate, G_learning_rate, train_d_times, train_g_times

        prev_D_learning_rate = D_learning_rate
        prev_G_learning_rate = G_learning_rate

        d_real_avg = sum(d_real_losses) / len(d_real_losses) if len(d_real_losses) > 0 else 0
        d_fake_avg = sum(d_fake_losses) / len(d_fake_losses) if len(d_fake_losses) > 0 else 0
        g_avg = sum(g_losses) / len(g_losses) if len(g_losses) > 0 else 0

        d_loss_avg = d_real_avg + d_fake_avg

        if abs(g_avg - d_loss_avg) > threshold:
            if g_avg > d_loss_avg:
                G_learning_rate *= 1.05
                D_learning_rate *= 0.95
            elif d_loss_avg > g_avg:
                G_learning_rate *= 0.95
                D_learning_rate *= 1.05

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


    for epoch_number in range(start_epoch, epochs + 1):
        print(f"Epoch:\t{epoch_number}.")
        current_time = time.time()
        for batch_index, batch in enumerate(real_images):
            if epoch_number == start_epoch and batch_index < start_batch:
                continue

            batch_start_time = time.time()
            d_time = 0
            g_time = 0
            # Train discriminator
            for _ in range(train_d_times):
                start_time = time.time()
                
                real_labels = np.random.uniform(REAL_LABEL_MIN, REAL_LABEL_MAX, size=(BATCH_SIZE, 1))
                
                d_output_real = discriminator.forward(batch)
                d_output_real = np.clip(d_output_real, 1e-7, 1 - 1e-7)
                
                # Compute loss with label smoothing
                d_output_real_loss = - (real_labels * np.log(d_output_real) + (1 - real_labels) * np.log(1 - d_output_real))
                
                # Compute gradient
                d_output_real_loss_gradient = - (real_labels / d_output_real) + ((1 - real_labels) / (1 - d_output_real))
                
                # Backward pass
                discriminator.backward(d_output_real_loss_gradient)
                
                fake_labels = np.random.uniform(FAKE_LABEL_MIN, FAKE_LABEL_MAX, size=(BATCH_SIZE, 1))
                
                # Generate fake images
                latent_vector = np.random.normal(0, 1, (BATCH_SIZE, 1, 1, 256))

                fake_image = generator.forward(latent_vector)
                
                d_output_fake = discriminator.forward(fake_image)
                d_output_fake = np.clip(d_output_fake, 1e-7, 1 - 1e-7)
                
                # Compute loss with label smoothing
                d_output_fake_loss = - (fake_labels * np.log(d_output_fake) + (1 - fake_labels) * np.log(1 - d_output_fake))
                
                # Compute gradient
                d_output_fake_loss_gradient = - (fake_labels / d_output_fake) + ((1 - fake_labels) / (1 - d_output_fake))
                
                # Backward pass
                discriminator.backward(d_output_fake_loss_gradient)

                discriminator.applyDeltas(D_learning_rate)
                discriminator.resetDeltas()
                d_time += time.time() - start_time

            # Train generator
            for _ in range(train_g_times):
                start_time = time.time()
                
                # Generate fake images
                latent_vector = np.random.normal(0, 1, (BATCH_SIZE, 1, 1, 256))

                fake_image = generator.forward(latent_vector)
                
                # Forward pass through discriminator
                d_output_fake = discriminator.forward(fake_image)
                d_output_fake = np.clip(d_output_fake, 1e-7, 1 - 1e-7)
                
                # Define generator target labels with smoothing
                gen_labels = np.random.uniform(REAL_LABEL_MIN, REAL_LABEL_MAX, size=(BATCH_SIZE, 1))
                
                # Compute generator loss
                g_loss = - (gen_labels * np.log(d_output_fake) + (1 - gen_labels) * np.log(1 - d_output_fake))
                
                # Compute gradient
                g_loss_gradient = - (gen_labels / d_output_fake) + ((1 - gen_labels) / (1 - d_output_fake))
                
                # Backward pass through discriminator
                d_gradient_max = discriminator.backward(g_loss_gradient)
                
                # Backward pass through generator
                generator.backward(d_gradient_max)
                
                # Update generator parameters
                generator.applyDeltas(G_learning_rate)
                generator.resetDeltas()
                g_time += time.time() - start_time

            # Update sliding window losses
            d_real_losses.append(np.mean(d_output_real_loss))
            d_fake_losses.append(np.mean(d_output_fake_loss))
            g_losses.append(np.mean(g_loss))

            if (batch_index + 1) % adjust_rates_every == 0 and len(d_real_losses) == window_size:
                adjustLearningRates()

            batch_time = time.time() - batch_start_time
            print(
                f"{batch_index}/{batch_num}\tD Real Loss:\t{np.mean(d_output_real_loss):.5f} | "
                f"D Fake Loss:\t{np.mean(d_output_fake_loss):.5f}\tG Loss:\t{np.mean(g_loss):.5f} | "
                f"Batch Time: {batch_time:.3f}s (D (trained {train_d_times} times): {d_time:.3f}s, G (trained {train_g_times} times): {g_time:.3f}s)"
            )

            # Save generated images
            fake_image_processed = ((fake_image + 1) * 127.5).clip(0, 255).astype(np.uint8)
            fake_image_to_save = Image.fromarray(fake_image_processed[-1])
            fake_image_to_save.save(f"generated_images/fake_{batch_index}.png")
            
            if batch_index % 50 == 0:
                with open(f"model_saves/discriminator_state_epoch_{epoch_number}_{batch_index}.pkl", "wb") as d_state_file:
                    pickle.dump(discriminator.state_dict(), d_state_file)

                with open(f"model_saves/generator_state_epoch_{epoch_number}_{batch_index}.pkl", "wb") as g_state_file:
                    pickle.dump(generator.state_dict(), g_state_file)

                # Save hyperparameters
                hyperparams = {
                    'D_learning_rate': D_learning_rate,
                    'G_learning_rate': G_learning_rate,
                    'train_d_times': train_d_times,
                    'train_g_times': train_g_times,
                    'epoch_number': epoch_number,
                    'batch_index': batch_index
                }
                with open(f"model_saves/hyperparams_state_epoch_{epoch_number}_{batch_index}.pkl", "wb") as hyperparams_file:
                    pickle.dump(hyperparams, hyperparams_file)
                print(f"Checkpoint saved for epoch {epoch_number}, batch {batch_index}.")
        print(f"Epoch Time: {time.time() - current_time:.3f}s")
        print("Reshuffling data...")
        np.random.shuffle(real_images)
        print("Reshuffling done.")

train()
