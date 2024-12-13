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
    real_images = np.load("data/32_characters.npy")
    for _ in range(100, 110):
        plt.imshow(real_images[_])
        plt.show()

    print(np.shape(real_images))
    noise = np.random.normal(0, 0.05, real_images.shape)
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
    epochs = 400
    D_learning_rate = 0.0002
    G_learning_rate = 0.0002
    train_d_times = 5
    train_g_times = 1

    # with open(f"model_saves/hyperparams_state_epoch_{load_epoch}_{load_batch}.pkl", "rb") as hyperparams_file:
    #     hyperparams = pickle.load(hyperparams_file)
    #     # D_learning_rate = hyperparams['D_learning_rate']
    #     # G_learning_rate = hyperparams['G_learning_rate']
    #     # train_d_times = hyperparams['train_d_times']
    #     # train_g_times = hyperparams['train_g_times']
    #     start_epoch = hyperparams['epoch_number']
    #     start_batch = hyperparams['batch_index']
    #     print(f"Loaded: {hyperparams}")

    for epoch_number in range(start_epoch, epochs + 1):
        print(f"Epoch:\t{epoch_number}.")
        np.random.shuffle(real_images)
        current_time = time.time()
        for batch_index, batch in enumerate(real_images):

            d_reals = []
            d_fakes = []
            g_losses = []

            # skip batches after loading: 
            if epoch_number == start_epoch and batch_index < start_batch:
                continue

            batch_start_time = time.time()
            d_time = 0
            g_time = 0
            # Train discriminator
            for _ in range(train_d_times):
                start_time = time.time()
                # D for real images:
                d_real_score = discriminator.forward(batch)
                d_real_loss = -1.0 * d_real_score
                # maximize real loss:
                discriminator.backward(np.full((BATCH_SIZE), -1.0)) # deltas of weights and biases are accumulated in D's layers after calling backward()
                # D for fake images:
                latent_vector = np.random.normal(0, 1, (BATCH_SIZE, 1, 1, 100))
                fake_batch = generator.forward(latent_vector)
                d_fake_score = discriminator.forward(fake_batch)
                d_fake_loss = 1.0 * d_fake_score
                # minimize fake loss:
                discriminator.backward(np.full((BATCH_SIZE), 1.0)) # deltas of weights and biases are accumulated in D's layers after calling backward()
                # appply accumulated deltas stored in D:
                discriminator.applyDeltas()
                discriminator.resetDeltas()

                d_reals.append(d_real_loss)
                d_fakes.append(d_fake_loss)

                # clip D's weights after uptading them:
                for layer in discriminator.layers:
                    np.clip(layer.W, -0.01, 0.01, out=layer.W)

                d_time += time.time() - start_time
            # Train generator
            fake_image_to_save = None
            for _ in range(train_g_times):
                start_time = time.time()

                latent_vector = np.random.normal(0, 1, (BATCH_SIZE, 1, 1, 100))
                fake_batch = generator.forward(latent_vector)
                g_score = discriminator.forward(fake_batch)
                g_loss = -1.0 * g_score
                 # maximize G loss:
                d_gradient = discriminator.backward(np.full((BATCH_SIZE), -1.0)) # deltas of weights and biases are accumulated in D's layers after calling backward()
                generator.backward(d_gradient) # deltas of weights and biases are accumulated in G's layers after calling backward()
                # appply accumulated deltas stored in G and omit deltas stored in D:
                generator.applyDeltas()
                generator.resetDeltas()
                g_losses.append(g_loss)
                discriminator.resetDeltas()
                g_time += time.time() - start_time
                # save the last image from the batch
                if _ == train_g_times - 1:
                    fake_image_to_save = fake_batch[-1]
                
    
            batch_time = time.time() - batch_start_time
            print(
                f"{batch_index}/{batch_num}\tD Real Loss:\t{np.mean(d_reals):.7f} | "
                f"D Fake Loss:\t{np.mean(d_fakes):.7f}\tG Loss:\t{np.mean(g_losses):.7f} | "
                f"Batch Time: {batch_time:.3f}s (D (trained {train_d_times} times): {d_time:.3f}s, G (trained {train_g_times} times): {g_time:.3f}s)"
            )
            # save last generated image from each batch
            fake_image = ((fake_image_to_save + 1) * 127.5).clip(0, 255).astype(np.uint8)
            fake_image_converted = Image.fromarray(fake_image)
            fake_image_converted.save(f"generated_images/fake_{batch_index}.png")
            
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
        

train()