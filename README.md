## Try It Out!

Click the button below to open the interactive demo in Google Colab. You'll be able to run the model, generate pixel heroes, and view the results in your browser:

[![Run in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vsevolod-kovalev/DCGAN_Pixel_Heroes/blob/main/notebook_demo.ipynb)

# DCGAN: Deep Convolutional Generative Adversarial Network

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** using raw NumPy and native Python libraries. The framework mimics **TensorFlow’s Keras functionality**, allowing customizable Generator and Discriminator architectures with layers like `Dense`, `Conv2D`, and `Transposed Conv2D`, and others.
  <div align="center">
    <p align="center">
      <img width="800" src="https://github.com/user-attachments/assets/cc7af1c3-1707-4c2a-a14e-69eb777d22fc" alt="16x16x32">
    </p>
    <em>Progression of scratch-built DCGAN: from random noise to 64x64 images of game characters.</em><br>
  </div>

### Key Features

- **Custom Architectures**  
  - Build Generator and Discriminator architectures using a Sequential-like syntax.
- **Flexible Layers**  
  - Support for custom activation functions, batch normalization, and frozen layers.
- **Training Enhancements**  
  - Binary Cross-Entropy (BCE) loss and dynamic learning rate scheduling.
- **Progressive Image Generation**  
  - Compatibility with progressive resolution training (16x16 → 32x32 → 64x64) for improved stability.
- **No High-Level Libraries**  
  - Implemented without TensorFlow, PyTorch, or other ML libraries.

## Task

To train a discriminator and generator to generate colorful, detailed, and diverse 64x64 game characters without resorting to any machine learning libraries.

## Dataset

The Pixel Characters Dataset from <a href="https://www.kaggle.com/datasets/volodymyrpivoshenko/pixel-characters-dataset">Kaggle</a> was chosen for its detailed and diverse heroes. It provides **3,648** 64x64 images of pixel characters in various outfits, accessories, armor, tools, weapons, etc. 

**800** images were filtered based on labels (centered position, white background) and augmented using Pillow with transformations such as hue adjustments, flips, noise, and masks, extending the dataset to **8,000** samples. These samples were preprocessed to match the **tanh** activation function range *(-1, 1)* for the generator and converted into a single *.npy* file.

## Architecture
<p align="center">
  <img src="https://github.com/user-attachments/assets/7cd0d2d2-6270-4a8f-bf36-04bc6874824a" alt="16x16x32">
</p>

<p align="center">
  <em>Illustration of a Deep Convolutional Generative Adversarial Network to generate colored 32x32 images.</em><br>
  <em>Source: <a href="https://www.researchgate.net/figure/The-architecture-of-the-generator-and-the-discriminator-in-the-DCGAN-model_fig2_332445799">ResearchGate</a></em>
</p>


Generating 64x64 images requires the generator (G) and discriminator (D) architectures to be deep enough to capture fine details (eyes, horns, clothes, helmets) while also having enough filters to encapsulate the dataset's diversity. However, they must not be overly deep or complex to avoid overfitting on the relatively small dataset or leading to excessively long training times.

### Generator Architecture

To achieve this, the generator architecture includes one fully connected layer to process the 100-dimensional latent vector, reshaped to match the input shape for the reverse convolution layers. Three transposed convolution layers are then added to expand the noise vector into a 64x64x3 tensor, representing a colored fake image:

```python
[
    Dense(batch_size=batch_size, input_shape=(1, 1, 100), num_neurons=8 * 8 * 256, activation="lrelu", batch_norm=True),
    TConv2D(batch_size=batch_size, input_shape=(8, 8, 256), num_filters=128, kernel_size=4, stride=2, padding=1, activation="lrelu", batch_norm=True),
    TConv2D(batch_size=batch_size, input_shape=(16, 16, 128), num_filters=64, kernel_size=4, stride=2, padding=1, activation="lrelu", batch_norm=True),
    TConv2D(batch_size=batch_size, input_shape=(32, 32, 64), num_filters=3, kernel_size=4, stride=2, padding=1, activation="tanh")
]
```

### Discriminator Architecture

The discriminator processes the generated 64x64x3 images through two convolutional layers, interleaved with dropout layers to aid generalization by randomly "turning off" features. The final dense layer contains a single neuron:

For the discriminator: uses a sigmoid function to output the probability of the image being real (0 to 1).  
For the critic: unbounded, with no activation function, enabling more diverse feedback.
```python
[  
    Conv2D(batch_size=batch_size, input_shape=(64, 64, 3), num_filters=64, kernel_size=4, stride=2, padding=1, activation="lrelu"),  
    Dropout(0.3),  
    Conv2D(batch_size=batch_size, input_shape=(32, 32, 64), num_filters=128, kernel_size=4, stride=2, padding=1, activation="lrelu"),  
    Dropout(0.3),  
    Dense(batch_size=batch_size, input_shape=(16, 16, 128), num_neurons=1, activation="sigmoid")  
]
```

This architecture enables capturing distinguishable features in 64x64 images within a reasonable time, relying solely on NumPy.

## Training

<p align="center">
  <img height="600" src="https://github.com/user-attachments/assets/2e01ebd9-9d23-40fa-ad85-54e39084b2b2" alt="Training feedback">
</p>

<p align="center">
  <em>Illustration of the training process to produce colored 64x64 images.</em><br>
  <em>Source: <a href="https://www.researchgate.net/figure/DCGAN-Deep-Convolutional-Generative-Adversarial-Network-generator-used-for-LSUN_fig1_340884113">ResearchGate</a></em>
</p>


Training a DCGAN is a game with two players: the Discriminator and the Generator. The Discriminator is fed a sample of a real, genuine image labeled with 1, and a sample of fake data produced by the Generator labeled with 0. The Discriminator’s output reflects how "real" the provided image is perceived.

One of the most popular formulas used to calculate the loss is Binary Cross-Entropy (BCE):  
<p align="center">
  <img width="523" alt="image" src="https://github.com/user-attachments/assets/e9f07bd6-2c8d-46d5-a8ba-4a53af334058" />
</p>

### Real Image Loss (y = 1)

When fed a real image (y = 1), the formula simplifies to:  
<p align="center">
  <img width="167" alt="image" src="https://github.com/user-attachments/assets/65f29552-f382-4fad-b3e6-057584ad01d4" />
</p>

For example, if the Discriminator is provided with a real image labeled as 1 and its sigmoid output D(x) = 0.9, the loss is approximately: 
<p align="center">
  <img width="195" alt="image" src="https://github.com/user-attachments/assets/9ef39b92-f075-48d6-8399-2efc0bcbe9d7" />
</p>

### Fake Image Loss (y = 0)

Alternatively, for a fake image (y = 0), the formula simplifies to:  
<p align="center">
  <img width="199" alt="image" src="https://github.com/user-attachments/assets/d678c621-dfa5-42ca-a48f-c41079b19b2b" />
</p>

In this case, the loss increases significantly as the Discriminator "flips" the prediction by subtracting the output from 1.

### Training the Generator

Training the Generator involves creating a noise vector, which is passed through transposed convolutional layers to extend it to the desired image size. The Generator’s output layer produces a fake image, which is then evaluated by the Discriminator alongside a real image.

The Generator’s objective is to "fool" the Discriminator, minimizing its loss:  
<p align="center">
  <img width="284" alt="image" src="https://github.com/user-attachments/assets/b2643562-8a9b-4b17-98c2-5c9c4cd52522" />
</p>

Where G(z) is the fake image generated by the Generator from noise z.

### Gradient Updates

Once forward propagation is complete, the weights of the Discriminator and Generator are updated using mini-batch gradient descent:  
<p align="center">
  <img width="218" alt="image" src="https://github.com/user-attachments/assets/10648c7e-dba8-453f-84c9-096657fd7859" />
</p>

Biases are similarly updated:  
<p align="center">
  <img width="193" alt="image" src="https://github.com/user-attachments/assets/40204ce4-625a-4360-9a82-bc4224618387" />
</p>

The gradient then flows back to the Generator, updating the weights of its layers.

Due to the min-max nature of this game, GANs typically do not converge. Instead, the results are visually inspected, and training is stopped once satisfactory outputs are achieved.

## Challenges

### 1. Debugging Convolutional and Transposed Convolutional Layers

Debugging convolutional and reverse convolutional layers was the biggest challenge. While forward propagation is relatively straightforward, correctly calculating and passing all derivatives through the Discriminator and back to the Generator requires meticulous and time-consuming debugging. Additionally, the absence of high-level libraries meant that all convolution operations had to be:

- **Implemented in nested Python loops**, or  
- **Carefully vectorized using NumPy’s einsum** to achieve a performance boost (~4x). For instance, with a batch size of 64, processing time per batch decreased from ~2.5 seconds to ~0.67 seconds.  

The smallest calculation mistake necessitated retraining the networks from scratch and redebugging.

### 2. Lack of GPU Support

Since the project relies on NumPy and native Python libraries (e.g., os, random, time), which inherently do not support GPU usage, computational challenges arose. Drop-in replacement libraries like CuPy were not used to stay within the project’s library requirements.

**Solution**:  
Training was conducted on a cloud machine *(c2d-standard-56, 56 vCPU, 28 cores, 224 GB memory)*. The project was saved as a Docker image and deployed on Google Compute Engine (GCE), where it was controlled via SSH for training.

### 3. Unstable GAN Training and Learning Rate Sensitivity

GAN training is naturally unstable and highly sensitive to changes in learning rates.

- If the Generator is trained too often relative to the Discriminator or if their learning rates are unbalanced, the Generator can exploit Discriminator weaknesses, leading to mode collapse (generating identical samples).  
<div align="center">
  <table>
    <tr>
      <td align="center" style="padding: 20px;">
        <strong>Before (Epoch 3, 32x32 resolution):</strong><br>
        <img src="https://github.com/user-attachments/assets/9d57d1fb-249d-4544-a2b5-43adc7d3129c" width="300">
      </td>
      <td align="center" style="padding: 20px;">
        <strong>After (Epoch 4, mode collapse):</strong><br>
        <img src="https://github.com/user-attachments/assets/946bf80b-03b4-4a26-a24f-edeaf50e030e" width="350">
      </td>
    </tr>
  </table>
</div>

- Conversely, if the Discriminator becomes too powerful, it ceases to provide useful gradients to the Generator, leading to vanishing gradients and minimal improvement in Generator performance.

**Solution**:  
A dynamic learning rate scheduler with decay was implemented. It tracked the Discriminator’s real loss, fake loss, and the Generator’s loss to adjust learning rates dynamically (by 1–5%) and modify the training ratio of Discriminator to Generator from the default 1:1 as needed.

```python
 def adjustLearningRates():
        ...
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

        ...

        if d_loss_avg > g_avg * 1.5:
            train_d_times += 1
            train_g_times = max(1, train_g_times - 1)
        elif g_avg > d_loss_avg * 1.5:
            train_g_times += 1
            train_d_times = max(1, train_d_times - 1)
        if train_g_times == train_d_times:
            train_g_times, train_d_times = 1, 1
        ...
```

Additionally, **label smoothing** was applied:  
- Real labels were smoothed to a range of 0.8–0.9 instead of 1.0.  
- Fake labels were adjusted to a range of 0.0–0.2 if the Discriminator’s fake loss was too low.  

```python
REAL_LABEL_MIN = 0.8
REAL_LABEL_MAX = 1.0
FAKE_LABEL_MIN = 0.0
FAKE_LABEL_MAX = 0.2
```

### 4. Slow Training Speeds

Despite optimizations in vectorizing backward and forward passes for TConv2D and Conv2D layers, training speeds were significantly slower **(~19x for backward and ~17x for forward methods on an M1 MacBook Pro CPU)** compared to TensorFlow on the same architecture.

**Identifying Optimal Parameters**  
Determining optimal model parameters was time-consuming due to the slower training process. Small learning rates showed minimal changes in generated images during initial epochs, while large learning rates caused exploding gradients.

**Example (16x16, unbounded gradient problem):**

| Iteration | D Real Loss | D Fake Loss | G Loss     |
|-----------|-------------|-------------|------------|
| 22727     | 1.37706     | 0.17706     | 1.50141    |
| 22728     | 0.41753     | 0.18962     | 3.77718    |
| 22729     | 1.24531     | 0.23126     | 7.45698    |
| 22730     | 4.79243     | 0.25025     | 15.67032   |
| 22731     | 1.25465     | 0.30939     | 31.27190   |

**Solutions Implemented:**
1. **Weight Initialization:**
   - Replaced `np.random` weight initialization with **Xavier Initialization** and **He Initialization**, which improved gradient flow and stabilized training.

2. **Batch Normalization:**
   - Added **batch normalization** as a toggleable feature for Convolutional and Dense layers to normalize pre-activation values. This mitigated exploding or vanishing gradients and improved convergence.

3. **Loss Function Changes:**
   - Switched from **Binary Cross-Entropy (BCE)** to **Wasserstein GAN (WGAN) loss** with weight clipping, which prevents vanishing gradients and improves convergence.

   **Weight Clipping:** Ensures weights remain within a predefined range (e.g., *[-0.02, 0.02]*).  

4. **Dynamic Learning Rates:**
   - Implemented a scheduler to dynamically adjust learning rates based on the training progress, which helped optimize convergence speed and stability.

### 5. Higher Resolution (64x64)

64x64 images are **4** times larger than 32x32 images and **16** times larger than 16x16 images. Greater output resolution requires:
- More filters and depth in the architecture to capture finer details (e.g., hairstyles, armor).
- Longer training times due to increased complexity.

**Challenges:**
- Training GANs directly at 64x64 resolution often resulted in instability due to the quadratic growth in pixel count.
- Overfitting was more likely due to the relatively small dataset.

**Solutions:**
1. **Progressive Growing of GANs:**
   - Started training the networks on **16x16 images** to capture basic features.
    <div align="center">
      <table>
        <tr>
          <td align="center" style="padding: 15px;">
            <strong>16x16 Resolution</strong><br>
            <img src="https://github.com/user-attachments/assets/ac30e416-6ebc-46d2-acd9-b84601dfb6f0" width="250">
          </td>
          <td align="center" style="padding: 15px;">
            <strong>32x32 Resolution</strong><br>
            <img src="https://github.com/user-attachments/assets/423af04b-fd67-4f37-9cbf-25ee754659a3" width="250">
          </td>
          <td align="center" style="padding: 15px;">
            <strong>64x64 Resolution</strong><br>
            <img src="https://github.com/user-attachments/assets/5d382116-6126-4c5b-a403-237e05de82f5" width="350">
          </td>
        </tr>
      </table>
    </p>


2. **Frozen Layers:**
   - Added a **frozen layer** feature to all trainable layers. This allowed focusing on training new layers while keeping previously trained layers unchanged.

- **16x16 Output:** Basic shapes, indistinguishable details.
- **32x32 Output:** Clearer shapes with some visible details.
- **64x64 Output:** Detailed outputs, including hairstyles, helmets, belts, and capes.
  <p align="center">
    <img width="649" alt="fake_O" src="https://github.com/user-attachments/assets/cd5ff808-6a20-4dee-ad87-fe48bdd29904" />
    <p>The progression of a partially frozen network after extending the dimension from 16x16 to 32x32 to help the GAN adapt to changes</p>
  </p>


## Documentation

### Supported Layers

The framework supports various layers for building Generator and Discriminator (critic) architectures with customizable features. The following layers are implemented:

#### **Dense Layer**
- **Parameters:** `input_shape`, `batch_size`, `num_neurons`, `activation`, `frozen`, `batch_norm`.  
- Fully connected layer with adjustable neurons and optional batch normalization.

**Supported Activation Functions:**
- **ReLU (Rectified Linear Unit):** Introduces non-linearity.  
    <p align="center">
      <img width="543" alt="image" src="https://github.com/user-attachments/assets/334c447d-529f-4cae-bd95-215940f021b3" />
    </p>
- **Leaky ReLU:** Prevents dying neurons with adjustable `LRELU_ALPHA`.  
    <p align="center">
      <img width="586" alt="image" src="https://github.com/user-attachments/assets/762340de-64ab-4f17-b91d-09ca1d93c3bc" />
    </p>
- **Sigmoid:** Maps pre-activation outputs into the (0, 1) range.  
    <p align="center">
      <img width="558" alt="image" src="https://github.com/user-attachments/assets/59d49bf8-5be1-499c-a580-dab380cab172" />
    </p>

- **Linear:** Allows unbounded neuron output to the next layer (e.g., WGAN critic loss).

**Additional Features:**
- **Batch Normalization:** Normalizes pre-activations before passing to the activation function.  
- **Frozen Layer:** Disables updates for weights and biases during training.

#### **Conv2D Layer**
- **Parameters:** `input_shape`, `batch_size`, `num_filters`, `kernel_size`, `stride`, `padding`, `activation`, `frozen`.  
- Performs 2D convolution operations to extract spatial features from inputs.
- **Tahn:** Symmetrically maps pre-activations to the range (-1, 1).
    <p align="center">
      <img width="450" alt="image" src="https://github.com/user-attachments/assets/4cfca45a-1331-4196-ac54-77e6ddc0d8f7" />
    </p>

#### **TConv2D (Transposed Convolution)**
- **Parameters:** `input_shape`, `batch_size`, `num_filters`, `kernel_size`, `stride`, `padding`, `activation`, `batch_norm`, `frozen`.  
- Expands input pixels with filters, producing larger feature maps.

#### **Dropout**
- **Parameters:** `rate`.  
- Randomly "turns off" inputs with `rate * 100%` probability, preventing overfitting.

#### **Batch Normalization**
If enabled (`batch_norm`), normalizes each pre-activation channel across all samples in the batch.

- Input/Output Handling
Flattening and reshaping of inputs and outputs are handled automatically.

#### **Discriminator and Generator**
- Combine layers to build models.
- Initialize weights, freeze/unfreeze trainable parameters, and save/load weights using PyTorch-like `state_dict` and `load_dict` functions.

#### **train.py**
- Provides a sample implementation of the training flow using Binary Cross-Entropy (BCE) with a learning rate and scheduler.
  <p align="center">
    <img width="463" alt="image" src="https://github.com/user-attachments/assets/43568c9b-7a88-4ce6-9a0f-f46bc4c91970" />
  </p>

## Additional Features:
- Monitors Discriminator real/fake loss and Generator loss.
- Saves networks and hyperparameters in PyTorch dictionary format.
- Saves fake images produced by the Generator for visual inspection.
- Loads networks and hyperparameters from a specific epoch and batch number to resume training.

#### **generate.py**
- Uses a trained Generator to produce grids of fake images.
- Saves generated images for inspection.
