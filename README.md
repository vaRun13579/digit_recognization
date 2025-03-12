# MNIST Handwritten Digit Classification using PyTorch

## Overview
This project implements a simple feedforward neural network using PyTorch to classify handwritten digits from the MNIST dataset. The model is trained using the Adam optimizer and cross-entropy loss function.

## Prerequisites
Make sure you have Python installed along with the following libraries:
- PyTorch
- Torchvision
- Matplotlib
- NumPy (implicitly required by PyTorch)

You can install the necessary packages using:
```sh
pip install torch torchvision matplotlib
```

## Dataset
The MNIST dataset is automatically downloaded and loaded using `torchvision.datasets.MNIST`. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9), each of size 28x28 pixels.

## Code Explanation
### 1. Import Necessary Libraries
The script imports PyTorch, Torchvision, and other necessary modules to handle data processing, model creation, and training.

### 2. Load and Preprocess Data
The dataset is loaded using `torchvision.datasets.MNIST` with transformations applied:
- Convert images to tensors (`ToTensor()`)
- Normalize pixel values to [-1, 1] (`Normalize((0.5,), (0.5,))`)

The dataset is then divided into training and test sets using DataLoader.

### 3. Define the Neural Network
A simple feedforward neural network `SimpleNN` is created with three fully connected layers:
- **Input Layer**: 28×28 = 784 neurons
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons (digits 0-9)

**Note:** The `__init__` method has an incorrect definition (`_init_` instead of `__init__`). Fix it before running the script.

### 4. Training the Model
The model is trained for 5 epochs using:
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate = 0.001)
- **Batch Size**: 64

Loss is printed every 100 mini-batches.

### 5. Evaluating the Model
The trained model is tested on the test dataset to calculate accuracy.

### 6. Visualizing Predictions
A test image is displayed along with its predicted label using `matplotlib`.

## How to Run
1. Save the script in a Python file (e.g., `mnist_classification.py`).
2. Run the script:
   ```sh
   python mnist_classification.py
   ```
3. The script will train the model and display the accuracy on test images.
4. A sample image with its predicted label will be shown.

## Expected Output
- Training loss updates every 100 batches.
- Final accuracy on the test set.
- A sample test image with predicted and true labels.

## License
This project is open-source and can be modified as needed.

