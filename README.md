# Image Classification using PyTorch

This repository contains a Colab notebook for practicing image classification using **PyTorch**. The notebook demonstrates how to build, train, and evaluate a simple neural network for classifying images from the **Fashion MNIST** dataset. It also covers how to save the trained model and load it later for inference.

## Project Overview

In this project, we create a custom neural network model to classify images of clothing into one of 10 classes, using the Fashion MNIST dataset. The dataset consists of 28x28 grayscale images of fashion items, such as shirts, pants, shoes, etc.

The notebook includes:
- Loading and preprocessing the Fashion MNIST dataset
- Building a custom neural network model using PyTorch
- Training the model using the training data
- Evaluating the model on the test set
- Saving and loading the trained model
- Plotting some training images to visualize the dataset

## Dataset

The dataset used in this notebook is **Fashion MNIST**, a collection of 60,000 training images and 10,000 test images in 10 classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Fashion MNIST is available for download directly via PyTorch’s `torchvision` package.

## Requirements

To run this notebook, you’ll need the following Python libraries:

- **PyTorch** (`torch`, `torchvision`)
- **Matplotlib** (for plotting)
- **NumPy** (for numerical operations)
- **Google Colab** (if running in Colab)

You can install the required dependencies by running:

```bash
pip install torch torchvision matplotlib numpy

