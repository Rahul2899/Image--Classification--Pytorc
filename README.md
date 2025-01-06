# Image Classification Using Pytorch

This repository contains a PyTorch implementation for training and testing a neural network on the FashionMNIST dataset. The project demonstrates a simple pipeline for loading data, building a neural network model, training it, and evaluating its performance.

## Requirements

Ensure you have the following libraries installed:

- PyTorch 
- torchvision
- matplotlib
- numpy

Install the required libraries using pip:
```bash
pip install torch torchvision matplotlib numpy
```

## Project Structure

1. **Data Loading:**
   - Downloads the FashionMNIST dataset and applies transformations using `ToTensor`.
   - Prepares data loaders for training and testing.

2. **Model Definition:**
   - Defines a simple feedforward neural network with three fully connected layers and ReLU activations.

3. **Training Loop:**
   - Implements the training process with loss calculation and backpropagation.

4. **Testing Loop:**
   - Evaluates the model on the test dataset, computing accuracy and loss.

5. **Saving and Loading Model:**
   - Demonstrates saving the model's state and reloading it for predictions.

6. **Prediction and Visualization:**
   - Predicts the class of a test image and visualizes a grid of training images with labels.

## Usage

### Running the Code in Google Colab

1. **Open Colab:**
   - Go to [Google Colab](https://colab.research.google.com/).

2. **Upload the Notebook:**
   - Upload the provided notebook file or copy the code into a new Colab notebook.

3. **Run the Cells:**
   - Execute the cells step by step to train and test the model.

4. **Save the Model:**
   - You can save the trained model to your Google Drive or local machine.

### Key Components

#### Model Architecture
```python
class NeuralNetworks(nn.Module):
    def __init__(self):
        super(NeuralNetworks, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relue_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relue_stack(x)
        return logits
```

#### Training and Testing Loops
```python
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}")


def test(dataloader, model, loss_fn):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).sum().item()
    print(f"Accuracy: {correct / len(dataloader.dataset):>0.1f}%")
```

## Results

After training for 50 epochs, the model achieves a test accuracy of ~85%. The results may vary depending on hyperparameters and hardware.


## Future Improvements

- Implement data augmentation for improved generalization.
- Experiment with different optimizers (e.g., Adam, RMSprop).
- Add more layers or explore CNN architectures for better accuracy.


## Acknowledgments

- PyTorch official documentation: https://pytorch.org/docs/
- FashionMNIST dataset: https://github.com/zalandoresearch/fashion-mnist
