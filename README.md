# NeuralNetX: A Modular Neural Network Suite for Digit Classification and Language Identification

## Project Overview

NeuralNetX is a self-driven machine learning project focused on building and training neural networks from scratch using PyTorch. The project covers key machine learning concepts and architectures such as the perceptron, nonlinear regression, digit classification using the MNIST dataset, and language identification via recurrent neural networks (RNNs). It culminates with a custom implementation of convolution operations applied to image data.

This project showcases practical skills in designing, implementing, and optimizing neural network models, including handling batching, designing architectures, tuning hyperparameters, and working with real-world datasets.

---

## Features

- **Perceptron Implementation**: Built a binary perceptron model capable of binary classification.
- **Nonlinear Regression**: Designed a feed-forward neural network to approximate sinusoidal functions.
- **Digit Classification**: Developed a neural network that classifies handwritten digits from the MNIST dataset, achieving >97% accuracy.
- **Language Identification**: Created a recurrent neural network (RNN) to identify the language of single words from five languages.
- **Custom Convolution Layer**: Implemented convolution operations from first principles to enhance digit classification with convolutional neural networks.

---

## Installation

This project requires Python 3.x and the following libraries:

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

### Setup

1. Create and activate a conda environment (recommended):

```bash
conda create -n neuralnetx python=3.9
conda activate neuralnetx
Install required packages:

bash
Copy
Edit
pip install numpy matplotlib torch
Verify installation by running:

bash
Copy
Edit
python autograder.py --check-dependencies
Project Structure
models.py - Contains implementations of various neural network models: Perceptron, Regression, Digit Classification, Language Identification, and Convolutional models.

autograder.py - Automated testing suite for validating model correctness.

backend.py - Supporting backend code (not edited).

data/ - Datasets including MNIST digit images and multilingual word data.

How to Run
Train and evaluate each model by running the autograder with the corresponding question flag:

Perceptron:
python autograder.py -q q1

Nonlinear Regression:
python autograder.py -q q2

Digit Classification:
python autograder.py -q q3

Language Identification:
python autograder.py -q q4

Convolutional Digit Classification (Extra Credit):
python autograder.py -q q5

Design Considerations
Implemented batch processing for efficient model training.

Applied ReLU activations for non-linearity between layers.

Utilized PyTorch’s autograd and optimizer frameworks for gradient computation and parameter updates.

Designed recurrent neural networks to handle variable-length inputs for language identification.

Constructed convolution operations manually to deepen understanding of spatial feature extraction.

Results
Achieved over 97% accuracy on MNIST digit classification.

Achieved over 81% accuracy on language identification task.

Developed a modular and extensible codebase suitable for experimenting with network architectures and hyperparameters.

Future Improvements
Integrate GPU acceleration for faster training.

Expand dataset variety for language identification.

Experiment with deeper convolutional architectures and regularization techniques.