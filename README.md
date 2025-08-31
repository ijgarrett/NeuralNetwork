# Neural Network from Scratch for Binary Classification

This is an improved, more usable version of my NeuralNetworkFromScratch that can be used in other scenarios like sentiment analysis. This project implements a fully connected neural network from scratch using only base Python and NumPy to perform classification on datasets. I completed this project in August 2025, building a complete pipeline including layer definitions, forward and backward propagation, loss computation, and parameter optimization with the Adam optimizer.

## Project Overview

The goal is to classify samples into two classes using a neural network implemented entirely from scratch. The network supports multiple layers, ReLU activation for hidden layers, softmax activation for the output layer, dropout regularization, L2 regularization, and training with the Adam optimizer. The pipeline includes preprocessing, one-hot encoding of labels, forward and backward passes, and training with mini-batches.

## Dataset

- Example dataset: Breast Cancer dataset from `sklearn.datasets`
- Features: 30 numerical features per sample
- Classes: 2 (benign, malignant)
- Split:
  - Training set: 80%
  - Test set: 20%
- Labels converted to one-hot encoding for compatibility with softmax output

## Tools and Libraries

- Python  
- NumPy (matrix operations, random initialization)  
- scikit-learn (`load_breast_cancer`, `train_test_split`, `StandardScaler`)  
- pandas (optional, data handling)  

## Process and Methodology

### 1. Neural Network Layers
- `Layer_Dense`: fully connected layer with weights and biases initialized from normal distribution, supports L2 regularization  
- `Activation_ReLU`: ReLU activation for hidden layers  
- `Activation_Softmax`: softmax activation for output layer  
- `Layer_Dropout`: dropout regularization to reduce overfitting  

### 2. Loss Function
- `Loss_CategoricalCrossEntropy`: categorical cross-entropy for multi-class (binary works as 2-class special case)  
- Combined layer `Activation_Softmax_Loss_CategoricalCrossEntropy` for output layer to optimize computation of forward and backward passes  

### 3. Optimizer
- `Optimizer_Adam` implements the Adam optimization algorithm with parameters:
  - Learning rate
  - Decay
  - Beta1 and Beta2 for momentum
  - Epsilon for numerical stability

### 4. Forward and Backward Pass
- Forward pass through each layer computes outputs from input features  
- Backward pass calculates gradients for weights and biases using chain rule and stores them in layers  
- Gradients include contributions from L2 regularization and dropout masking  

### 5. Training
- Shuffles dataset at each epoch  
- Mini-batch training for efficiency  
- Updates weights and biases using Adam optimizer  
- Verbose output logs training loss and accuracy per epoch  
- Optional validation set for tracking performance  

## Final Model Performance

- Test set accuracy evaluated using argmax of softmax output  
- Example run on Breast Cancer dataset achieved high accuracy (~95%)  
- Demonstrates that a neural network implemented from scratch can effectively perform binary classification  

## Files in This Project

- `neural_network.py` — defines layers, activations, loss functions, optimizer, and the NeuralNetwork class  
- `train_network.py` — example training script with dataset preprocessing, network creation, and training loop  
- `README.md` — project documentation  

## Timeline

8/31/25  

## Future Improvements

- Implement additional activation functions (Sigmoid, Tanh, LeakyReLU)  
- Add batch normalization  
