# Chengyuan_RBM

# Restricted Boltzmann Machine (RBM) Implementation

This project contains a Python implementation of a Restricted Boltzmann Machine (RBM), a type of stochastic neural network used for unsupervised learning.

## Description

An RBM is capable of learning a probability distribution over its set of inputs. This implementation is designed to understand and learn the underlying patterns of given input data.

## Features

- Initialization of visible and hidden biases
- Sampling methods using probabilities
- Sigmoid activation function
- Gibbs sampling step for hidden and visible layer sampling
- Weight updates based on gradient descent

## Requirements

- Python 3.x
- NumPy

Ensure you have the required packages installed by running:

```bash
pip install numpy

Usage

To use this RBM implementation, import the RBM class from the file and initialize it with the desired parameters:

from rbm import RBM  # Ensure your file is named rbm.py or change this accordingly

visible_dim = 6  # Number of visible nodes
hidden_dim = 3   # Number of hidden nodes
learning_rate = 0.1
number_of_iterations = 1000

rbm = RBM(visible_dim, hidden_dim, learning_rate, number_of_iterations)

Once the RBM is initialized, you can train it using your data:

data = np.array([...])  # Your data here as a NumPy array
rbm.train(data)

Acknowledgments

Inspired by the original papers on Restricted Boltzmann Machines.
Thanks to the community for helpful discussions and insights.
