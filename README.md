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
rbm.train(data)```

## References

- Fischer, A., & Igel, C. (2014). Training Restricted Boltzmann Machines: An Introduction. _Pattern Recognition_, 47(1), 25-39. [https://doi.org/10.1016/j.patcog.2013.05.025](https://doi.org/10.1016/j.patcog.2013.05.025)

- Glorot, X., & Bengio, Y. Understanding the Difficulty of Training Deep Feedforward Neural Networks. 

- Hinton, G. E. (2012). A Practical Guide to Training Restricted Boltzmann Machines. In G. Montavon, G. B. Orr, & K.-R. Müller (Eds.), _Neural Networks: Tricks of the Trade_ (Vol. 7700, pp. 599-619). Springer Berlin Heidelberg. [https://doi.org/10.1007/978-3-642-35289-8_32](https://doi.org/10.1007/978-3-642-35289-8_32)

- Salakhutdinov, R., et al. (2007). Restricted Boltzmann Machines for Collaborative Filtering. _Proceedings of the 24th International Conference on Machine Learning_ (pp. 791-798). ACM. [https://doi.org/10.1145/1273496.1273596](https://doi.org/10.1145/1273496.1273596)

- Tieleman, T. (2008). Training Restricted Boltzmann Machines Using Approximations to the Likelihood Gradient. _Proceedings of the 25th International Conference on Machine Learning - ICML ’08_ (pp. 1064-1071). ACM Press. [https://doi.org/10.1145/1390156.1390290](https://doi.org/10.1145/1390156.1390290)


**Thanks to Dr.Soohyun Kim for help**
