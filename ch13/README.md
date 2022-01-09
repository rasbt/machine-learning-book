##  Chapter 13: Going Deeper – The Mechanics of PyTorch


### Chapter Outline

- The key features of 
- PyTorch's computation graphs
  - Understanding computation graphs
  - Creating a graph in PyTorch
- PyTorch tensor objects for storing and updating model parameters
- Computing gradients via automatic differentiation
  - Computing the gradients of the loss with respect to trainable variables
  - Understanding automatic differentiation
  - Adversarial examples
- Simplifying implementations of common architectures via the torch.nn module
  - Implementing models based on nn.Sequential
  - Choosing a loss function
  - Solving an XOR classification problem
  - Making model building more flexible with nn.Module
  - Writing custom layers in PyTorch
- Project one - predicting the fuel efficiency of a car
  - Working with feature columns
  - Training a DNN regression model
- Project two - classifying MNIST handwritten digits
- Higher-level PyTorch APIs: a short introduction to PyTorch Lightning
  - Setting up the PyTorch Lightning model
  - Setting up the data loaders for Lightning
  - Training the model using the PyTorch Lightning Trainer class
  - Evaluating the model using TensorBoard
- Summary

**Please refer to the [README.md](../ch01/README.md) file in [`../ch01`](../ch01) for more information about running the code examples.**