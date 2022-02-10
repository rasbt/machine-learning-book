
##  Chapter 12: Parallelizing Neural Network Training with PyTorch


### Chapter Outline

- PyTorch and training performance
  - Performance challenges
  - What is PyTorch?
  - How we will learn PyTorch
- First steps with PyTorch
  - Installing PyTorch
  - Creating tensors in PyTorch
  - Manipulating the data type and shape of a tensor
  - Applying mathematical operations to tensors
  - Split, stack, and concatenate tensors
- Building input pipelines in PyTorch
  - Creating a PyTorch DataLoader from existing tensors
  - Combining two tensors into a joint dataset
  - Shuffle, batch, and repeat
  - Creating a dataset from files on your local storage disk
  - Fetching available datasets from the torchvision.datasets library
- Building an NN model in PyTorch
  - The PyTorch neural network module (torch.nn)
  - Building a linear regression model
  - Model training via the torch.nn and torch.optim modules 
  - Building a multilayer perceptron for classifying flowers in the Iris dataset
  - Evaluating the trained model on the test dataset
  - Saving and reloading the trained model
- Choosing activation functions for multilayer neural networks
  - Logistic function recap
  - Estimating class probabilities in multiclass classification via the softmax function
  - Broadening the output spectrum using a hyperbolic tangent
  - Rectified linear unit activation
- Summary




**Installing PyTorch**

We recommend consulting the official [pytorch.org](https://pytorch.org) installer menu to select the conda or pip command to install PyTorch for your operating system. 



**Please refer to the [README.md](../ch01/README.md) file in [`../ch01`](../ch01) for more information about running the code examples.**

