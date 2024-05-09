import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

# s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
s = '/home/jperkinson/dev/machine-learning-book-jperk224/ch02/iris.data'

df = pd.read_csv(s, header=None, encoding='utf-8')

# print(df.tail())

# select setosa and versicolor class labels
# take the 4th column (the class label) of the first 100 rows
# these are the class labels of interest for this exercise
y = df.iloc[0:100, 4]
y = np.where(y == 'Iris-setosa', 0, 1)

# extract sepal length and petal length into our feature matrix
X = df.iloc[0:100, [0,2]].values

# plot the data
# sepal length (x) and petal length (y) for Setosa (1st 50 rows of the data set)
# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')

# sepal length (x) and petal length (y) for Versicolor (Last 50 rows of the data set)
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')

# plt.xlabel('Sepal length [cm]')
# plt.ylabel('Petal length [cm]')
# plt.legend(loc='upper left')
# plt.show()

# time to train the model
# plot misclassification errors for each epoch to see is the algorithm converges
# cerror convergence uncovers our decision boundary
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Updates')
plt.show()
