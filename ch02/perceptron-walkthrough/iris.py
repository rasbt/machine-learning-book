import os
import pandas as pd

import perceptron

# s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
s = '/home/jperkinson/dev/machine-learning-book-jperk224/ch02/iris.data'

df = pd.read_csv(s, header=None, encoding='utf-8')

print(df.tail())
