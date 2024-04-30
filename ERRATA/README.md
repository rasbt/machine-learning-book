# Current Errata

## Chapter 3

**Page 63**

A small stylistic issue: The summation symbol at the bottom currently shows 

```math
\sum_{i=1}
```

 but should be either 
 
 ```math
 \sum_{i}
```

or 

```math
\sum_{i=1}^{n}
```

**Page 66**

The doc strings of the LogisticRegressionGD classifier reference "Mean squared error loss" -- this is a copy-paste error and should be "Log loss".

**Page 84**

A larger gamma value should create a smaller (not larger) radius.

## Chapter 11

**Page 341**

Add bias unit to the net input.

**Page 354**

The MSE is normalized via `mse = mse/i` but should be normalized via `mse = mse/(i+1)` instead. (This does not affect the results in shown below though. The MSE is still 0.3.)

**Page 366**

It says 

```math
\frac{\partial L}{\partial w_{1,1}^{(\text {out })}} = ...
```

 but should be 
 
 ```math
\frac{\partial L}{\partial w_{1,1}^{(\text {h })}}
```

to match the figure above and text below.

**Page 361**

```math
\frac{\partial}{\partial w_{j, l}^{(l)}}=L(\boldsymbol{W}, \boldsymbol{b}) 
```

should be 

```math
\frac{\partial L}{\partial w_{j, l}^{(l)}}
```


## Chapter 12

**Page 380**

We use `TensorDataset` even though we defined the custom `JointDataset`

## Chapter 14

**Page 459**

The `conv1d()` and `conv2d()` functions on page 459 was improved through a kind [pull request](https://github.com/rasbt/machine-learning-book/pull/168) by [@JaGeo](https://github.com/JaGeo), enabling it to handle cases with strides different from (1,1).

## Chapter 13

**Page 431**

When using Torchmetrics 0.8.0 or newer, the following lines

```python
self.train_acc = Accuracy()
self.valid_acc = Accuracy()
self.test_acc = Accuracy()
```

need to be changed to

```python
self.train_acc = Accuracy(task="multiclass", num_classes=10)
self.valid_acc = Accuracy(task="multiclass", num_classes=10)
self.test_acc = Accuracy(task="multiclass", num_classes=10)
```

## Chapter 14

**Page 489**

Not an error, but for legibility, it would be good to change 

```python
for j in range(num_epochs):
    img_batch, label_batch = next(iter(data_loader))
```

to

```python
for img_batch, label_batch in data_loader:
```

## Chapter 15

**Page 505**

The equation is technically correct, but it looks like the character 0 (zero) was used instead of the letter o:

```math
\mathbf{o}^{\left( t \right)} = \sigma_{0}\left( \mathbf{W}_{ho}\mathbf{h}^{\left( t \right)}+\mathbf{b}_{0} \right)
```
should be

```math
\mathbf{o}^{\left( t \right)} = \sigma_{o}\left( \mathbf{W}_{ho}\mathbf{h}^{\left( t \right)}+\mathbf{b}_{o} \right)
```

**Page 530**

The line `from torch.utils.data import Dataset` appears twice.

## Chapter 16

**Page 547**

Not an error, but where we are summing over the columns via `attention_weights.sum(dim=1)`, we could mention that this matrix is symmetric and that we could also sum over the rows and get the same results.

---



For books printed before 16 Nov 2022, please see the [Old Errata](old-errata).



