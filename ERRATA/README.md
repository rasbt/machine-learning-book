# *Machine Learning with PyTorch and Scikit-Learn* Book

##  Errata



### Chapter 2



**Page 24**

> It is important to note that the convergence of the perceptron is only guaranteed if the two classes are linearly separable, which means that the two classes cannot be perfectly separated by a linear decision boundary. 

- Linearly separable means that the two classes ***can*** be perfectly separated (it mistakenly said "cannot"). [[#10](https://github.com/rasbt/machine-learning-book/issues/10)]



### Chapter 6

**Page 188**

The following import is missing:

    >>> import scipy.stats


### Chapter 12

**Page 380**

> We can also simply utilize the torch.utils.data.TensorDataset class, if the second dataset is a labeled dataset in the form of tensors. So, instead of using our self-defined Dataset class, JointDataset, we can create a joint dataset as follows:
>
> **>>>** joint_dataset = JointDataset(t_x, t_y)

- Here, we mistakenly used `JointDataset` again. It should have been

```python
from torch.utils.data import TensorDataset
joint_dataset = TensorDataset(t_x, t_y)
```

**Page 397**

In the line 

```python
accuracy_hist[epoch] += is_correct.mean()
```

it should be `is_correct.sum()` instead of `is_correct.mean()`. The resulting figures etc. are all correct, though.


### Chapter 15

**Page 508**

In the following line

    ht = torch.matmul(xt, torch.transpose(w_xh, 0, 1)) + b_hh

the bias should be  `b_xh` instead of `b_hh`. However, the resulting output is correct.




