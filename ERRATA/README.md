# Current Errata

## Chapter 3

**Page 66**

The doc strings of the LogisticRegressionGD classifier reference "Mean squared error loss" -- this is a copy-paste error and should be "Log loss".

## Chapter 11

**Page 341**

Add bias unit to the net input.

**Page 361**

$$\frac{\partial}{\partial w_{j, l}^{(l)}}=L(\boldsymbol{W}, \boldsymbol{b})$$ 

should be 

$$\frac{\partial L}{\partial w_{j, l}^{(l)}}$$ 

## Chapter 18

**Page 380**

We use `TensorDataset` even though we defined the custom `JointDataset`

## Chapter 15



**Page 530**

The line `from torch.utils.data import Dataset` appears twice.



---



For books printed before 16 Nov 2022, please see the [Old Errata](old-errata).



