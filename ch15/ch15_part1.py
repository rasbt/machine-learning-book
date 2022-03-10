# coding: utf-8


import sys
from python_environment_check import check_packages
import torch
import torch.nn as nn

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:





d = {
    'torch': '1.8.0',
}
check_packages(d)


# # Chapter 15: Modeling Sequential Data Using Recurrent Neural Networks (Part 1/3)

# **Outline**
# 
# - [Introducing sequential data](#Introducing-sequential-data)
#   - [Modeling sequential data -- order matters](#Modeling-sequential-data----order-matters)
#   - [Sequential data versus time series data](#Sequential-data-versus-time-series-data)
#   - [Representing sequences](#Representing-sequences)
#   - [The different categories of sequence modeling](#The-different-categories-of-sequence-modeling)
# - [RNNs for modeling sequences](#RNNs-for-modeling-sequences)
#   - [Understanding the dataflow in RNNs](#Understanding-the-dataflow-in-RNNs)
#   - [Computing activations in an RNN](#Computing-activations-in-an-RNN)
#   - [Hidden recurrence versus output recurrence](#Hidden-recurrence-versus-output-recurrence)
#   - [The challenges of learning long-range interactions](#The-challenges-of-learning-long-range-interactions)
#   - [Long short-term memory cells](#Long-short-term-memory-cells)





# # Introducing sequential data
# 
# ## Modeling sequential data⁠—order matters
# 
# ## Representing sequences
# 
# 





# ## The different categories of sequence modeling





# # RNNs for modeling sequences
# 
# ## Understanding the RNN looping mechanism
# 









# ## Computing activations in an RNN
# 









# ## Hidden-recurrence vs. output-recurrence








torch.manual_seed(1)

rnn_layer = nn.RNN(input_size=5, hidden_size=2, num_layers=1, batch_first=True) 

w_xh = rnn_layer.weight_ih_l0
w_hh = rnn_layer.weight_hh_l0
b_xh = rnn_layer.bias_ih_l0
b_hh = rnn_layer.bias_hh_l0

print('W_xh shape:', w_xh.shape)
print('W_hh shape:', w_hh.shape)
print('b_xh shape:', b_xh.shape)
print('b_hh shape:', b_hh.shape)




x_seq = torch.tensor([[1.0]*5, [2.0]*5, [3.0]*5]).float()

## output of the simple RNN:
output, hn = rnn_layer(torch.reshape(x_seq, (1, 3, 5)))

## manually computing the output:
out_man = []
for t in range(3):
    xt = torch.reshape(x_seq[t], (1, 5))
    print(f'Time step {t} =>')
    print('   Input           :', xt.numpy())
    
    ht = torch.matmul(xt, torch.transpose(w_xh, 0, 1)) + b_xh    
    print('   Hidden          :', ht.detach().numpy())
    
    if t>0:
        prev_h = out_man[t-1]
    else:
        prev_h = torch.zeros((ht.shape))

    ot = ht + torch.matmul(prev_h, torch.transpose(w_hh, 0, 1)) + b_hh
    ot = torch.tanh(ot)
    out_man.append(ot)
    print('   Output (manual) :', ot.detach().numpy())
    print('   RNN output      :', output[:, t].detach().numpy())
    print()


# ## The challenges of learning long-range interactions
# 





# 
# ## Long Short-Term Memory cells 





# 
# ---

# 
# 
# Readers may ignore the next cell.
# 




