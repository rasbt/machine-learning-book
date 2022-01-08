# coding: utf-8


import sys
from python_environment_check import check_packages
import torch
import torch.nn.functional as F

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:





d = {
    'torch': '1.9.0',
}
check_packages(d)


# # Chapter 16: Transformers – Improving Natural Language Processing with Attention Mechanisms (Part 1/3)

# **Outline**
# 
# - [Adding an attention mechanism to RNNs](#Adding-an-attention-mechanism-to-RNNs)
#   - [Attention helps RNNs with accessing information](#Attention-helps-RNNs-with-accessing-information)
#   - [The original attention mechanism for RNNs](#The-original-attention-mechanism-for-RNNs)
#   - [Processing the inputs using a bidirectional RNN](#Processing-the-inputs-using-a-bidirectional-RNN)
#   - [Generating outputs from context vectors](#Generating-outputs-from-context-vectors)
#   - [Computing the attention weights](#Computing-the-attention-weights)
# - [Introducing the self-attention mechanism](#Introducing-the-self-attention-mechanism)
#   - [Starting with a basic form of self-attention](#Starting-with-a-basic-form-of-self-attention)
#   - [Parameterizing the self-attention mechanism: scaled dot-product attention](#Parameterizing-the-self-attention-mechanism-scaled-dot-product-attention)
# - [Attention is all we need: introducing the original transformer architecture](#Attention-is-all-we-need-introducing-the-original-transformer-architecture)
#   - [Encoding context embeddings via multi-head attention](#Encoding-context-embeddings-via-multi-head-attention)
#   - [Learning a language model: decoder and masked multi-head attention](#Learning-a-language-model-decoder-and-masked-multi-head-attention)
#   - [Implementation details: positional encodings and layer normalization](#Implementation-details-positional-encodings-and-layer-normalization)





# ## Adding an attention mechanism to RNNs

# ### Attention helps RNNs with accessing information









# ### The original attention mechanism for RNNs





# ### Processing the inputs using a bidirectional RNN
# ### Generating outputs from context vectors
# ### Computing the attention weights

# ## Introducing the self-attention mechanism

# ### Starting with a basic form of self-attention

# - Assume we have an input sentence that we encoded via a dictionary, which maps the words to integers as discussed in the RNN chapter:





# input sequence / sentence:
#  "Can you help me to translate this sentence"

sentence = torch.tensor(
    [0, # can
     7, # you     
     1, # help
     2, # me
     5, # to
     6, # translate
     4, # this
     3] # sentence
)

sentence


# - Next, assume we have an embedding of the words, i.e., the words are represented as real vectors.
# - Since we have 8 words, there will be 8 vectors. Each vector is 16-dimensional:



torch.manual_seed(123)
embed = torch.nn.Embedding(10, 16)
embedded_sentence = embed(sentence).detach()
embedded_sentence.shape


# - The goal is to compute the context vectors $\boldsymbol{z}^{(i)}=\sum_{j=1}^{T} \alpha_{i j} \boldsymbol{x}^{(j)}$, which involve attention weights $\alpha_{i j}$.
# - In turn, the attention weights $\alpha_{i j}$ involve the $\omega_{i j}$ values
# - Let's start with the $\omega_{i j}$'s first, which are computed as dot-products:
# 
# $$\omega_{i j}=\boldsymbol{x}^{(i)^{\top}} \boldsymbol{x}^{(j)}$$
# 
# 



omega = torch.empty(8, 8)

for i, x_i in enumerate(embedded_sentence):
    for j, x_j in enumerate(embedded_sentence):
        omega[i, j] = torch.dot(x_i, x_j)


# - Actually, let's compute this more efficiently by replacing the nested for-loops with a matrix multiplication:



omega_mat = embedded_sentence.matmul(embedded_sentence.T)




torch.allclose(omega_mat, omega)


# - Next, let's compute the attention weights by normalizing the "omega" values so they sum to 1
# 
# $$\alpha_{i j}=\frac{\exp \left(\omega_{i j}\right)}{\sum_{j=1}^{T} \exp \left(\omega_{i j}\right)}=\operatorname{softmax}\left(\left[\omega_{i j}\right]_{j=1 \ldots T}\right)$$
# 
# $$\sum_{j=1}^{T} \alpha_{i j}=1$$




attention_weights = F.softmax(omega, dim=1)
attention_weights.shape


# - We can conform that the columns sum up to one:



attention_weights.sum(dim=1)






# - Now that we have the attention weights, we can compute the context vectors $\boldsymbol{z}^{(i)}=\sum_{j=1}^{T} \alpha_{i j} \boldsymbol{x}^{(j)}$, which involve attention weights $\alpha_{i j}$
# - For instance, to compute the context-vector of the 2nd input element (the element at index 1), we can perform the following computation:



x_2 = embedded_sentence[1, :]
context_vec_2 = torch.zeros(x_2.shape)
for j in range(8):
    x_j = embedded_sentence[j, :]
    context_vec_2 += attention_weights[1, j] * x_j
    
context_vec_2


# - Or, more effiently, using linear algebra and matrix multiplication:



context_vectors = torch.matmul(
        attention_weights, embedded_sentence)


torch.allclose(context_vec_2, context_vectors[1])


# ###  Parameterizing the self-attention mechanism: scaled dot-product attention







torch.manual_seed(123)

d = embedded_sentence.shape[1]
U_query = torch.rand(d, d)
U_key = torch.rand(d, d)
U_value = torch.rand(d, d)




x_2 = embedded_sentence[1]
query_2 = U_query.matmul(x_2)




key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)




keys = U_key.matmul(embedded_sentence.T).T
torch.allclose(key_2, keys[1])




values = U_value.matmul(embedded_sentence.T).T
torch.allclose(value_2, values[1])




omega_23 = query_2.dot(keys[2])
omega_23




omega_2 = query_2.matmul(keys.T)
omega_2




attention_weights_2 = F.softmax(omega_2 / d**0.5, dim=0)
attention_weights_2




#context_vector_2nd = torch.zeros(values[1, :].shape)
#for j in range(8):
#    context_vector_2nd += attention_weights_2[j] * values[j, :]
    
#context_vector_2nd




context_vector_2 = attention_weights_2.matmul(values)
context_vector_2


# ## Attention is all we need: introducing the original transformer architecture





# ###  Encoding context embeddings via multi-head attention 



torch.manual_seed(123)

d = embedded_sentence.shape[1]
one_U_query = torch.rand(d, d)




h = 8
multihead_U_query = torch.rand(h, d, d)
multihead_U_key = torch.rand(h, d, d)
multihead_U_value = torch.rand(h, d, d)




multihead_query_2 = multihead_U_query.matmul(x_2)
multihead_query_2.shape




multihead_key_2 = multihead_U_key.matmul(x_2)
multihead_value_2 = multihead_U_value.matmul(x_2)




multihead_key_2[2]




stacked_inputs = embedded_sentence.T.repeat(8, 1, 1)
stacked_inputs.shape




multihead_keys = torch.bmm(multihead_U_key, stacked_inputs)
multihead_keys.shape




multihead_keys = multihead_keys.permute(0, 2, 1)
multihead_keys.shape




multihead_keys[2, 1] # index: [2nd attention head, 2nd key]




multihead_values = torch.matmul(multihead_U_value, stacked_inputs)
multihead_values = multihead_values.permute(0, 2, 1)




multihead_z_2 = torch.rand(8, 16)








linear = torch.nn.Linear(8*16, 16)
context_vector_2 = linear(multihead_z_2.flatten())
context_vector_2.shape


# ### Learning a language model: decoder and masked multi-head attention





# ### Implementation details: positional encodings and layer normalization





# ---
# 
# Readers may ignore the next cell.









