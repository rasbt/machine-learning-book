# coding: utf-8


import sys
from python_environment_check import check_packages
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer
from transformers import GPT2Model

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:





d = {
    'torch': '1.9.0',
    'transformers': '4.9.1',
}
check_packages(d)


# # Chapter 16: Transformers – Improving Natural Language Processing with Attention Mechanisms (Part 2/3)

# **Outline**
# 
# - [Building large-scale language models by leveraging unlabeled data](#Building-large-scale-language-models-by-leveraging-unlabeled-data)
#   - [Pre-training and fine-tuning transformer models](#Pre-training-and-fine-tuning-transformer-models)
#   - [Leveraging unlabeled data with GPT](#Leveraging-unlabeled-data-with-GPT)
#   - [Using GPT-2 to generate new text](#Using-GPT-2-to-generate-new-text)
#   - [Bidirectional pre-training with BERT](#Bidirectional-pre-training-with-BERT)
#   - [The best of both worlds: BART](#The-best-of-both-worlds-BART)





# ## Building large-scale language models by leveraging unlabeled data
# ##  Pre-training and fine-tuning transformer models
# 
# 





# ## Leveraging unlabeled data with GPT









# ### Using GPT-2 to generate new text





generator = pipeline('text-generation', model='gpt2')
set_seed(123)
generator("Hey readers, today is",
          max_length=20,
          num_return_sequences=3)





tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Let us encode this sentence"
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input




model = GPT2Model.from_pretrained('gpt2')




output = model(**encoded_input)
output['last_hidden_state'].shape


# ### Bidirectional pre-training with BERT
# 













# ### The best of both worlds: BART





# ---
# 
# Readers may ignore the next cell.









