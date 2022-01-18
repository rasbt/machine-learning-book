##  Chapter 16: Transformers – Improving Natural Language Processing with Attention Mechanisms

### Chapter Outline

- Adding an attention mechanism to RNNs
  - Attention helps RNNs with accessing information
  - The original attention mechanism for RNNs
  - Processing the inputs using a bidirectional RNN
  - Generating outputs from context vectors
  - Computing the attention weights
- Introducing the self-attention mechanism
  - Starting with a basic form of self-attention
  - Parameterizing the self-attention mechanism: scaled dot-product attention
- Attention is all we need: introducing the original transformer architecture
  - Encoding context embeddings via multi-head attention
  - Learning a language model: decoder and masked multi-head attention
  - Implementation details: positional encodings and layer normalization
- Building large-scale language models by leveraging unlabeled data
  - Pre-training and fine-tuning transformer models
  - Leveraging unlabeled data with GPT
  - Using GPT-2 to generate new text
  - Bidirectional pre-training with BERT
  - The best of both worlds: BART
- Fine-tuning a BERT model in PyTorch
  - Loading the IMDb movie review dataset
  - Tokenizing the dataset
  - Loading and fine-tuning a pre-trained BERT model
  - Fine-tuning a transformer more conveniently using the Trainer API
- Summary

**Please refer to the [README.md](../ch01/README.md) file in [`../ch01`](../ch01) for more information about running the code examples.**

