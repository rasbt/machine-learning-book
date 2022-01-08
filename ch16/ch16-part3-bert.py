# coding: utf-8


import sys
from python_environment_check import check_packages
import gzip
import shutil
import time
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torchtext
import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric
import numpy as np

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:





d = {
    'pandas': '1.3.2',
    'torch': '1.9.0',
    'torchtext': '0.11.0',
    'datasets': '1.11.0',
    'transformers': '4.9.1',
}
check_packages(d)


# # Chapter 16: Transformers – Improving Natural Language Processing with Attention Mechanisms (Part 3/3)

# **Outline**
# 
# - [Fine-tuning a BERT model in PyTorch](#Fine-tuning-a-BERT-model-in-PyTorch)
#   - [Loading the IMDb movie review dataset](#Loading-the-IMDb-movie-review-dataset)
#   - [Tokenizing the dataset](#Tokenizing-the-dataset)
#   - [Loading and fine-tuning a pre-trained BERT model](#[Loading-and-fine-tuning-a-pre-trained-BERT-model)
#   - [Fine-tuning a transformer more conveniently using the Trainer API](#Fine-tuning-a-transformer-more-conveniently-using-the-Trainer-API)
# - [Summary](#Summary)

# ---
# 
# Quote from https://huggingface.co/transformers/custom_datasets.html:
# 
# > DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased , runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark.
# 
# ---





# ## Fine-tuning a BERT model in PyTorch

# ### Loading the IMDb movie review dataset
# 







# **General Settings**



torch.backends.cudnn.deterministic = True
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 3


# **Download Dataset**

# The following cells will download the IMDB movie review dataset (http://ai.stanford.edu/~amaas/data/sentiment/) for positive-negative sentiment classification in as CSV-formatted file:



url = "https://github.com/rasbt/machine-learning-book/raw/main/ch08/movie_data.csv.gz"
filename = url.split("/")[-1]

with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)

with gzip.open('movie_data.csv.gz', 'rb') as f_in:
    with open('movie_data.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# Check that the dataset looks okay:



df = pd.read_csv('movie_data.csv')
df.head()




df.shape


# **Split Dataset into Train/Validation/Test**



train_texts = df.iloc[:35000]['review'].values
train_labels = df.iloc[:35000]['sentiment'].values

valid_texts = df.iloc[35000:40000]['review'].values
valid_labels = df.iloc[35000:40000]['sentiment'].values

test_texts = df.iloc[40000:]['review'].values
test_labels = df.iloc[40000:]['sentiment'].values


# ## Tokenizing the dataset



tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')




train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)




train_encodings[0]


# **Dataset Class and Loaders**



class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
valid_dataset = IMDbDataset(valid_encodings, valid_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)




train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)


# ## Loading and fine-tuning a pre-trained BERT model



model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(DEVICE)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=5e-5)


# **Train Model -- Manual Training Loop**



def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        
        for batch_idx, batch in enumerate(data_loader):
        
        ### Prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
        
        return correct_pred.float()/num_examples * 100




start_time = time.time()

for epoch in range(NUM_EPOCHS):
    
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
        
        ### Prepare data
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        ### Forward
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']
        
        ### Backward
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        ### Logging
        if not batch_idx % 250:
            print (f'Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d} | '
                   f'Batch {batch_idx:04d}/{len(train_loader):04d} | '
                   f'Loss: {loss:.4f}')
            
    model.eval()

    with torch.set_grad_enabled(False):
        print(f'Training accuracy: '
              f'{compute_accuracy(model, train_loader, DEVICE):.2f}%'
              f'\nValid accuracy: '
              f'{compute_accuracy(model, valid_loader, DEVICE):.2f}%')
        
    print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
    
print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')




del model # free memory


# ### Fine-tuning a transformer more conveniently using the Trainer API

# Reload pretrained model:



model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(DEVICE)
model.train();






optim = torch.optim.Adam(model.parameters(), lr=5e-5)
training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=3,     
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,   
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)




# install dataset via pip install datasets


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred # logits are a numpy array, not pytorch tensor
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(
               predictions=predictions, references=labels)




optim = torch.optim.Adam(model.parameters(), lr=5e-5)


training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=3,     
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,   
    logging_dir='./logs',
    logging_steps=10
)

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    optimizers=(optim, None) # optimizer and learning rate scheduler
)

# force model to only use 1 GPU (even if multiple are availabe)
# to compare more fairly to previous code

trainer.args._n_gpu = 1




start_time = time.time()
trainer.train()
print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')




trainer.evaluate()




model.eval()
model.to(DEVICE)
print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')


# ...

# ---
# 
# Readers may ignore the next cell.









