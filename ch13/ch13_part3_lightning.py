# coding: utf-8


import sys
from python_environment_check import check_packages
import pytorch_lightning as pl
import torch 
import torch.nn as nn 
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:





d = {
    'torch': '1.8',
    'torchvision': '0.9.0',
    'tensorboard': '2.7.0',
    'pytorch_lightning': '1.5.0',
    'torchmetrics': '0.6.2'
}
check_packages(d)


# # Chapter 13: Going Deeper -- the Mechanics of PyTorch (Part 3/3)

# **Outline**
# 
# - [Higher-level PyTorch APIs: a short introduction to PyTorch Lightning](#Higher-level-PyTorch-APIs-a-short-introduction-to-PyTorch-Lightning)
#   - [Setting up the PyTorch Lightning model](#Setting-up-the-PyTorch-Lightning-model)
#   - [Setting up the data loaders for Lightning](#Setting-up-the-data-loaders-for-Lightning)
#   - [Training the model using the PyTorch Lightning Trainer class](#Training-the-model-using-the-PyTorch-Lightning-Trainer-class)
#   - [Evaluating the model using TensorBoard](#Evaluating-the-model-using-TensorBoard)
# - [Summary](#Summary)

# ## Higher-level PyTorch APIs: a short introduction to PyTorch Lightning

# ### Setting up the PyTorch Lightning model

# ## Higher-level PyTorch APIs: a short introduction to PyTorch Lightning

# ### Setting up the PyTorch Lightning model








class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, image_shape=(1, 28, 28), hidden_units=(32, 16)):
        super().__init__()
        
        # new PL attributes:
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()
        
        # Model similar to previous section:
        input_size = image_shape[0] * image_shape[1] * image_shape[2] 
        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units: 
            layer = nn.Linear(input_size, hidden_unit) 
            all_layers.append(layer) 
            all_layers.append(nn.ReLU()) 
            input_size = hidden_unit 
 
        all_layers.append(nn.Linear(hidden_units[-1], 10)) 
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss
    
    def validation_epoch_end(self, outs):
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


# ### Setting up the data loaders



 




class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_path='./'):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def prepare_data(self):
        MNIST(root=self.data_path, download=True) 

    def setup(self, stage=None):
        # stage is either 'fit', 'validate', 'test', or 'predict'
        # here note relevant
        mnist_all = MNIST( 
            root=self.data_path,
            train=True,
            transform=self.transform,  
            download=False
        ) 

        self.train, self.val = random_split(
            mnist_all, [55000, 5000], generator=torch.Generator().manual_seed(1)
        )

        self.test = MNIST( 
            root=self.data_path,
            train=False,
            transform=self.transform,  
            download=False
        ) 

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=4)
    
    
torch.manual_seed(1) 
mnist_dm = MnistDataModule()


# ### Training the model using the PyTorch Lightning Trainer class





mnistclassifier = MultiLayerPerceptron()

callbacks = [ModelCheckpoint(save_top_k=1, mode='max', monitor="valid_acc")] # save top 1 model

if torch.cuda.is_available(): # if you have GPUs
    trainer = pl.Trainer(max_epochs=10, callbacks=callbacks, gpus=1)
else:
    trainer = pl.Trainer(max_epochs=10, callbacks=callbacks)

trainer.fit(model=mnistclassifier, datamodule=mnist_dm)


# ### Evaluating the model using TensorBoard



trainer.test(model=mnistclassifier, datamodule=mnist_dm, ckpt_path='best')










# Start tensorboard








path = 'lightning_logs/version_0/checkpoints/epoch=8-step=7739.ckpt'

if torch.cuda.is_available(): # if you have GPUs
    trainer = pl.Trainer(
        max_epochs=15, callbacks=callbacks, resume_from_checkpoint=path, gpus=1
    )
else:
    trainer = pl.Trainer(
        max_epochs=15, callbacks=callbacks, resume_from_checkpoint=path
    )

trainer.fit(model=mnistclassifier, datamodule=mnist_dm)












trainer.test(model=mnistclassifier, datamodule=mnist_dm)




trainer.test(model=mnistclassifier, datamodule=mnist_dm, ckpt_path='best')




path = "lightning_logs/version_0/checkpoints/epoch=13-step=12039.ckpt"
model = MultiLayerPerceptron.load_from_checkpoint(path)


# ## Summary

# ---
# 
# Readers may ignore the next cell.




