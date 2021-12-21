# coding: utf-8


import sys
from python_environment_check import check_packages
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import pathlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from itertools import islice

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:





d = {
    'numpy': '1.21.2',
    'matplotlib': '3.4.3',
    'torch': '1.9.0',
}
check_packages(d)


# # Chapter 12: Parallelizing Neural Network Training with PyTorch (Part 1/2)
# 

# - [PyTorch and training performance](#PyTorch-and-training-performance)
#   - [Performance challenges](#Performance-challenges)
#   - [What is PyTorch?](#What-is-PyTorch?)
#   - [How we will learn PyTorch](#How-we-will-learn-PyTorch)
# - [First steps with PyTorch](#First-steps-with-PyTorch)
#   - [Installing PyTorch](#Installing-PyTorch)
#   - [Creating tensors in PyTorch](#Creating-tensors-in-PyTorch)
#   - [Manipulating the data type and shape of a tensor](#Manipulating-the-data-type-and-shape-of-a-tensor)
#   - [Applying mathematical operations to tensors](#Applying-mathematical-operations-to-tensors)
#   - [Split, stack, and concatenate tensors](#Split,-stack,-and-concatenate-tensors)

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).





# ## PyTorch and training performance

# ### Performance challenges



IPythonImage(filename='figures/12_01.png', width=500)


# ### What is PyTorch?



IPythonImage(filename='figures/12_02.png', width=500)


# ### How we will learn PyTorch

# ## First steps with PyTorch

# ### Installing PyTorch



#! pip install torch





print('PyTorch version:', torch.__version__)

np.set_printoptions(precision=3)






# ### Creating tensors in PyTorch



a = [1, 2, 3]
b = np.array([4, 5, 6], dtype=np.int32)

t_a = torch.tensor(a)
t_b = torch.from_numpy(b)

print(t_a)
print(t_b)




torch.is_tensor(a), torch.is_tensor(t_a)




t_ones = torch.ones(2, 3)

t_ones.shape




print(t_ones)




rand_tensor = torch.rand(2,3)

print(rand_tensor)


# ### Manipulating the data type and shape of a tensor



t_a_new = t_a.to(torch.int64)

print(t_a_new.dtype)




t = torch.rand(3, 5)

t_tr = torch.transpose(t, 0, 1)
print(t.shape, ' --> ', t_tr.shape)




t = torch.zeros(30)

t_reshape = t.reshape(5, 6)

print(t_reshape.shape)




t = torch.zeros(1, 2, 1, 4, 1)

t_sqz = torch.squeeze(t, 2)

print(t.shape, ' --> ', t_sqz.shape)


# ### Applying mathematical operations to tensors



torch.manual_seed(1)

t1 = 2 * torch.rand(5, 2) - 1
t2 = torch.normal(mean=0, std=1, size=(5, 2))




t3 = torch.multiply(t1, t2)
print(t3)




t4 = torch.mean(t1, axis=0)
print(t4)




t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))

print(t5)




t6 = torch.matmul(torch.transpose(t1, 0, 1), t2)

print(t6)




norm_t1 = torch.linalg.norm(t1, ord=2, dim=1)

print(norm_t1)




np.sqrt(np.sum(np.square(t1.numpy()), axis=1))


# ### Split, stack, and concatenate tensors



torch.manual_seed(1)

t = torch.rand(6)

print(t)

t_splits = torch.chunk(t, 3)

[item.numpy() for item in t_splits]




torch.manual_seed(1)
t = torch.rand(5)

print(t)

t_splits = torch.split(t, split_size_or_sections=[3, 2])
 
[item.numpy() for item in t_splits]




A = torch.ones(3)
B = torch.zeros(2)

C = torch.cat([A, B], axis=0)
print(C)




A = torch.ones(3)
B = torch.zeros(3)

S = torch.stack([A, B], axis=1)
print(S)


# ## Building input pipelines in PyTorch

# ### Creating a PyTorch DataLoader from existing tensors




t = torch.arange(6, dtype=torch.float32)
data_loader = DataLoader(t)




for item in data_loader:
    print(item)




data_loader = DataLoader(t, batch_size=3, drop_last=False)

for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', batch)


# ### Combining two tensors into a joint dataset




class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]




torch.manual_seed(1)

t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.arange(4)
joint_dataset = JointDataset(t_x, t_y)

# Or use TensorDataset directly
joint_dataset = TensorDataset(t_x, t_y)

for example in joint_dataset:
    print('  x: ', example[0], 
          '  y: ', example[1])


# ### Shuffle, batch, and repeat



torch.manual_seed(1)
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)

for i, batch in enumerate(data_loader, 1):
        print(f'batch {i}:', 'x:', batch[0], 
              '\n         y:', batch[1])
        
for epoch in range(2):
    print(f'epoch {epoch+1}')
    for i, batch in enumerate(data_loader, 1):
        print(f'batch {i}:', 'x:', batch[0], 
              '\n         y:', batch[1])


# ### Creating a dataset from files on your local storage disk




imgdir_path = pathlib.Path('cat_dog_images')

file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])

print(file_list)






fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_list):
    img = Image.open(file)
    print('Image shape: ', np.array(img).shape)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)
    
#plt.savefig('figures/12_03.pdf')
plt.tight_layout()
plt.show()




labels = [1 if 'dog' in os.path.basename(file) else 0
          for file in file_list]
print(labels)




class ImageDataset(Dataset):
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels

    def __getitem__(self, index):
        file = self.file_list[index]      
        label = self.labels[index]
        return file, label

    def __len__(self):
        return len(self.labels)
    
image_dataset = ImageDataset(file_list, labels)
for file, label in image_dataset:
    print(file, label)





class ImageDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(self.file_list[index])        
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
    def __len__(self):
        return len(self.labels)

img_height, img_width = 80, 120
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width)),
])
    
image_dataset = ImageDataset(file_list, labels, transform)




fig = plt.figure(figsize=(10, 6))
for i, example in enumerate(image_dataset):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example[0].numpy().transpose((1, 2, 0)))
    ax.set_title(f'{example[1]}', size=15)
    
plt.tight_layout()
plt.savefig('figures/12_04.pdf')
plt.show()


# ### Fetching available datasets from the torchvision.datasets library



# ! pip install torchvision






# **Fetching CelebA dataset**
# 
# ---

# 1. Downloading the image files manually
# 
#     - You can try setting `download=True` below. If this results in a `BadZipfile` error, we recommend downloading the `img_align_celeba.zip` file manually from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. In the Google Drive folder, you can find it under the `Img` folder as shown below:



IPythonImage(filename='figures/gdrive-download-location-1.png', width=500)


# - You can also try this direct  link: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ
# - After downloading, please put this file into the `./celeba` subolder and unzip it.

# 2. Next,  you need to download the annotation files and put them into the same `./celeba` subfolder. The annotation files can be found under `Anno`:



IPythonImage(filename='figures/gdrive-download-location-2.png', width=300)


# - direct links are provided below:
#   - [identity_CelebA.txt](https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=sharing)
#   - [list_attr_celeba.txt](https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing&resourcekey=0-YW2qIuRcWHy_1C2VaRGL3Q)
#   - [list_bbox_celeba.txt](https://drive.google.com/file/d/0B7EVK8r0v71pbThiMVRxWXZ4dU0/view?usp=sharing&resourcekey=0-z-17UMo1wt4moRL2lu9D8A)
#   - [list_landmarks_align_celeba.txt](https://drive.google.com/file/d/0B7EVK8r0v71pd0FJY3Blby1HUTQ/view?usp=sharing&resourcekey=0-aFtzLN5nfdhHXpAsgYA8_g)
#   - [list_landmarks_celeba.txt](https://drive.google.com/file/d/0B7EVK8r0v71pTzJIdlJWdHczRlU/view?usp=sharing&resourcekey=0-49BtYuqFDomi-1v0vNVwrQ)



IPythonImage(filename='figures/gdrive-download-location-3.png', width=300)


# 3. Lastly, you need to download the file `list_eval_partition.txt` and place it under `./celeba`:

# - [list_eval_partition.txt](https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view?usp=sharing&resourcekey=0-i4TGCi_51OtQ5K9FSp4EDg)

# After completing steps 1-3 above, please ensure you have the following files in your `./celeba` subfolder, and the files are non-empty (that is, they have similar file sizes as shown below):



IPythonImage(filename='figures/celeba-files.png', width=400)


# ---



image_path = './'
celeba_dataset = torchvision.datasets.CelebA(image_path, split='train', target_type='attr', download=False)

assert isinstance(celeba_dataset, torch.utils.data.Dataset)




example = next(iter(celeba_dataset))
print(example)




fig = plt.figure(figsize=(12, 8))
for i, (image, attributes) in islice(enumerate(celeba_dataset), 18):
    ax = fig.add_subplot(3, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image)
    ax.set_title(f'{attributes[31]}', size=15)
    
#plt.savefig('figures/12_05.pdf')
plt.show()




mnist_dataset = torchvision.datasets.MNIST(image_path, 'train', download=True)

assert isinstance(mnist_dataset, torch.utils.data.Dataset)

example = next(iter(mnist_dataset))
print(example)

fig = plt.figure(figsize=(15, 6))
for i, (image, label) in islice(enumerate(mnist_dataset), 10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image, cmap='gray_r')
    ax.set_title(f'{label}', size=15)

#plt.savefig('figures/12_06.pdf')
plt.show()


# ---
# 
# Readers may ignore the next cell.




