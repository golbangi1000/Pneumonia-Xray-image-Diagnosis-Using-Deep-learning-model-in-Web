#%%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from pytictoc import TicToc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pydicom import dcmread
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

t = TicToc()
t.tic()

# load csv file (label)
label_data = pd.read_csv('./stage_1_train_labels.csv')
columns = ['patientId', 'Target']

label_data = label_data.filter(columns)

# train label, validation label split
train_labels, val_labels = train_test_split(label_data.values, test_size=0.1)

#%%

# train image directory path
train_f = './stage_1_train_images'

# train image path and validation image path List
train_paths = [os.path.join(train_f, image[0]) for image in train_labels]
val_paths = [os.path.join(train_f, image[0]) for image in val_labels]

# check train set size and validation set size
print("train set:", len(train_paths))
print("validationw set:", len(val_paths))

#%%

# data preprocessing pipe line 
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor()])

# data load and data preprocessing function 
class load_tans_image(data.Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = dcmread(f'{self.paths[index]}.dcm')
        img = img.pixel_array
        img = img / 255.0
        img = (img*255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img).convert('RGB')
        label = self.labels[index][1]
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.paths)

# start data preprocessing
train_dataset = load_tans_image(train_paths, train_labels, transform=transform)
val_dataset = load_tans_image(val_paths, val_labels, transform=transform)

# make dataloader 
train_loader = data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=256, shuffle=False)

#%%

# CPU or GPU selection 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

# loss and optimizer define
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# start model training
loss_list = []
val_score_list = []
num_epochs = 20
total_step = len(train_loader)
for epoch in range(num_epochs):
    total_loss=0
    for i, (images, labels) in tqdm(enumerate(train_loader)):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i+1) % 2000 == 0:
            print("Epoch: {}/{}, Step: {}/{}, train Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    correct = 0
    total = 0  

    # print model validation score
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        _, predicted = torch.max(predictions, 1)
        total += labels.size(0)
        correct += (labels == predicted).sum()
    loss_list.append(total_loss/len(train_loader))
    val_score_list.append({correct * 100/total})
    print(f'Epoch: {epoch+1}/{num_epochs}, Val_Acc: {correct * 100/total}')
t.toc() # end time

#%%
# model loss plotting 
import matplotlib.pyplot as plt
alist = [i+1 for i in range(num_epochs)]
plt.plot(alist,loss_list)
plt.ylim(0,1)

#%%
# model validation score plotting 
blist= [] 
for i in val_score_list:
    blist.append(list(i)[0].detach().cpu().numpy())
alist = [i+1 for i in range(num_epochs)]
plt.plot(alist, blist)
plt.ylim(0,100)

#%%
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

PATH = './weights/'

# save model
torch.save(model, PATH + 'resmodel_0413.pt')   

# save model state_dict
torch.save(model.state_dict(), PATH + 'resmodel_0413_state_dict.pt')  

# save model state_dict and optimizer
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, PATH + 'all.tar')  
#%%
# load model
model = torch.load(PATH + 'model.pt')  
# load model state dict
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  

# load model whole data 
checkpoint = torch.load(PATH + 'all.tar')   
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

#%%
# check model validation 
model.eval()
correct = 0
total = 0  
for images, labels in tqdm(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    predictions = model(images)
    _, predicted = torch.max(predictions, 1)
    total += labels.size(0)
    correct += (labels == predicted).sum()
print(f'Val_Acc: {100*correct/total}')