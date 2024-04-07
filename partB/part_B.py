import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import math
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import wandb as wb
import gc
from tqdm import tqdm
import time
from torchvision.transforms import ToTensor, Resize
from torchvision import models
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",default="myprojectname")
parser.add_argument("-we","--wandb_entity",default="myname")
parser.add_argument("-sc","--scheme",default="freeze_all", help="type of freezing you want, if you choose freeze_k then enter value of k",choices=["freeze_all","freeze_k"])
parser.add_argument("-k","--numberOfLayers",default=1,type=int,help="how many layers you want to freeze ")
parser.add_argument("-batchN","--batchNormalization",default=True,type=bool,choices=[True,False])
parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network",
type= int, default=10)
parser.add_argument("--batch_size","-b",help="Batch size used to train neural network"
, type =int, default=16)
parser.add_argument("--optimizer","-o",help="batch size is used to train neural network",
default= "sgd", choices=["sgd","rmsprop","adam","nadam"])

parser.add_argument("--learning_rate","-lr", default=0.0001, type=float)

args = parser.parse_args()



def freeze_all_layers_except_last(model, num_classes):
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def freeze_upto_k_layers(model, num_classes, k):

    total_layers = sum(1 for _ in model.parameters())
    layers_to_freeze = k
    for i, param in enumerate(model.parameters()):
        if i < layers_to_freeze:
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_default_device():
    """ Set Device to GPU or CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    "Move data to the device"
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)

class DeviceDataLoader():
    """ Wrap a dataloader to move data to a device """
    
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        """ Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
            
    def __len__(self):
        """ Number of batches """
        return len(self.dl)

def evaluate(model ,dataset_tensor, datatype ,use_cuda = True):
    model.eval()
    correct = 0
    total = 0
    total_loss  = []
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, labels in dataset_tensor:
            images, label = inputs.to(device) , labels.to(device)
           # print(images.device)
            outputs = model(images)
#             loss+=F.cross_entropy(outputs,labels)
            loss= criterion(outputs,label)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == label).sum().item()
            total_loss.append(loss)
    loss = torch.stack(total_loss).mean().item()
    acc = (100*correct/total)
    print(f'{datatype}_accuracy: {acc}, {datatype}_loss: {loss}')
#     wb.log({f'{datatype}_accuracy': acc})
#     wb.log({f'{datatype}_loss': loss})

def train(model, train_dl, val_dl , criterion, optimizer, num_epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    opti = optimizer(model.parameters(),lr = 0.001)
    for epoch in range(num_epochs):
        for ind, (inputs, labels) in enumerate(tqdm(train_dl, desc=f'Training Progress {epoch+1}')):
            images, label = inputs.to(device), labels.to(device)

            opti.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, label)

            loss.backward()
            opti.step()
        training_acc = evaluate(model,train_dl,'training')
        validation_acc = evaluate(model,val_dl,'validation')




# Load custom dataset

data_dir = 'inaturalist_12K/train'
test_data_dir = 'inaturalist_12K/val'
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
val_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
batch_size = args.batch_size
train_dl = DataLoader(dataset, batch_size, shuffle = True)
val_dl = DataLoader(val_dataset, batch_size,shuffle = True)
#device = get_default_device()
criterion = nn.CrossEntropyLoss()
pretrained_model = models.resnet50(pretrained=True)
if args.scheme=="freeze_all":
    pretrained_model = freeze_all_layers_except_last(pretrained_model,10)
elif args.scheme=="freeze_k":
    pretrained_model= freeze_upto_k_layers(pretrained_model,10,args.numberOfLayers)
optimizers_dict = {'sgd':optim.SGD,'nadam':optim.NAdam,"adam":optim.Adam,"rmsprop":optim.RMSprop}
optimizer = args.optimizer
train(pretrained_model, train_dl, val_dl, criterion, optimizers_dict[optimizer], args.epochs)