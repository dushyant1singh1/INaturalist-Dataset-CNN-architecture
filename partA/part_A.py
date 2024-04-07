import argparse
import numpy as np
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


parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",default="myprojectname")
parser.add_argument("-we","--wandb_entity",default="myname")
parser.add_argument("-mul","--multiplier",default=2,type=int,help = "this number is important it will multiply the number of filters in each layer")
parser.add_argument("-filters","--filters",default=16,type = int,help = "Number of filter in first conv layer after first layer each layer will have filter = filters*multiplier")
parser.add_argument("-filter_s","--filterSize",default=2,type=int,help="size of filter matrix")
parser.add_argument("-pool","--poolingSize",default=2,type = int,help="size of pooling matrix")
parser.add_argument("-stride","--stride",default=1,type=int , help="stride for pooling and filter matrix")
parser.add_argument("-denseLayers","--denseLayers",default=1,type=int,help="number of dense layers after convolution layers")
parser.add_argument("-denseLSize","--denseLayerSize",default=128,type=int)
parser.add_argument("-aug","--augmentation",default=False,type = bool,choices=[True,False])
parser.add_argument("-batchN","--batchNormalization",default=True,type=bool,choices=[True,False])
parser.add_argument("-drop","--dropout",default=0,type=float)

parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network",
type= int, default=10)
parser.add_argument("--batch_size","-b",help="Batch size used to train neural network"
, type =int, default=16)
parser.add_argument("--optimizer","-o",help="batch size is used to train neural network",
default= "sgd", choices=["sgd","rmsprop","adam","nadam"])

parser.add_argument("--learning_rate","-lr", default=0.0001, type=float)

parser.add_argument("-a","--activation",choices=["selu","selu","gelu","mish","silu"],default="relu")
args = parser.parse_args()

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


data_dir = 'inaturalist_12K/train'
test_data_dir = 'inaturalist_12K/val'
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((512,512)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
val_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((512,512)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
batch_size = args.batch_size
train_dl = DataLoader(dataset, batch_size, shuffle = True,pin_memory=True, num_workers = 2)
val_dl = DataLoader(val_dataset, batch_size,shuffle = True,pin_memory = True, num_workers = 2)
device = get_default_device()
print(device)
train_dl_gpu = DeviceDataLoader(train_dl,device)
val_dl_gpu = DeviceDataLoader(val_dl,device)

height = 512
width = 512
class CNNArchitecture(nn.Module):
    def __init__(self,param,h,w):
        super(CNNArchitecture,self).__init__()
        
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, param.filters[0], param.filterSize)
        if param.batchNormalization==True:
            self.batchnorm1 = nn.BatchNorm2d(param.filters[0])
        self.conv2 = nn.Conv2d(param.filters[0],param.filters[1],param.filterSize)
        if param.batchNormalization==True:
            self.batchnorm2 = nn.BatchNorm2d(param.filters[1])
        self.conv3 = nn.Conv2d(param.filters[1],param.filters[2],param.filterSize)
        if param.batchNormalization==True:
            self.batchnorm3 = nn.BatchNorm2d(param.filters[2])
        self.conv4 = nn.Conv2d(param.filters[2],param.filters[3],param.filterSize)
        if param.batchNormalization==True:
            self.batchnorm4 = nn.BatchNorm2d(param.filters[3])
        self.conv5 = nn.Conv2d(param.filters[3],param.filters[4],param.filterSize)
        if param.batchNormalization==True:
            self.batchnorm5 = nn.BatchNorm2d(param.filters[4])
        
        self.flatten_features =None
        #we need flatten features as an input for first dense layers without this our model will not be compatible
        # we are sending dummy image to our cnn layers and calculating what will be the parameters of it
        self.calculateFeatures(param,torch.rand(1,3,h,w))
        self.linearLayers = nn.ModuleList()

        #TODO: I didn't added activation layers here I have to do this work in forward pass
        if param.denseLayers!=0:
            self.linearLayers.append(nn.Linear(self.flatten_features,param.denseLayersSize))
            for _ in range(param.denseLayers-1):
                if int(param.dropout)!=0:
                    self.linearLayers.append(nn.Dropout(param.dropout))
                self.linearLayers.append(nn.Linear(param.denseLayersSize,param.denseLayersSize))
                
            self.linearLayers.append(nn.Linear(param.denseLayersSize,10))
        else:
            self.linearLayers.append(nn.Linear(param.flatten_features,10))

    def calculateFeatures(self,param,x):
        z = param.poolingSize
        activation = param.activation
        print(z)
        x = F.max_pool2d(activation(self.conv1(x)), z)
        print(x.size())
        x = F.max_pool2d(activation(self.conv2(x)), z)
        print(x.size())
        x = F.max_pool2d(activation(self.conv3(x)),z)
        print(x.size())
        x = F.max_pool2d(activation(self.conv4(x)),z)
        print(x.size())
        x = F.max_pool2d(activation(self.conv5(x)),z)
        print(x.size())
        self.flatten_features = x.size(1) * x.size(2) * x.size(3)
        
        
        
    def forward(self,param, x):
        z = param.poolingSize
        activation = param.activation
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(activation(self.conv1(x)), z)
        if param.batchNormalization==True:
            x = self.batchnorm1(x)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(activation(self.conv2(x)), z)
        if param.batchNormalization == True:
            x = self.batchnorm2(x)
        x = F.max_pool2d(activation(self.conv3(x)),z)
        if param.batchNormalization == True:
            x = self.batchnorm3(x)
        x = F.max_pool2d(activation(self.conv4(x)),z)
        if param.batchNormalization == True:
            x = self.batchnorm4(x)
        x = F.max_pool2d(activation(self.conv5(x)),z)
        if param.batchNormalization == True:
            x = self.batchnorm5(x)
        
        
        
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension 
        for i in range(len(self.linearLayers)-1):
            x = activation(self.linearLayers[i](x))
        x = self.linearLayers[-1](x)
        return x

class Parameters:
    def __init__(self,filters,filter_size,pooling_size,stride,multiplier,dense_layers,dense_layer_size,aug,normalization,dropout,activation,optimizers,lr):
        self.cnnLayers = 5
        self.filterMultiplier = multiplier# number float
        self.filters = self.settingFilters(filters,multiplier,self.cnnLayers)
        self.filterSize = filter_size
        self.poolingSize = pooling_size
        self.stride = stride
        self.denseLayers = dense_layers
        self.denseLayersSize = dense_layer_size
        self.dataAugmentation = aug
        self.batchNormalization = normalization# true or false
        self.dropout = dropout # probabilty
        self.activation_dict = { 'relu':F.relu,'selu':F.selu,'gelu':F.gelu,'mish':F.mish,'silu':F.silu}
        self.optimzers_dict = {'sgd':optim.SGD,'nadam':optim.NAdam,"adam":optim.Adam,"rmsprop":optim.RMSprop}
        self.activation = self.activation_dict[activation]
        self.optimizer = self.optimzers_dict[optimizers]
        self.learning_rate = lr
        
    def settingFilters(self,filters,multiplier,layers):
        return [filters*(multiplier**i) for i in range(layers)]

def evaluate(model, ob ,dataset_tensor, datatype ,use_cuda = True):
    model.eval()
    correct = 0
    total = 0
    total_loss  = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data in dataset_tensor:
            images, labels = data
           # print(images.device)
            outputs = model.forward(ob,images)
#             loss+=F.cross_entropy(outputs,labels)
            loss= criterion(outputs,labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss.append(loss)
    loss = torch.stack(total_loss).mean().item()
    acc = (100*correct/total)
    print(f'{datatype}_accuracy: {acc}, {datatype}_loss: {loss}')
#     wb.log({f'{datatype}_accuracy': acc})
#     wb.log({f'{datatype}_loss': loss})

    
# this function will also work without gpu
def fit(ob,model,train_gpu,val_gpu,epochs):
    optimizer = ob.optimizer(model.parameters(),lr = ob.learning_rate)
    history =[]
    for i in range(epochs):
        model.train()
#         training_loss = []
        acc =[]
        for ind, (images, labels) in enumerate(tqdm(train_gpu, desc=f'Training Progress {i+1}')):
            optimizer.zero_grad()
            pred = model.forward(ob,images)
        
            loss = F.cross_entropy(pred, labels)
#             training_loss.append(loss)
            loss.backward()
            optimizer.step()
        training_acc = evaluate(model,ob,train_gpu,'training')
        validation_acc = evaluate(model,ob,val_gpu,'validation')
    return model


# print(args.multiplier)
# print(args.filters)
# print(args.filterSize)
# print(args.poolingSize)
# print(args.stride)
# print(args.denseLayers)
# print(args.denseLayerSize)
# print(args.augmentation)
# print(args.batchNormalization)
# print(args.dropout)
# print(args.epochs)
# print(args.batch_size)
# print(args.optimizer)
# print(args.learning_rate)
# print(args.activation)

ob = Parameters(args.filters,args.filterSize,args.poolingSize,args.stride,args.multiplier,args.denseLayers,args.denseLayerSize,args.augmentation,args.batchNormalization,args.dropout,args.activation,args.optimizer,args.learning_rate)
model = CNNArchitecture(ob,height,width)
if torch.cuda.is_available():
        model.to(device)
model = fit(ob,model,train_dl_gpu,val_dl_gpu,args.epochs)
print("Model is trained successfully")