# INaturalist-Dataset-CNN-architecture
In this repository we are trying to create a cnn architecture using pytorch library
### Instruction to run the code -

- First you have to download the dataset from https://storage.googleapis.com/wandb_datasets/nature_12K.zip
- This dataset should be in the same directory as python file or you can change the data paths in the part_A.py and part_B.py files
***PART A***
Parameters that we can pass with their default values and type of part_A.py are:
```python
"-mul","--multiplier",default=2,type=int,help = "this number is important it will multiply the number of filters in each layer"
"-filters","--filters",default=16,type = int,help = "Number of filter in first conv layer after first layer each layer will have filter = filters*multiplier"
"-filter_s","--filterSize",default=2,type=int,help="size of filter matrix"
"-pool","--poolingSize",default=2,type = int,help="size of pooling matrix"
"-stride","--stride",default=1,type=int , help="stride for pooling and filter matrix"
"-denseLayers","--denseLayers",default=1,type=int,help="number of dense layers after convolution layers"
"-denseLSize","--denseLayerSize",default=128,type=int
"-aug","--augmentation",default=False,type = bool,choices=[True,False]
"-batchN","--batchNormalization",default=True,type=bool,choices=[True,False]
"-drop","--dropout",default=0,type=floatwand
"--epochs","-e", help= "Number of epochs to train neural network",
type= int, default=10
"--batch_size","-b",help="Batch size used to train neural network", type =int, default=16
"--optimizer","-o",help="batch size is used to train neural network",default= "sgd", choices=["sgd","rmsprop","adam","nadam"]
"--learning_rate","-lr", default=0.0001, type=float
"-a","--activation",choices=["selu","selu","gelu","mish","silu"],default="relu"
```
- CNN architecture is in the class named - ``` CNNArchitecture ```  you can change the number of convolution layers by adding more layers in the class. But don't forget to add these extra layers in ```calculateFeatures``` and ```forward``` functions. ```calculateFeatures``` it calculates the features after all the convolution and maxpooling layers and `forward``` function is used to calculate the forward pass.
- All the parameters that are passed on run times are further forwarded to ```class Parameters```
- Requirement to run this files are 
```
 torch, numpy, matplotlib, torchvision, tqdm, wandb
```
- to run the file type ```python part_A.py```

***Part B***
to run this file first download the dataset
parameters we can pass are-
```python
"-sc","--scheme",default="freeze_all", help="type of freezing you want, if you choose freeze_k then enter value of k", choices=["freeze_all", "freeze_k"]
"-k", "--numberOfLayers", default=1, type=int, help="how many layers you want to freeze "
```
to run the code type ``` python part_B.py ``` 
