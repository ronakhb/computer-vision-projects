'''
custom_convolution.py
Project 5

Created by Ronak Bhanushali and Rucha Pendharkar on 4/1/24

This file contains the model definitions used across the project and extensions

'''
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

#Defining the model
class mnistNet(nn.Module):
    def __init__(self):
        super(mnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5,padding=2)
        self.conv2 = nn.Conv2d(10,20,5,padding=2)
        self.fcnn1 = nn.Linear(980,50)
        self.pool = nn.MaxPool2d(2)
        self.fcnn2 = nn.Linear(50,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.dropout(x,p = 0.5)
        x = self.pool(x)
        x = F.relu(x)
        x = torch.flatten(x,1)        
        x = self.fcnn1(x)
        x = F.relu(x)
        x = self.fcnn2(x)
        x = F.log_softmax(x,dim=1)
        
        return x

#Model for task4    
class mnistNet2(nn.Module):
    def __init__(self):
        super(mnistNet2, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5,padding=2)
        self.conv2 = nn.Conv2d(10,20,5,padding=2)
        self.conv3 = nn.Conv2d(20,30,5,padding=2)
        self.fcnn1 = nn.Linear(270,50)
        self.fcnn2 = nn.Linear(50,10)
        self.pool = nn.MaxPool2d(2)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.dropout(x,p = 0.5)
        x = self.pool(x)
        x = F.relu(x)
        x = torch.flatten(x,1)
        x = self.fcnn1(x)
        x = F.relu(x)
        x = self.fcnn2(x)
        x = F.log_softmax(x,dim=1)
        
        return x
    
#Model for task4
class mnistNet3(nn.Module):
    def __init__(self):
        super(mnistNet3, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5,padding=2)
        self.conv2 = nn.Conv2d(10,20,5,padding=2)
        self.fcnn1 = nn.Linear(320,50)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(3)
        self.fcnn2 = nn.Linear(50,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.dropout(x,p = 0.5)
        x = self.pool(x)
        x = F.relu(x)
        x = torch.flatten(x,1)
        x = self.fcnn1(x)
        x = F.relu(x)
        x = self.fcnn2(x)
        x = F.log_softmax(x,dim=1)
        
        return x
    
#Extension - using MobileNetV3
class MobileNetMNIST(nn.Module):
    def __init__(self):
        super(MobileNetMNIST, self).__init__()
    
        self.backbone = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT')
        print('Loading pretrained MobileNet model......')
        in_features = self.backbone.classifier[0].in_features
        self.linear_layer = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(in_features, 10))
        self.batch_norm = nn.BatchNorm1d(in_features,track_running_stats=True)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.squeeze(x,dim=2)
        x = torch.squeeze(x,dim=2)
        x = self.batch_norm(x)
        x = self.linear_layer(x)
        x = F.log_softmax(x,dim=1)
        return x

#main function
def main():
    model = mnistNet()
    print(model)


if __name__ == "__main__":
    main()