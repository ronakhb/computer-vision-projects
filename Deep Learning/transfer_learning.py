'''
transfer_learning.py
Project 5

Created by Ronak Bhanushali and Rucha Pendharkar on 4/1/24

This file contains code used for task 3.

'''

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from train import train_network,visualizeData,test
from models.custom_convolution import mnistNet
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from inference import *

#Defining the transforms needed for Greek letters
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

#main function

def inferCustomGreek():
    train_transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                GreekTransform(),
                                                torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,) ) ] )
    test_dataset = torchvision.datasets.ImageFolder(root='datasets/greek_test', transform=train_transform)
    test_loader = DataLoader(test_dataset, batch_size=15, shuffle=True)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    model = mnistNet()
    input_features = model.fcnn2.in_features
    model.fcnn2 = nn.Linear(input_features,3)
    checkpoint_path = "checkpoints/transfer_learning_checkpoints/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    
    performInference(model,example_data,example_targets)
    test(model,test_loader,[])
    

def trainGreek():
    train_transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                GreekTransform(),
                                                torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,) ) ] )

    full_dataset = torchvision.datasets.ImageFolder(root='datasets/greek_train/greek_train', transform=train_transform)

    # Define the sizes of train and test sets
    train_size = int(0.8 * len(full_dataset))  # 80% of the dataset for training
    test_size = len(full_dataset) - train_size  # Remaining 20% for testing

    # Split the dataset into train and test sets
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoader objects for train and test sets
    train_loader = DataLoader(full_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    visualizeData(train_loader)
    model = mnistNet()
    checkpoint_path = "checkpoints/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    
    
    input_features = model.fcnn2.in_features
    model.fcnn2 = nn.Linear(input_features,3)

    #Defining hyper parameters
    epochs = 40
    learning_rate = 0.01
    log_interval = 10
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(epochs+1)]

    test(model,test_loader,test_losses)
    epoch_100_accuracy = train_network(epochs,model,optimizer,train_loader,test_loader,train_losses,train_counter,log_interval,test_losses,"checkpoints/transfer_learning_checkpoints")
    print(epoch_100_accuracy)
    
    validation_dataset = torchvision.datasets.ImageFolder(root='datasets/greek_test', transform=train_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=28, shuffle=False)
    
    checkpoint_path = "checkpoints/transfer_learning_checkpoints/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path)
    print(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(model,validation_loader,[])
    fig = plt.figure()
    plt.scatter(test_counter, test_losses, color='cyan')
    plt.plot(train_counter, train_losses, color=(159/255, 2/255, 81/255))
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.show()

def trainGreekCustom():
    train_transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                GreekTransform(),
                                                torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,) ) ] )

    full_dataset = torchvision.datasets.ImageFolder(root='datasets/greek_train_custom/', transform=train_transform)

    # Define the sizes of train and test sets
    train_size = int(0.8 * len(full_dataset))  # 80% of the dataset for training
    test_size = len(full_dataset) - train_size  # Remaining 20% for testing

    # Split the dataset into train and test sets
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoader objects for train and test sets
    train_loader = DataLoader(full_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    visualizeData(train_loader)
    model = mnistNet()
    checkpoint_path = "checkpoints/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    
    
    input_features = model.fcnn2.in_features
    model.fcnn2 = nn.Linear(input_features,6)

    #Defining hyper parameters
    epochs = 20
    learning_rate = 0.01
    log_interval = 10
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(epochs+1)]

    test(model,test_loader,test_losses)
    epoch_100_accuracy = train_network(epochs,model,optimizer,train_loader,test_loader,train_losses,train_counter,log_interval,test_losses,"checkpoints/transfer_learning_custom")
    print(epoch_100_accuracy)
    
    validation_dataset = torchvision.datasets.ImageFolder(root='datasets/greek_test_custom', transform=train_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=28, shuffle=False)
    
    checkpoint_path = "checkpoints/transfer_learning_custom/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path)
    print(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(model,validation_loader,[])
    fig = plt.figure()
    plt.scatter(test_counter, test_losses, color='cyan')
    plt.plot(train_counter, train_losses, color=(159/255, 2/255, 81/255))
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.show()

def main():
    
    trainGreek()
    inferCustomGreek()
    trainGreekCustom()
    # inferCustomGreek()
    
if __name__ == "__main__":
    main()