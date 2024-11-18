'''
task4.py
Project 5

Created by Ronak Bhanushali and Rucha Pendharkar on 4/1/24

This file contains the code for task4 and one extension.  

'''
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from train import train_network,visualizeData,test
from models.custom_convolution import *
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

#Helper function for task4 
def generate_network(id):
    if id == 0:
        return mnistNet()
    elif id == 1:
        return mnistNet2()
    elif id == 2:
        return mnistNet3()
    else:
        return mnistNet()

#Function for training different models on MNIST Fashion Data
    
def trainFashion():
    
    # Define hyperparameters
    epochs = 10
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    # Set random seed
    random_seed = 3
    # torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Define train and test transformations
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28,28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load train and test datasets
    train_dataset = torchvision.datasets.FashionMNIST('data/', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST('data/', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,num_workers = 10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    visualizeData(test_loader)

    data = np.zeros((3,5,5,2))
    # Initialize model, optimizer, and lists to store losses
    for network in tqdm([0,1,2],desc="Model",position=0):
        for filter_idx,num_filters in tqdm(enumerate([5,10,20,50,90]),desc="Number of Filters",leave=False,position=1):
            for kernel_idx,kernel_size in tqdm(enumerate([2,3,4,5,6]),desc="Kernel",leave=False,position=2):
                model = generate_network(network)
                model.conv1.out_channels = num_filters
                model.conv2.in_channels = num_filters
                model.conv2.kernel_size = kernel_size
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
                train_losses = []
                train_counter = []
                test_losses = []
                epoch_100 = train_network(epochs, model, optimizer, train_loader, test_loader, train_losses, train_counter, log_interval, test_losses,checkpoint_root_path = "checkpoints/experiment/")
                accuracy = test(model, test_loader, test_losses,accuracy_mode = 1)
                data[network,filter_idx,kernel_idx,0] = accuracy
                if len(epoch_100)!=0:
                    data[network,filter_idx,kernel_idx,1] = epoch_100[0]
                else:
                    data[network,filter_idx,kernel_idx,1] = 1000
                np.save("data/data.npy", data)

#Extension 
                
def trainFashionExtension(epochs):
    
    # Define hyperparameters 
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    # Set random seed
    random_seed = 3
    # torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Define train and test transformations
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28,28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load train and test datasets
    train_dataset = torchvision.datasets.FashionMNIST('data/', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST('data/', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,num_workers = 10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    visualizeData(test_loader)

    data = np.zeros((3,5,5,2))
    # Initialize model, optimizer, and lists to store losses
    for network in tqdm([0],desc="Model",position=0):
        for filter_idx,num_filters in tqdm(enumerate([90]),desc="Number of Filters",leave=False,position=1):
            for kernel_idx,kernel_size in tqdm(enumerate([3]),desc="Kernel",leave=False,position=2):
                model = generate_network(network)
                model.conv1.out_channels = num_filters
                model.conv2.in_channels = num_filters
                model.conv2.kernel_size = kernel_size
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
                train_losses = []
                train_counter = []
                test_losses = []
                epoch_100 = train_network(epochs, model, optimizer, train_loader, test_loader, train_losses, train_counter, log_interval, test_losses,checkpoint_root_path = "checkpoints/experiment/")
                accuracy = test(model, test_loader, test_losses,accuracy_mode = 1)
                data[network,filter_idx,kernel_idx,0] = accuracy
                if len(epoch_100)!=0:
                    data[network,filter_idx,kernel_idx,1] = epoch_100[0]
                else:
                    data[network,filter_idx,kernel_idx,1] = 1000
                np.save("data/data_extension_epoch50.npy", data)

#Visualize the extension
def visualizeResultsExtension():
    # Load the data
    data5 = np.load('data/data_extension_epoch5.npy')
    data20 = np.load('data/data_extension_epoch20.npy')
    data40 = np.load('data/data_extension_epoch40.npy')
    data50 = np.load('data/data_extension_epoch50.npy')

    # Extract the accuracies
    data5 = data5[0, 0, 0, 0]
    data20 = data20[0, 0, 0, 0]
    data40 = data40[0, 0, 0, 0]
    data50 = data50[0, 0, 0, 0]

    # Plot the points
    epochs = [5, 20, 40, 50]
    data_values = [data5, data20, data40, data50]

    plt.plot(epochs, data_values, '-o') 
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy at Different Epochs for Kernel Size = 3 and Filters = 90')
    plt.grid(True)
    plt.show()

#main function 
def main():

    #Pass number of epochs as argument. 
    trainFashionExtension(epochs = 50)
    visualizeResultsExtension()
    
if __name__ == "__main__":
    main()