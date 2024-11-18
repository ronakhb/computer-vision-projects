'''
plot.py
Project 5

Created by Ronak Bhanushali and Rucha Pendharkar on 4/1/24

This file contains the code for plotting task4 

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

#Defining the function
def plot_data(data):
    num_networks = data.shape[0]
    num_filters = data.shape[1]
    num_kernels = data.shape[2]
    
    # Plotting accuracy vs. number of filters for each kernel size
    for kernel_idx in range(num_kernels):
        plt.figure(figsize=(10, 6))
        plt.title(f'Accuracy vs. Number of Filters (Kernel Size: {kernel_idx+2})')
        plt.xlabel('Number of Filters')
        plt.ylabel('Accuracy')
        
        for network_idx in range(num_networks):
            accuracy = data[network_idx, :, kernel_idx, 0]
            plt.plot(range(1, num_filters + 1), accuracy, label=f'Network {network_idx}')
        
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_filters + 1))
        plt.tight_layout()
    plt.show()
    
    # Plotting accuracy vs. kernel size for each number of filters
    for filter_idx in range(num_filters):
        plt.figure(figsize=(10, 6))
        plt.title(f'Accuracy vs. Kernel Size (Number of Filters: {filter_idx+1})')
        plt.xlabel('Kernel Size')
        plt.ylabel('Accuracy')
        
        for network_idx in range(num_networks):
            accuracy = data[network_idx, filter_idx, :, 0]
            plt.plot(range(2, num_kernels + 2), accuracy, label=f'Network {network_idx}')
        
        plt.legend()
        plt.grid(True)
        plt.xticks(range(2, num_kernels + 2))
        plt.tight_layout()
    plt.show()

#main function
def main():
    file_path = "data/data.npy"
    # Load data from the file
    data = np.load(file_path)

    max_position = np.unravel_index(np.argmax(data[:,:,:,0]), data[:,:,:,0].shape)
    print("Position of max accuracy value:", max_position)
    print(np.max(data[:,:,:,0]))        
    plot_data(data)


if __name__ == "__main__":
    main()