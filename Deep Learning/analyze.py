'''
analyze.py
Project 5

Created by Ronak Bhanushali and Rucha Pendharkar on 4/1/24

This file contains code used for task 2.

'''

import torch
from models.custom_convolution import mnistNet
import matplotlib.pyplot as plt
import cv2
import torchvision
import numpy as np
from PIL import Image

#Analyze the weights of the first convolutional layer of the provided model and visualize them.
def analyze_weights(model):
    # Get the weights of the first convolutional layer
    first_layer_weights = model.conv1.weight.detach().cpu()

    # Print size and weights of the first layer
    print("Size of the tensor:", first_layer_weights.size())
    print("Weights of Model Layer 1:\n", first_layer_weights)

    # Visualize the first layer weights
    num_filters = first_layer_weights.size(0)
    fig = plt.figure(figsize=(10, 8))
    for i in range(num_filters):
        plt.subplot(3, 4, i+1)
        plt.imshow(first_layer_weights[i, 0].numpy(), cmap='viridis')
        plt.title('Filter {}'.format(i))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

#Apply filters from the first convolutional layer of the provided model to the input image.
def apply_filters(model, image):

    # Extract the weights of the first convolutional layer
    with torch.no_grad():
        first_layer_weights = model.conv1.weight.detach().cpu()

    # Apply each filter to the input image using filter2D and visualize the filtered images
    fig = plt.figure(figsize=(10, 8))
    for i in range(first_layer_weights.size(0)):
        filter_weights = first_layer_weights[i, 0].numpy() # Convert filter weights to NumPy array
        filtered_image = cv2.filter2D(image, -1, filter_weights)  # Apply filter using filter2D

        plt.subplot(3, 4, i+1)
        plt.imshow(filtered_image, cmap='gray')
        #plt.imshow(first_layer_weights[i, 0].numpy(), cmap='gray')

        plt.title('Filter {}'.format(i))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

#main function  
def main():
    # Define the model
    model = mnistNet()

    # Load the state dict
    checkpoint_path = "checkpoints/checkpoint_best.pth"
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print("Checkpoint file not found. Please make sure the path is correct.")

    # Analyze the weights of the model
    analyze_weights(model)

    # Get the first training example image
    dataset = torchvision.datasets.MNIST('data/', train=False, download=True)
    img = cv2.imread("/home/rucha/CS5330-Project5/datasets/Task7/7.jpg") #Replace with appropriate filepath name
    first_image = 255 - cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('First Image from MNIST Dataset', first_image)
    
    # Apply filters to the first image
    apply_filters(model, first_image)

if __name__ == "__main__":
    main()