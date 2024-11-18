'''
inference.py
Project 5

Created by Ronak Bhanushali and Rucha Pendharkar on 4/1/24

This file contains code used for task 1.

'''

import torch
import torchvision
import matplotlib.pyplot as plt
from models.custom_convolution import mnistNet
import os
from train import test
from PIL import Image, ImageOps
import numpy as np

def performInference(model,data,targets):
    # Plotting example predictions
    model.eval()
    with torch.no_grad():
        output = model(data)

    torch.set_printoptions(precision=2)
    torch.set_printoptions(sci_mode=False)
    
    for i in range(10):
        print(output[i])
        print("Best Match: ", torch.argmax(output[i]))
        print("Ground Truth: ", targets[i])

    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i+1][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.argmax(dim=1)[i+1].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def loadCustomDataset(transofrm,path="datasets/Task7"):
    images = sorted(os.listdir(path))
    input_images = []
    targets = []
    for image in images:
        target = int(image.split(".")[0])
        targets.append(target)
        input_image = Image.open(os.path.join(path,image))
        input_image = ImageOps.exif_transpose(input_image)
        input_image = ImageOps.invert(input_image)
        binary_image = input_image.convert("L")
        image_tensor = transofrm(binary_image)
        image_tensor = torch.unsqueeze(image_tensor,dim=0)
        input_images.append(image_tensor)
    input_tensor = torch.cat(input_images)
    return input_tensor,targets

def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28,28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST('data/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    model = mnistNet()
    checkpoint_path = "checkpoints/checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    
    performInference(model,example_data,example_targets)
    custom_data,targets = loadCustomDataset(transform)
    performInference(model,custom_data,targets)
    test(model,test_loader,[])
    
    
if __name__ == "__main__":
    main()