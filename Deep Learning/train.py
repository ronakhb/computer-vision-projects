'''
train.py
Project 5

Created by Ronak Bhanushali and Rucha Pendharkar on 4/1/24

This file contains the code for training the network

'''
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.custom_convolution import *
from tqdm import tqdm

# Define function to train model
def train_network(epochs, model, optimizer, train_loader, test_loader, train_losses, train_counter, log_interval, test_losses,checkpoint_root_path = "checkpoints"):
    model = model.cuda()
    model.train()
    min_loss = 100000
    epoch_100_accuracy = []
    for epoch in tqdm(range(epochs),desc="Epochs",leave=False,position = 4):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.cuda()).cpu()
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            if batch_idx % log_interval == 0:
                train_losses.append(loss.item())
                train_counter.append((batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Train Epoch: {epoch+1}/{epochs}\tAverage Loss: {avg_epoch_loss:.6f}')

        accuracy = test(model, test_loader, test_losses)
        if accuracy:
            epoch_100_accuracy.append(epoch)
        checkpoint_path = checkpoint_root_path + "/checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        if loss<min_loss:
            min_loss = loss
            checkpoint_path = checkpoint_root_path + "/checkpoint_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
    return epoch_100_accuracy

# Define function to test the model
def test(model, test_loader, test_losses,accuracy_mode = 0):
    model = model.cuda()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.cuda()).cpu()
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    if(accuracy_mode==1):
        return accuracy
    else:
        if accuracy == 100:
            return True
        else:
            return False

#Function to visualize data
def visualizeData(data_loader):
    # Visualize example data
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def trainMNIST():
    # Define hyperparameters
    epochs = 10
    batch_size_train = 64
    batch_size_test = 128
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
    train_dataset = torchvision.datasets.MNIST('data/', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('data/', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,num_workers = 10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    visualizeData(test_loader)

    # Initialize model, optimizer, and lists to store losses
    model = mnistNet()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]
    
    # Train and test the model
    test(model, test_loader, test_losses)
    train_network(epochs, model, optimizer, train_loader, test_loader, train_losses, train_counter, log_interval, test_losses,checkpoint_root_path="checkpoints/")

     # Plotting training and test losses
    fig = plt.figure()
    plt.scatter(test_counter, test_losses, color='cyan')
    plt.plot(train_counter, train_losses, color=(159/255, 2/255, 81/255))
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.show()
    
def trainMNISTFashionMobilenet():
    # Define hyperparameters
    epochs = 20
    batch_size_train = 64
    batch_size_test = 128
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    # Set random seed
    random_seed = 3
    # torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Define train and test transformations
    expand_to_three_channels = lambda x: x.expand(3, -1, -1)

    transform_mobilenet = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(expand_to_three_channels),  # Expand single channel to three channels
        torchvision.transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))  # Normalize for three channels
    ])

    # Load train and test datasets
    train_dataset = torchvision.datasets.FashionMNIST('data/', train=True, download=True, transform=transform_mobilenet)
    test_dataset = torchvision.datasets.FashionMNIST('data/', train=False, download=True, transform=transform_mobilenet)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,num_workers = 10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    visualizeData(test_loader)

    # Initialize model, optimizer, and lists to store losses
    model = MobileNetMNIST()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]

    # Train and test the model
    test(model, test_loader, test_losses)
    train_network(epochs, model, optimizer, train_loader, test_loader, train_losses, train_counter, log_interval, test_losses,checkpoint_root_path="checkpoints/mobilenet/")

    # Plotting training and test losses
    fig = plt.figure()
    plt.scatter(test_counter, test_losses, color='cyan')
    plt.plot(train_counter, train_losses, color=(159/255, 2/255, 81/255))
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.show()
#main function 
def main():
    trainMNIST()
    # trainMNISTFashionMobilenet()

if __name__ == "__main__":
    main()