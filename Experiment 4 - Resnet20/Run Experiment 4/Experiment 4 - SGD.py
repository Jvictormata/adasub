import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.optim import Optimizer
import time
from resnet import resnet20
import os
import math


num_epochs = 30
batch_size = 400
learning_rate = 0.1

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])



train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def evaluate():
    evaluation = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
        evaluation.append(acc)
        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
            evaluation.append(acc)
    return evaluation
    


def save_progress():
	evaluation = np.array(acc)
	with open('evaluation_SGD010LR.npy', 'wb') as f:
		np.save(f, evaluation)

	losses_arr = np.array(losses)
	with open('losses_SGD010LR.npy', 'wb') as f:
		np.save(f, losses_arr)
		
	torch.save(model.state_dict(), 'model_SGD010LR.pt')


device = torch.device("cuda")
print(device)

def force_cudnn_initialization():
    s = 32
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=device), torch.zeros(s, s, s, s, device=device))
force_cudnn_initialization()

model = resnet20()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)
iteration = 0
losses = []
acc = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
       
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        if iteration == 0:
            print(f'Initial loss = {loss.item():.4f}\n\n')

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(create_graph=True)
        optimizer.step()

        losses.append([iteration,loss.item()])
        iteration += 1

        if (i+1) % 1 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    acc.append(evaluate())
    save_progress()

