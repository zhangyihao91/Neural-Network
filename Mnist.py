import cv2
import os
import numpy as np
from sklearn import neighbors
import struct
import matplotlib.pyplot as plt
import gzip
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# def load_mnist(path, kind='train'):
   
#     labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
#     images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
#     with open(labels_path, 'rb') as lbpath:
#         magic, n = struct.unpack('>II',
#                                  lbpath.read(8))
#         labels = np.fromfile(lbpath,
#                              dtype=np.uint8)

#     with open(images_path, 'rb') as imgpath:
#         magic, num, rows, cols = struct.unpack('>IIII',
#                                                imgpath.read(16))
#         images = np.fromfile(imgpath,
#                              dtype=np.uint8).reshape(len(labels), 784)

#     return images, labels

# file_path = "/home/zhang/Desktop/MNIST"

# train_images, train_labels = load_mnist(file_path, "train")

# test_images, test_labels = load_mnist(file_path, "t10k")

train_set = torchvision.datasets.MNIST(root='/home/zhang/Documents', train=True, transform=transforms.ToTensor(),download= True )

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,shuffle=True, num_workers=2)

test_set = torchvision.datasets.MNIST(root='/home/zhang/Documents', train=False,transform=transforms.ToTensor(), download=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,shuffle=False, num_workers=2)

class NeuralNet(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_num)
        self.fc2 = nn.Linear(hidden_num, output_num)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return y

epoches = 10
lr = 0.001
input_num = 784
hidden_num = 500
output_num = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_num, hidden_num, output_num)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epoches):
    for i, data in enumerate(train_loader):
        (images, labels) = data
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoches, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("The accuracy of total {} images: {}%".format(total, 100 * correct/total))
