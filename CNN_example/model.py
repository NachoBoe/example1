import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(torch.Module):
    def __init__(self,in_channels,classes):
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=20,kernel_size=(5,5))
        self.relu = nn.ReLU()
        self.maxPool =nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.conv2 = nn.Conv2d(in_channels=20,out_channels=50,kernel_size=(5,5))
        self.relu2 = nn.ReLU()
        self.maxPool2 =nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.conv3 = nn.Conv2d(in_channels=in_channels,out_channels=10,kernel_size=(3,3))
        self.relu3 = nn.ReLU()
        self.maxPool3 =nn.MaxPool2d(kernel_size=(2,2))

        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxPool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxPool3(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return self.logSoftmax(x)

