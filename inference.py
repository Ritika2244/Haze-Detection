import imutils
import cv2
import io
import numpy as np
import torch 
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image 
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2)
        )
        
            
        self.fc1 = nn.Linear(32*5*5, 512)
        self.batch_norm1 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        out = self.block(x)
        out = self.block2(out)
        out = self.block2(out)
        out = self.block2(out)
        out = self.block2(out)
        out = out.view(out.size(0), -1)   # flatten out a input for Dense Layer
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def func(image):
    print("Yes 1")
    checkpoint_path='model.pt'
    model = Net()
    model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'),strict=False)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ])

    rgbimg = Image.new("RGB", image.size)
    rgbimg.paste(image)
    image = data_transforms(rgbimg).unsqueeze(0)
    image = Variable(image)
    vals = ['CLEAR','HAZE']
    print(model(image))
    a = np.argmax(model(image).detach().numpy())
    print(vals[a])

    return vals[a]

image = Image.open('0022.1.jpg')
func(image)