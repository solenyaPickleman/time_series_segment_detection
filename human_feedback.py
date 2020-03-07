
####EVAL on data2.csv
import pandas as pd 
import datetime
import operator
import numpy as np 
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import torch.optim as optim


from matplotlib import pyplot as plt 
from matplotlib.widgets import Button
from functools import reduce
from collections import Counter
from tqdm import tqdm
#read in data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# device = torch.device("cpu")
# print(device)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(16)
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear( 4096, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


PATH = "C:/Users/brady/time_series_segment_detection/model.sd"
net = Net()
net.load_state_dict(torch.load(PATH))
net.eval()
net.to(device)


human = []
machine = []
for files in ['C:/Users/brady/time_series_segment_detection/data.csv', 'C:/Users/brady/time_series_segment_detection/data2.csv']+ glob.glob('C:/Users/brady/Downloads/*.csv') + ['C:/Users/brady/time_series_segment_detection/data.csv', 'C:/Users/brady/time_series_segment_detection/data2.csv']:
    dat = pd.read_csv(files)
    dat.columns= ['date','value']
    dat = dat.sort_values(by=['date'])
    #dat = dat

    #build existence vectors, sorted by date of first appearance
    #take advantage of python 3.7's guarantee to preserve order

    onehot = dict.fromkeys(dat['value']).keys()

    def build_windows(dat, onehot):
        #get day's activity
        groups  = dat.groupby('date') 
        windows = []
        days = []
        for d, df in groups:
            vals = set(df['value'].unique()) #set for lookup sppppppeeeeeeeeeeeed
            vals = [1 if x in vals else 0 for x in onehot]
            windows.append(vals)
            days.append(d)
        return windows,days 

    windows, days = build_windows(dat, onehot)

    x = []
    y = []
    for i, vals in enumerate(windows):
        for ind,v in enumerate(vals):
            if v != 0:
                x.append(i)
                y.append(ind)

#sooooo y = mx? 
    plt.scatter(x,y)
    plt.scatter([max(x)-a for a in x],y)

    axstatic = plt.axes([0.7, 0.05, 0.1, 0.075])
    axdynamic = plt.axes([0.81, 0.05, 0.1, 0.075])
    bstatic = Button(axstatic, 'Static')
    bstatic.on_clicked(lambda x :[human.append('static'), plt.close()])
    bdynamic = Button(axdynamic, 'Dynamic')
    bdynamic.on_clicked(lambda x :[human.append('dynamic'), plt.close()])



    plt.show()


    windows,_ = build_windows(dat ,onehot)
    inputs = torch.Tensor(windows).to(device)
    h,w = inputs.shape
    inputs = inputs.reshape(1,1,h,w)
    inputs.to(device)
    outputs = net(inputs)
    predicted_static = outputs[0][0] > outputs[0][1]
    machine.append('static' if  predicted_static else 'dynamic')
print(list(zip(human, machine)))