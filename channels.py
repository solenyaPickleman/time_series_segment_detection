import pandas as pd 
import datetime
import operator
import numpy as np 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import torch.optim as optim


from matplotlib import pyplot as plt 
from functools import reduce
from collections import Counter
from tqdm import tqdm

#read in data
dat = pd.read_csv('C:/Users/brady/time_series_segment_detection/data.csv')
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

plt.scatter(x,y)
#plt.show()


#Build data based on samples(dynamic) and slices (static)
#in actual cases, we can just do random samples of N percent 
#on the data (once its been human classified via plots)
#breakpoints at 1000, 7000
training = []
targets = []

#dynamic
for _ in tqdm(range(1500)) :
    r =random.randint(75,100)/100.0
    size = len(dat)
    samplesize = int(r*size)
    windows , _  = build_windows (dat.sample(samplesize), onehot)
    training.append(windows)
    targets.append([0,1])

#static
for start, finish in zip([0, 1000, 7000], [1000, 7000, len(dat)-1]):
    print(start, finish)
    for _ in tqdm(range(500)) :
        r =random.randint(75,100)/100.0
        size = finish-start
        samplesize = int(r*size)
        windows , _  = build_windows (dat.iloc[start:finish].sample(samplesize), onehot)
        training.append(windows)
        targets.append([1,0])


dat = pd.read_csv('C:/Users/brady/time_series_segment_detection/data2.csv')
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

plt.scatter(x,y)
# plt.show()

#dynamic
for _ in tqdm(range(250)) :
    r =random.randint(75,100)/100.0
    size = len(dat)
    samplesize = int(r*size)
    windows , _  = build_windows (dat.sample(samplesize), onehot)
    training.append(windows)
    targets.append([0,1])

#static
for start, finish in zip([0, 4000], [4000, len(dat)-1]):
    for _ in tqdm(range(100)) :
        r =random.randint(75,100)/100.0
        size = finish-start
        samplesize = int(r*size)
        windows , _  = build_windows (dat.iloc[start:finish].sample(samplesize), onehot)
        training.append(windows)
        targets.append([1,0])




c = list(zip(training, targets))
random.shuffle(c)
training,targets = zip(*c)    
split = int(len(training)*.8)
test = training[split:]
test_targets = targets[split:]
training = training[:split]
training_targets = targets[:split]

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

net = Net()
net.to(device)
training_targets = torch.Tensor(training_targets).to(device)  # a dummy target, for example
#target = target.view(1, -1)  # make it the same shape as output
test_targets = torch.Tensor(test_targets) .to(device)  # a dummy target, for example

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(training):
        # get the inputs; data is a list of [inputs, labels]
        inputs = torch.Tensor(data).to(device)
        h,w = inputs.shape
        inputs = inputs.reshape(1,1,h,w)
        labels = training_targets[i].view(1,-1).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize  
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

print('Finished Training')


#test accuracy on test set
accuracy =0 
for i,data in tqdm(enumerate(test)):
        inputs = torch.Tensor(data)
        h,w = inputs.shape
        inputs = inputs.reshape(1,1,h,w)
        labels = test_targets[i].view(1,-1)

        #accomodate gpu
        inputs = inputs.to(device)
        labels = labels.to(device)

        is_static =test_targets[i][0] > test_targets[i][1]
        output = net(inputs)
        predicted_static = output[0][0] > output[0][1]
        if is_static == predicted_static:
             accuracy += 1

print("Model accuracy on test data: " ,accuracy/len(test))

#how to save
torch.save(net.state_dict(), PATH)

