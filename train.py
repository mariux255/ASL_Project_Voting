import tqdm
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets, models
from Dataloader.WSASL_Videos_load import MyCustomDataset
from Model.CNN_Vanilla_frame_classification import Net
from torch.utils.data import random_split
import csv
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = MyCustomDataset("labels_100")
dataset_size = (len(dataset))

val_size = int(np.floor(dataset_size * 0.1))
train_size = int(dataset_size - val_size)
trainset, validset = random_split(dataset, [train_size, val_size])
bs = 20
num_classes = 100
dataloader_train = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
dataloader_val = DataLoader(validset, batch_size=bs, shuffle=True, num_workers=2)


net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
net.fc = nn.Linear(num_ftrs, 100)

net = net.to(device)

#net = Net()
#net = net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)
criterion = criterion.to(device)
scheduler = StepLR(optimizer, step_size = 1, gamma=0.1)
def voting(outputs):
    answer = [0] * (num_classes+1)
    predictions = np.empty((bs,num_classes), np.dtype('float32'))
    y = torch.argmax(outputs, dim = 1)
    for instance in y:
        answer[instance] += 1

    temp = 1
    for j in range(len(answer)):
        if answer[j]>=answer[temp]:
            temp = j
    return temp

def accuracy(ys, ts):
    y = ys
    x = ts
    correct = 0
    for i in range(len(y)):
        if y[i] == x[i]:
            correct += 1
    return correct/len(y)

now = datetime.now()
filename = "{}".format(now.strftime("%H:%M:%S") + ".csv")
title = ['{}'.format(net)]
headers = ['ID', 'Type','Epoch','Loss','Accuracy']
with open(filename,'w') as csvfile:
    Identification = 1
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(headers)
    for epoch in range(2):  # loop over the dataset multiple times
        
        net.train()
        training_loss = 0.0
        running_acc = 0
        for i,(inputs, labels) in enumerate(dataloader_train):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.view(-1, 3,16, 112, 112)
            inputs = inputs.permute(0,2,1,3,4)
            inputs = inputs.to(device)
            labels = torch.LongTensor(labels).to(device)

            optimizer.zero_grad()
            predictions = np.empty((bs), np.dtype('float32'))
            inner_loss = 0
            for j in range((inputs.shape)[0]):
                outputs = net(inputs[j])
                labels_frames = [labels[j]] * 16
                labels_frames = np.asarray(labels_frames)
                labels_frames = torch.LongTensor(labels_frames).to(device)
                loss = criterion(outputs, (labels_frames))
                predictions[j] = voting(outputs)
                loss.backward()
                inner_loss += loss.item()
                optimizer.step()
            training_loss += (inner_loss/20)
            running_acc += accuracy(predictions,labels)
            if i%30 == 0:
                #print("pred:", predictions)
                csvwriter.writerow(['{}'.format(Identification),'{}'.format("Training"),'{}'.format(epoch),'{}'.format(training_loss/(i+1)),'{}'.format(running_acc/(i+1))])
                print(f"Training phase, Epoch: {epoch}. Loss: {training_loss/(i+1)}. Accuracy: {running_acc/(i+1)}.")
                Identification += 1

        net.eval()
        valError = 0
        for i, (inputs,labels) in enumerate(dataloader_val):
            with torch.no_grad():
                inputs = inputs.view(-1, 3,16, 112, 112)
                inputs = inputs.permute(0,2,1,3,4)
                inputs = inputs.to(device)
                labels = torch.LongTensor(labels).to(device)

                predictions = np.empty((bs), np.dtype('float32'))
                inner_loss = 0
                for j in range((inputs.shape)[0]):
                    outputs = net(inputs[j])
                    labels_frames = [labels[j]] * 16
                    labels_frames = np.asarray(labels_frames)
                    labels_frames = torch.LongTensor(labels_frames).to(device)
                    loss = criterion(outputs, (labels_frames))
                    predictions[j] = voting(outputs)
                    inner_loss += loss.item()
                running_acc += accuracy(predictions,labels)
                valError += (inner_loss/20)
            if i%20 == 0:
                csvwriter.writerow(['{}'.format(Identification),'{}'.format("Validation"),'{}'.format(epoch),'{}'.format(valError/(i+1)),'{}'.format(accuracy(outputs,labels))])
                print(f"Validation phase, Epoch: {epoch}. Loss: {valError/(i+1)}. Accuracy: {accuracy(outputs,labels)}.")
                Identification += 1
        scheduler.step()
    csvwriter.writerow(title)
print('Finished Training')
