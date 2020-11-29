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
num_classes = 100
dataset = MyCustomDataset("labels_100")
#dataset = MyCustomDataset(category='labels_100',json_file_path="/home/marius/Documents/Projects/WLASL_v0.3.json", frame_location="/home/marius/Documents/Projects/Processed_data")

dataset_size = (len(dataset))

val_size = int(np.floor(dataset_size * 0.2))
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
scheduler = StepLR(optimizer, step_size = 10, gamma=0.1)
def voting(outputs):
    answer = np.array([0]*num_classes)
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
    for epoch in range(30):  # loop over the dataset multiple times

        net.train()
        training_loss = 0.0
        running_acc = 0
        for i,(inputs, labels) in enumerate(dataloader_train):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.view(-1, 3,64, 224, 224)
            inputs = inputs.float()
            inputs = inputs.permute(0,2,1,3,4)
            inputs = inputs.to(device)
            labels = torch.LongTensor(labels).to(device)

            optimizer.zero_grad()
            predictions = []
            inner_loss = 0

            for j in range((inputs.shape)[0]):
                outputs = net(inputs[j])

                frames_labels = ([labels[j]] * 64)
                frames_labels = torch.tensor(frames_labels)
                frames_labels = torch.LongTensor(frames_labels).to(device)
                loss = criterion(outputs, (frames_labels))
                predictions.append(voting(outputs))
                loss.backward()
                inner_loss += loss.item()
                optimizer.step()
            training_loss += (inner_loss/bs)
            predictions = torch.tensor(predictions)
            predictions = predictions.to(device)
            #running_acc += accuracy(predictions,labels)
            correct = (predictions == labels).sum().item()
            running_acc += (correct/(labels.size(0)))
            if i%1 == 0:
                #print("pred:", predictions)
                csvwriter.writerow(['{}'.format(Identification),'{}'.format("Training"),'{}'.format(epoch),'{}'.format(training_loss/(i+1)),'{}'.format(running_acc/(i+1))])
                print(f"Training phase, Epoch: {epoch}. Loss: {training_loss/(i+1)}. Accuracy: {running_acc/(i+1)}.")
                Identification += 1

        net.eval()
        valError = 0
        running_acc = 0
        for i, (inputs,labels) in enumerate(dataloader_val):
            with torch.no_grad():
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.view(-1, 3,64, 224, 224)
                inputs = inputs.permute(0,2,1,3,4)
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = torch.LongTensor(labels).to(device)

                predictions = []
                inner_loss = 0
                for j in range((inputs.shape)[0]):
                    outputs = net(inputs[j])

                    frames_labels = ([labels[j]] * 64)
                    frames_labels = torch.tensor(frames_labels)
                    frames_labels = torch.LongTensor(frames_labels).to(device)
                    loss = criterion(outputs, (frames_labels))
                    predictions.append(voting(outputs))
                    inner_loss += loss.item()
                predictions = torch.tensor(predictions)
                predictions = predictions.to(device)
                #running_acc += accuracy(predictions,labels)
                correct = (predictions == labels).sum().item()
                valError += (inner_loss/bs)
                #running_acc += accuracy(predictions,labels)
                correct = (predictions == labels).sum().item()
                running_acc += (correct/(labels.size(0)))
            if i%1 == 0:
                csvwriter.writerow(['{}'.format(Identification),'{}'.format("Validation"),'{}'.format(epoch),'{}'.format(valError/(i+1)),'{}'.format(running_acc/(i+1))])
                print(f"Validation phase, Epoch: {epoch}. Loss: {valError/(i+1)}. Accuracy: {running_acc/(i+1)}.")
                Identification += 1
        scheduler.step()
    csvwriter.writerow(title)
print('Finished Training')
