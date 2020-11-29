from torch.utils.data.dataset import Dataset
import torch
import json
import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import natsort

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
class MyCustomDataset(Dataset):
    def __init__(self, category = "labels_100", json_file_path="/scratch/s174411/WLASL_v0.3.json", frame_location="/scratch/s174411/Processed_data"):
        self.frame_location = frame_location
        # Defining count_dictionary which contains the number of videos for each class
        # and defining video_id_dictionary which has all of the video id's for each class.
        with open(json_file_path, "r") as read_file:
            data = json.load(read_file)

        self.count_dictionary = {}
        self.video_id_dictionary = {}

        for instance in data:
            current_label = instance['gloss']
            self.count_dictionary[current_label] = 0
            self.video_id_dictionary[current_label] = []
            inner_array = instance['instances']
            for video in inner_array:
                self.count_dictionary[current_label] += 1
                self.video_id_dictionary[current_label].append(video['video_id'])


        # Assigning an integer to each class.
        self.labels_iterated = {}
        self.inv_video_id_dictionary = {}
        self.training_data = []
        self.Categories(category, frame_location)

    def __getitem__(self, index):
        buffer, label = self.training_data[index]
        images = []
        for i in range(64):
            path = buffer[i]
            #print("PATH: ", path)
            img = np.array(cv2.imread(path))
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #img = img/255
            images.append(img)
        images = np.array(images)
        images = self.to_tensor(images)
        #temp = buffer.transpose(1,2,3,0)
        #img = temp[0]
        #imgplot = plt.imshow(img)
        #plt.show()
        return [images, label]


    def __len__(self):
        return self.total_videos()

    def total_videos(self):


        sum_count = 0
        for label in self.labels_x:
            for video in self.video_id_dictionary[label]:
                sum_count += 1
        return sum_count

    def to_tensor(self, buffer):
        return torch.from_numpy(buffer.transpose(3, 0, 1, 2))

    # Changes needed to be made from original version.
    # 1. Load only 16 frames into the buffer.
    # 2. Resize each frame to 224?
    # 3. One video has a label. So a batch of 16 frames is assigned a class.
    def make_training_data(self, labels_x,frame_location,buffer_size = 64,image_height = 112, image_width = 112):

        data_directory = ("{}/{}".format(frame_location, len(labels_x)))
        num_labels = len(labels_x)
        for label in tqdm(labels_x):
            for video in self.video_id_dictionary[label]:
                buffer = []
                path = os.path.join(data_directory, video)
                number_of_frames = len([file for file in os.listdir(path) if "jpg" in file])
                try:
                    save_frequency = np.floor(number_of_frames/buffer_size)
                    if save_frequency == 0:
                        save_frequency = 1
                    if number_of_frames % save_frequency == 0:
                        save_start = 0
                    else:
                        save_start = save_frequency
                    if number_of_frames % save_frequency  != 0 or number_of_frames / save_frequency > buffer_size:
                        save_start = (np.ceil(number_of_frames/save_frequency)-buffer_size)*save_frequency
                except Exception as e:
                    print(e)
                to_repeat = False
                if number_of_frames < buffer_size:
                    repeat = buffer_size - number_of_frames
                    to_repeat = True
                    save_frequency = 1
                    save_start = 1

                counter = 1
                #buffer = np.empty((buffer_size, image_height, image_width, 3), np.dtype('float32'))
                buffer = []
                index = 0
                fff = natsort.natsorted(os.listdir(path),reverse=False)

                for file in fff:
                    if (counter % save_frequency == 0 and counter > save_start) or (counter % save_frequency == 0 and counter >= save_start and number_of_frames % save_frequency != 0):
                        if "jpg" in file:
                            try:
                                path = os.path.join(data_directory, video, file)
                                #img = np.array(cv2.imread(path))
                                #img = cv2.resize(img, (image_height, image_width))
                                #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                                #img = img/255
                                buffer.append(path)
                                #print(path)
                                #buffer[index] = img
                                index += 1
                                if counter == number_of_frames and to_repeat == True:
                                    for i in range(repeat+1):
                                        #buffer[index] = img
                                        #buffer.append(img)
                                        buffer.append(path)
                                        index += 1
                            except Exception as e:
                                print(e)
                                pass
                    counter += 1
                # if len(buffer) != buffer_size:
                #     print("Buffer is not of the right size")
                #     print("video:",video ,len(buffer))
                #     print(f"Video:", video, "Number of frames:", number_of_frames, "Save_frequency:", save_frequency, "Save start:", save_start)
                self.training_data.append([(np.array(buffer)),self.labels_iterated[label]])


    def Categories(self, category,frame_location):
        # Creating a list for 100, 200, 500, 1000 and 2000 classes with the highest amount of videos.
        count = 0
        labels_100 = []
        labels_200 = []
        labels_500 = []
        labels_1000 = []
        labels_2000 = []

        for key in self.count_dictionary:
            if count < 100:
                labels_100.append(key)
            if count < 200:
                labels_200.append(key)
            if count < 500:
                labels_500.append(key)
            if count < 1000:
                labels_1000.append(key)
            labels_2000.append(key)
            count += 1

        if category == "labels_100":
            self.labels_x = labels_100
        elif category == "labels_200":
            self.labels_x = labels_200
        elif category == "labels_500":
            self.labels_x = labels_500
        elif category == "labels_1000":
            self.labels_x = labels_1000
        elif category == "labels_2000":
            self.labels_x = labels_2000

        # Assigning an integer to each class.
        counter = 0
        for label in self.labels_x:
            self.labels_iterated[label] = counter
            counter += 1

        # Creating a dictionary where given a video whe can fint its class.
        for k, v in self.video_id_dictionary.items():
            for video in v:
                self.inv_video_id_dictionary[video] = k

        self.make_training_data(self.labels_x, frame_location)
