from scipy.io import loadmat
from random import shuffle
import pandas as pd
import numpy as np
import os
import cv2

class DataManager(object):
    def __init__(self, dataset_name='fer2013', dataset_path=None, image_size=(48, 48)):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'fer2013':
            self.dataset_path = '../datasets/fer2013/fer2013.csv'
        else:
            raise Exception('Incorrect dataset name')

    def _load_fer2013(self): # load the csv file from given path. 
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

def get_labels(dataset_name): # according to Fer2013
    return {0:'angry',1:'disgust',2:'fear',3:'happy', 4:'sad',5:'surprise',6:'neutral'}


def get_class_to_arg(dataset_name): # according to Fer2013
    return {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6}

def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

def preprocess_input(x, v2=True): # change input image values. Make them all between (-1,1)
    x = x.astype('float32') # cast all the values from int to float
    x = x / 255.0           # all of them between (0,1)
    if v2:                  # make them all between (-1,1)
        x = x - 0.5
        x = x * 2.0
    return x
