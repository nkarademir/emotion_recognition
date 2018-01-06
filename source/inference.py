import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

def detect_faces(detect_model, im_array): 
    # return face from image. Detect_model is face detection of opencv.          
    return detect_model.detectMultiScale(im_array, 1.3, 5)


def draw_bounding_box(face_coord, im_array, color): 
    # drawing rectangle to output image
    x, y, w, h = face_coord
    cv2.rectangle(im_array, (x, y), (x + w, y + h), color, 2)


def apply_addition(coord, add):
    # to larger the initial face rectangle fo opencv function result                     
    x, y, w, h = coord
    x_add, y_add = add
    return ((x - x_add), (x + w + x_add), (y - y_add), (y + h + y_add))


def draw_text(coord, im_array, txt, color, x_offset=0, y_offset=0, font=2, thickness=2): 
    # writing the emotion mode to the image, above the face rectangle
    x, y = coord[:2]
    cv2.putText(im_array, txt, (x + x_offset, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                font, color, thickness, cv2.LINE_AA)

def preprocess_input(x):
    # change image array for emotion prediction. Makes all values in the array between (-1,1)
    x = x.astype('float32')
    x = x / 255.0
    x = x - 0.5
    x = x * 2.0
    return x