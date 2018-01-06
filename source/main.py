import cv2
from keras.models import load_model
import numpy as np
from statistics import mode
from inference import *

# parameters
labels = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}
EMOTIONS = ['angry:', 'disgusted:', 'fear:', 'happy:', 'sad:', 'surprised:', 'neutral:'] 

# load models, don't forget to change paths if it is necessary
face_detection = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')   # face detection model from opencv
classifier = load_model('model/fer2013.hdf5', compile=False)    # load emotion classifier model
target_size = classifier.input_shape[1:3]   # get model shapes 
em_window = []                              # list to calculate emotions

# video streaming
video_cap = cv2.VideoCapture(0)             # start to capture

while True:
    frame = video_cap.read()[1]                     # get the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray image
    rgb_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert frame to rgb image
    faces = detect_faces(face_detection, gray)      # detect faces in the gray image

    for face_coord in faces:
        x1, x2, y1, y2 = apply_addition(face_coord, (20, 40)) # larger the face coordinates from detedected face
        gray_face = gray[y1:y2, x1:x2]                         # gray face with new coordinates
        try:
            gray_face = cv2.resize(gray_face, (target_size))   # apply resizing get of face, get missing from prev. detected face
        except:
            continue

        # face reshaping for classifier
        gray_face = preprocess_input(gray_face)     # change all of the array for prediction. make it between (-1,1)
        gray_face = np.expand_dims(gray_face, 0)    # add axis since we applyied reduce operation:  (64,64) --> (1, 64, 64)
        gray_face = np.expand_dims(gray_face, -1)   # change dimensions:  (1, 64, 64) --> (1, 64, 64, 1)

        # make a prediction 
        prediction = classifier.predict(gray_face)  # make a prediction with gray_face array from classifier which is loaded with fer2013 model
        em_probability = np.max(prediction)         # calculate probability
        em_label = np.argmax(prediction)            # label the prediction from one of the 7 emotions
        em_txt = labels[em_label]                   # extract text of labeled emotion from emotion labels output window
        em_window.append(em_txt)                    # append peredicted emotion

        if len(em_window) > 10:    # keep size of list of emotion less then 10
            em_window.pop(0)
        try:
            emotion_mode = mode(em_window)
        except:
            continue
  
        color = em_probability * np.asarray((40,205,78))     # bounding box color purple
        color = color.astype(int)                               # apply color to emotions
        color = color.tolist()      

        # draw_bounding_box(face_coord, rgb_im, color)       # draw bounding box to face
        # draw_text(face_coord, rgb_im, (emotion_mode + (":  %") +str(int(em_probability* 100))), color, 0, -45, 1, 1) # write the text from emotion_label
        for index, emotion in enumerate(EMOTIONS):
        	cv2.putText(rgb_im, emotion + ("  %") +str(int(prediction[0][index] * 100)), (10, index * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,205,78), 1);
        	cv2.rectangle(rgb_im, (180, index * 20 + 10), (180 + int(prediction[0][index] * 100), (index + 1) * 20 + 4), (40,205,78), -1)

    frame = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)
    cv2.imshow('emotion_detection_result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
