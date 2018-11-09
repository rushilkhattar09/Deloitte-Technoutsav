#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 00:11:39 2018

@author: rushilkhattar
"""

# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

import cv2
import sys
from keras.models import load_model
import time
import numpy as np
from decimal import Decimal
from model_utils import define_model, model_weights
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
args = vars(ap.parse_args())

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"], queueSize=256).start()
time.sleep(1.0)
 
# start the FPS timer
fps = FPS().start()

def resize_img(image_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (48, 48))
    return True


 
def realtime_emotions():
   
    model = define_model()
    model = model_weights(model)
   

    
    save_loc = 'save_loc/1.jpg'
    
    result = np.array((1,7))
    
    once = False
   
    faceCascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
  
    EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

   
    emoji_faces = []
    for index, emotion in enumerate(EMOTIONS):
        emoji_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

    
   # video_capture = cv2.VideoCapture('vid5.mov')
    #fvs.set(3, 640)  
    #fvs.set(4, 480)  

    
    prev_time = time.time()
    vid = []
    # start webcam feed
    while fvs.more():
        stime = time.time()
        frame = fvs.read()
        
        #frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        

       
        faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            
            roi_color = frame[y-90:y+h+70, x-50:x+w+50]

           
            cv2.imwrite(save_loc, roi_color)
            
            cv2.rectangle(frame, (x-10, y-70),
                            (x+w+20, y+h+40), (15, 175, 61), 4)
            
            
            curr_time = time.time()
           
            if curr_time - prev_time >=1:
                
                img = cv2.imread(save_loc, 0)
                
                if img is not None:
                  
                    once = True

                   
                    img = cv2.resize(img, (48, 48))
                    img = np.reshape(img, (1, 48, 48, 1))
                   
                    result = model.predict(img)
                    print(EMOTIONS[np.argmax(result[0])])
                    
                
                prev_time = time.time()

            if once == True:
                total_sum = np.sum(result[0])
               
                emoji_face = emoji_faces[np.argmax(result[0])]
                for index, emotion in enumerate(EMOTIONS):
                    text = str(
                        round(Decimal(result[0][index]/total_sum*100), 2) ) + "%"
                   
                    cv2.rectangle(frame, (100, index * 20 + 10), (100 +int(result[0][index] * 100), (index + 1) * 20 + 4),
                                    (255, 0, 0), -1)
                   
                    cv2.putText(frame, emotion, (10, index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (173, 9, 136), 2)
                    
                    cv2.putText(frame, text, (105 + int(result[0][index] * 100), index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 55, 125), 1)
                    
                    
               
                for c in range(0, 3):
                    
                    foreground = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0)
                    background = frame[350:470, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
                    frame[350:470, 10:130, c] = foreground + background
            break
        print(time.time()-stime)
        # Display the resulting frame
        #cv2.imshow('Video', frame)
        vid.append(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_width = int(fvs.stream.get(3))
    frame_height = int(fvs.stream.get(4))
    out = cv2.VideoWriter('outpy7.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))

    for a in vid:
        out.write(a)

    video = cv2.VideoCapture('outpy7.avi')
    while True:
        ret, frame = video.read()
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    fvs.release()
    cv2.destroyAllWindows()

realtime_emotions()





