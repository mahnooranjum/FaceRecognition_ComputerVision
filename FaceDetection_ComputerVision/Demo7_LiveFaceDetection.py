##==============================================================================
##   Demo By: Mahnoor Anjum
##   Date: 31/03/2019
##   Codes inspired by:
##   Rajeev Ratab
##   Github.com/imvinod/
##   Official Documentation
##==============================================================================

import numpy as np
import cv2
# Classifier (XML file format)
face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray)
    if faces is ():
        return cv2.flip(image,1)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
    image = cv2.flip(image, 1)
    return image

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('Demo7.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    ret, frame = cap.read()
    
    cv2.imshow('Face Detector', detector(frame))
    out.write(detector(frame))

    if cv2.waitKey(1)==13:
        break
cap.release()
out.release()

cv2.destroyAllWindows()
