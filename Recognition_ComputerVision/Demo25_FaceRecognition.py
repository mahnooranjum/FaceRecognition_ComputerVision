##==============================================================================
##   Demo By: Mahnoor Anjum
##   Date: 27/04/2019
##   Codes inspired by:
##   Official Documentation
##==============================================================================


import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

cap = cv2.VideoCapture(0)
count = 0


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    if face_extractor(frame) is not None:
        count += 2
        face = cv2.resize(face_extractor(frame), (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file_name_path = 'faces/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropped', face)
    else:
        print("Faces not found")
        pass
    if cv2.waitKey(1) == 13 or count==100:
        break
    
cap.release()
cv2.destroyAllWindows()
print("Samples collected")
        
        

data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype= np.uint8))
    Labels.append(i)
    
Labels = np.asarray(Labels, dtype=np.int32)


model=cv2.face.LBPHFaceRecognizer_create(threshold=95)
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Trained")

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img, roi

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)
    frame = cv2.flip(frame,1)  

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        results = model.predict(face)
        if results[1] < 500:
            confidence = int (100 * ( 1 - ( results [1])/300))
            display_string = str(confidence) + "% confident"
        cv2.putText(image, display_string, (100,120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 25:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0), 2)
            cv2.imshow("Face cropper", image)
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2)
            cv2.imshow("Face cropper", image)
           
    except:
        cv2.putText(image, "No Face", (220, 120), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2)
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2)
        cv2.imshow("Face cropper", image)
        pass
    if cv2.waitKey(1) == 13:
        break
    
cap.release()
cv2.destroyAllWindows()
        
    
        
        
        
        
        
            