import cv2
import numpy as np



faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('VideoOfPeopleWalking.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 80)


if (cap.isOpened()==False):
    print('Error Reading video')

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,
    minNeighbors = 5, minSize=(25,25))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    if ret == True:
        cv2.imshow('Video',frame)
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()
