import cv2
import numpy as np
import dlib
from imutils import face_utils

#Initializing camera instance with default camera
cap = cv2.VideoCapture(0)

#Initialize the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0,0,0)

#For computation of euclidean distance between two points
def compute(pointA, pointB):   
    dist = np.linalg.norm(pointA - pointB)
    return dist

#There are 6 landmarks around eyes 4 have shorter distance and 2 points are farther from each other
def blinked(a,b,c,d,e,f):
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    #Check if eye blinked or not using ratio
    if(ratio>0.25):
        return 2
    elif(ratio>0.21 and ratio<=0.25):
        return 1
    else:
        return 0


while True:
    #Reading the frames from cap & Preprocessing of frame by gray conversion
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    #detected face in faces array captured from cam
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        #Creating a green colored rectangle around the copy face_frame
        cv2.rectangle(face_frame, (x1,y1), (x2,y2), (0,255,0),2)

        #using the predictor obj of 68 landmarks on converted gray img with current frame then creating numpy array
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        #The following numbers are actually landmarks which will show eye 0-based indexing
        # To Check if current frame has left or right eye blinked
        left_blink = blinked(landmarks[36],landmarks[37], 
        	landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
        	landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        if(left_blink==0 or right_blink==0):
            #sleep is incremented so that status don't keep on changing very frequently
            sleep += 1
            drowsy = 0
            active = 0
            #if the sleep persists then status updated with red color
            if(sleep>6):
                status = "SLEEPING!!"
                color = (255,0,0)
        elif(left_blink==1 or right_blink==1):
            sleep = 0
            drowsy += 1
            active = 0
            #if the drowsiness persists then status updated with blue color
            if(drowsy>6):
                status = "DROWSY!!"
                color = (0,0,255)
        else:
            sleep = 0
            drowsy = 0
            active += 1
            #if the person is active more than 6 frames then status updated with green color
            if(active>6):
                status = "ACTIVE :)"
                color = (0,255,0)

        #Put the status on actual frame captured by cam
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        #All the scanned landmarks are drawn with circles on face_frame
        for i in range(0,68):
            (x,y) = landmarks[i]
            cv2.circle(face_frame, (x,y), 1, (255,255,255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    #if esc is pressed user can exit the application
    if(key == 27):
        break
