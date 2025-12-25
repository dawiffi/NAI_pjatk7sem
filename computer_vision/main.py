import numpy as np
import cv2 


cap = cv2.VideoCapture("ad.mp4")
videoPlaying = True 

capCam = cv2.VideoCapture("vide.mp4")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def play_video(cap, videoPlaying):
    if videoPlaying:
        ret, frame = cap.read()
        cv2.imshow('Reklama',frame)

def capture_camera(capCam):
    ret, frame = capCam.read()
    eyes_total = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        eyes_total += len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
    cv2.imshow('camera', frame)
    return eyes_total

     
while True:
    play_video(cap, videoPlaying)
    eyes_total = capture_camera(capCam)
    if eyes_total < 2:
        videoPlaying = False
    elif eyes_total >= 2:
        videoPlaying = True

    if cv2.waitKey(1) == ord('q'):
        break

capCam.release()
cv2.destroyAllWindows()