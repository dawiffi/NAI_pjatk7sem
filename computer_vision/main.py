import numpy as np
import cv2 

# by Kacper Pach s27112 & Dawid Frontczak s29608
# rules & environment setup in readme (https://github.com/dawiffi/NAI_pjatk7sem/blob/main/computer_vision/README.md)

cap = cv2.VideoCapture("ad.mp4")
videoPlaying = True 

capCam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def play_video(cap, videoPlaying):
    """
    Renders the next frame of the video file if the playback state is active.

    Args:
        cap (cv2.VideoCapture): The video file capture object.
        videoPlaying (bool): Flag indicating if the video should currently play.
    """
    if videoPlaying:
        ret, frame = cap.read()
        cv2.imshow('Reklama',frame)

def capture_camera(capCam):
    """
    Captures a frame from the webcam, detects faces/eyes, and draws bounding boxes.

    Processes the frame to grayscale, identifies regions of interest (ROI) for faces,
    and then searches for eyes within those regions.

    Args:
        capCam (cv2.VideoCapture): The webcam capture object.

    Returns:
        int: Total number of eyes detected in the current frame.
    """
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
cap.release()
cv2.destroyAllWindows()