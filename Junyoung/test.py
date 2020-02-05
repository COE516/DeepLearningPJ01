import numpy as np
import cv2

cap = cv2.VideoCapture(0)   #카메라 객체 선언

ret, myframe = cap.read()

myframe = cv2.flip(myframe, 1)

while (1):

    # reads frames from a camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()