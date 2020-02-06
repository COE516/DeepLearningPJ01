# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture(0)   #카메라 객체 선언
#
# ret, myframe = cap.read()
# myframe = myframe + 50
#
# myframe = cv2.flip(myframe, 1)
#
# while (1):
#
#     # reads frames from a camera
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#
#     copy = np.copy(frame)
#     copy1 = np.copy(frame)
#
#     copy[frame[:, :, :] > myframe[:, :, :]] = 0
#     copy1[np.allclose(frame, myframe, atol=1, rtol=1)] = 0
#
#     cv2.imshow("copy1", copy1)
#     cv2.imshow("OR", frame)
#     cv2.imshow("back", myframe)
#
#     # Wait for Esc key to stop
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
# # Close the window
# cap.release()
#
# # De-allocate any associated memory usage
# cv2.destroyAllWindows()

import numpy as np
import cv2

kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0

cap = cv2.VideoCapture(0)   #카메라 객체 선언

ret, myframe = cap.read()
myframe = myframe + 50

myframe = cv2.flip(myframe, 1)
myframe = cv2.cvtColor(myframe, cv2.COLOR_BGR2GRAY)
myframe = cv2.filter2D(myframe, -1, kernel_sharpen_3)


while (1):

    # reads frames from a camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    copy = np.copy(frame)
    copy1 = np.copy(frame)

    copy[frame[:, :] > myframe[:, :]] = 0
    # copy1[np.allclose(frame, myframe, atol=2.2, rtol=4.5)] = 0

    cv2.imshow("copy1", copy1)
    cv2.imshow("OR", frame)
    cv2.imshow("back", myframe)

    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()