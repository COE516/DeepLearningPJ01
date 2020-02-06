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
for i in range(5):
    ret, myframe = cap.read()
M = np.ones(myframe.shape, dtype = "uint8") * 50
M1 = np.ones(myframe.shape, dtype = "uint8") * 0

myframe = cv2.flip(myframe, 1)
myframe = cv2.add(myframe, M)
myframe = cv2.filter2D(myframe, -1, kernel_sharpen_3)
comyframe = np.copy(myframe)
comyframe = cv2.cvtColor(comyframe, cv2.COLOR_BGR2HSV)
comyframe = cv2.medianBlur(comyframe, 11)

while (1):

    # reads frames from a camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.subtract(frame, M1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.medianBlur(frame, 11)

    copy = np.copy(frame)
    copy1 = np.copy(frame)

    # copy[frame[:, :, 0] == myframe[:, :, 0]] = 0


    degree = 20
    H = 0

    upper = comyframe[:, :, H] + degree
    lower = comyframe[:, :, H] - degree
    # copy[(lower < frame[:, :, H]) & (frame[:, :, H] < upper)] = 0
    copy[frame[:, :, H] > upper] = 0
    copy1[frame[:, :, H] < lower] = 0

    copy = cv2.cvtColor(copy, cv2.COLOR_HSV2BGR)
    copy1 = cv2.cvtColor(copy1, cv2.COLOR_HSV2BGR)

    result = copy & copy1
    result = cv2.medianBlur(result, 1)

    # result = ~result
    # result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Hf", frame)
    cv2.imshow("Hmfback", comyframe)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    # cv2.imshow("copy", copy)
    # cv2.imshow("copy1", copy1)
    cv2.imshow("result", result)
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