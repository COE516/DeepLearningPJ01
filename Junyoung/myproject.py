import numpy as np
import cv2
from plantcv import plantcv as pcv

# 이미지의 메디안에 따라 자동 에지 검출
# def auto_canny(image, sigma=0.33):
#     # compute the median of the single channel pixel intensities
#     v = np.median(image)
#
#     # apply automatic Canny edge detection using the computed median
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     edged = cv2.Canny(image, lower, upper)
#
#     # return the edged image
#     return edged
# 블러
size=5

kernel_motion_blur = np.zeros((size,size))
kernel_motion_blur[int((size-1)/2),:] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# output = cv2.filter2D(img,-1,kernel_motion_blur)

cap = cv2.VideoCapture(0)   #카메라 객체 선언

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ선명도 필터ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
kernel_sharpen_1 = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])
kernel_sharpen_2 = np.array([[1,1,1],[1,-7,1],[1,1,1]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

ret, myframe = cap.read()

myframe = cv2.flip(myframe, 1)
M = np.ones(myframe.shape, dtype = "uint8") * 30 # 이미지 픽셀만큼 공간만들고, 100으로
M2 = np.ones(myframe.shape, dtype = "uint8") * 50
# myframe = cv2.add(myframe, M)

# myframe = cv2.filter2D(myframe, -1, kernel_sharpen_2)
myframe = cv2.filter2D(myframe, -1, kernel_sharpen_3)





# loop runs if capturing has been initialized
while (1):

    # reads frames from a camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.subtract(frame, M2)

    cv2.imshow('Original', frame)
    cv2.imshow('MyOriginal', myframe)

    gray1 = cv2.cvtColor(myframe, cv2.COLOR_BGR2GRAY)
    blurred1 = cv2.GaussianBlur(gray1, (3, 3), 3)
    auto1 = pcv.canny_edge_detect(gray1, thickness=3.9)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 5)
    blurred = cv2.filter2D(blurred, -1, kernel_motion_blur)
    auto = pcv.canny_edge_detect(blurred, thickness=2.5) - auto1
    canny_median_blur = cv2.medianBlur(auto, 3)

    # cv2.imshow('Edges OR', auto)
    cv2.imshow('Edges', canny_median_blur)
    cv2.imshow('Edges1', auto1)


    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
