import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from model import Deeplabv3

import cv2
from pprint import pprint
# Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
# as original image.  Normalization matches MobileNetV2

#배경이미지
backimg = cv2.imread('back.png')

trained_image_width=512
mean_subtraction_value=127.5
# image = np.array(Image.open('imgs/image1.jpg'))

cap = cv2.VideoCapture(0)   #카메라 객체 선언
cap.set(3, 1280)   #화면 크기 설정
cap.set(4, 720)
cap.set(5, 60)
ret, image = cap.read()  #비디오의 한 프레임씩 읽습니다. 제대로 프레임을 읽으면 ret값이 True, 실패하면 False가 나타납니다. fram에 읽은 프레임이 나옵니다
image = cv2.flip(image, 1)  #그림 좌우 반전 0은 상하 반전

deeplab_model = Deeplabv3()

while 1:
    ret, image = cap.read()
    image = cv2.flip(image, 1)
    # resize to max dimension of images from training dataset
    w, h, _ = image.shape
    ratio = float(trained_image_width) / np.max([w, h])
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

    # apply normalization for trained dataset images
    resized_image = (resized_image / mean_subtraction_value) - 1.

    # pad array to square image to match training images
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_width - resized_image.shape[1])
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    # make prediction
    # deeplab_model = Deeplabv3()
    res = deeplab_model.predict(np.expand_dims(resized_image, 0))
    labels = np.argmax(res.squeeze(), -1)

    # remove padding and resize back to original image
    if pad_x > 0:
        labels = labels[:-pad_x]
    if pad_y > 0:
        labels = labels[:, :-pad_y]
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    labels[labels[:,:] < np.max(labels)] = 0 # 색으로 구분해놓은 레이블 (밣을수록 숫자가 크다) 사람만 분류하려고 np.max가 사람 레이블 보다 작은값 (사람보다 작은것들(어두운거))은 0으로 바꾼다.
    labels = cv2.cvtColor(labels, cv2.COLOR_GRAY2BGR)
    image[labels[:, :, :] == 0] = 0 #라벨이미지에서 0은 사람 아닌부분 이미지에 대해서 사람이 아닌 부분은 0(검정으로)




    labels = cv2.resize(labels, dsize=(512, 384), interpolation=cv2.INTER_LINEAR)  #(원본이미지, 결과이미지, 보간법)
    image = cv2.resize(image, dsize=(512, 384), interpolation=cv2.INTER_LINEAR)   #결과이미지크기는 Tuple형 (너비,높이)
                                                                                 #보간법은 이미지크기를 변경하는 경우
                                                                                 #변형된 이미지 픽셀은 추정해서 값을 할당
    row, col, channel = image.shape #이미지(사람)의 행,열 크기값 받아오기

    backcopy = np.copy(backimg)  #배경이미지
    dst = backcopy[100:100 + row, 50:50 + col] #삽입하고자 하는 이미지의 위치에 넣을그림크기 만큼 이미지 컷팅
    #사람 이미지를 GRAYSCALE로 변환
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #BGR에서 GRAY로 색상 변환

    #임계값 설정을 통한 mask이미지 생성하기 threshold함수는 GRAYSCALE만 사용 가능함
    #(대상이미지, 기준치, 적용값, 스타일)
    #해당 cv2.THRESH_BINARY는 이미지내의 픽셀값이 기준치 이상인 값들은 모두 255로 부여함
    #즉 픽셀값이 100이상이면 흰색, 100미안이면 검정색으로 표시
    #변환된 이미지는 mask에 담김
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    #임계값 설정을 한 이미지를 흑백 반전시킴
    mask_inv = cv2.bitwise_not(mask)
    #위에서 자른 사람크기의 배경사진 영역에 mask에 할당된 이미지의 0이 아닌 부분만 dst와 dst이미지를 AND연산
    #즉 배경이미지에서 사람크기만큼의 영역에 사람의 모양만 0값이 부여된다.
    img_bg = cv2.bitwise_and(dst, dst, mask=mask_inv)
    #사람이미지에서 사람모양을 제외하고 다 0값을 가지게 됩니다.
    img_fg = cv2.bitwise_and(image, image, mask=mask)

    # 사람크기만큼의 영역의 이미지에 사람이미지를 연산합니다
    addimg = cv2.add(img_bg, img_fg)
    backcopy[100:100+row, 50:50+col] = addimg

    cv2.imshow('image', backcopy)
    cv2.imshow('labels', labels)

    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()

# plt.figure("img")
# plt.imshow(image)
# plt.figure('labels')
# plt.imshow(labels)
# plt.show()