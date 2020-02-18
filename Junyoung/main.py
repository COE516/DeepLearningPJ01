import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from model import Deeplabv3

import cv2
from pprint import pprint
# Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
# as original image.  Normalization matches MobileNetV2

backimg = cv2.imread('back.png')

trained_image_width=512
mean_subtraction_value=127.5
# image = np.array(Image.open('imgs/image1.jpg'))

cap = cv2.VideoCapture(0)   #카메라 객체 선언
cap.set(3, 1280)
cap.set(4, 720)
cap.set(5, 60)
ret, image = cap.read()
image = cv2.flip(image, 1)

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
    labels[labels[:,:] < np.max(labels)] = 0
    labels = cv2.cvtColor(labels, cv2.COLOR_GRAY2BGR)
    image[labels[:, :, :] == 0] = 0




    labels = cv2.resize(labels, dsize=(512, 384), interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(image, dsize=(512, 384), interpolation=cv2.INTER_LINEAR)
    row, col, channel = image.shape

    backcopy = np.copy(backimg)
    dst = backcopy[100:100 + row, 50:50 + col]

    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(dst, dst, mask=mask_inv)
    img_fg = cv2.bitwise_and(image, image, mask=mask)

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