import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pygame

cap = cv2.VideoCapture("C:/Users/IMLAB/Desktop/greenline.mp4")

if cap.isOpened() == False:
    print("Error opening video stream or file")

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = 0
stop_frame = length-100
fig = plt.figure()

while (cap.isOpened()):
    frame_number = frame_number + 1
    time.sleep(0.1)
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape)

    framebgr = frame[:, :, :]
    # bgr = [140, 200, 220]
    bgr = [130, 160, 122]  # B G R
    # bgr = [0, 128, 0]

    thresh = 40

    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    maskBGR = cv2.inRange(framebgr, minBGR, maxBGR)
    resultBGR = cv2.bitwise_and(framebgr, framebgr, mask=maskBGR)

    # convert to HSV
    brightHSV = cv2.cvtColor(framebgr, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
    maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

    maskHSV = cv2.inRange(brightHSV, minHSV, maxHSV)
    resultHSV = cv2.bitwise_and(brightHSV, brightHSV, mask=maskHSV)

    maskarea1 = [17, 343]
    maskarea2 = [180, 401]
    for i in range(maskarea1[0], maskarea2[0]):
        for j in range(maskarea1[1], maskarea2[1]):
            resultHSV[j, i] = 0

    maskarea1 = [803, 1]
    maskarea2 = [1115, 59]
    for i in range(maskarea1[0], maskarea2[0]):
        for j in range(maskarea1[1], maskarea2[1]):
            resultHSV[j, i] = 0

    cv2.imshow('Result HSV', resultHSV)
