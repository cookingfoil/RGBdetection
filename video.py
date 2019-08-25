import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pygame

cap = cv2.VideoCapture("C:/Users/IMLAB/Desktop/DetectionTestVideo/thicker.mp4")

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
img = np.zeros([1,3], dtype=np.uint8)
#print(img)

frame_number = 0
stop_frame = length-100
fig = plt.figure()


# Read until video is completed
while (cap.isOpened()):

############# RGB detection
    # frame_number = frame_number + 1
    # time.sleep(0.5)
    # # Capture frame-by-frame
    # ret, frame = cap.read()
    # # print(frame.shape)
    # framergb = frame[:,:,:]
    # # lower_yellow = np.array([120,180,210])
    # # upper_yellow = np.array([160, 215, 255])
    #
    # # yellow boxes
    # # lower_yellow = np.array([100, 160, 190])
    # # upper_yellow = np.array([180, 230, 255])
    #
    # # green boxes
    # lower_yellow = np.array([149, 130, 130])
    # upper_yellow = np.array([170, 170, 160])
    #
    # mask = cv2.inRange(framergb, lower_yellow, upper_yellow)
    # res = cv2.bitwise_and(frame, frame, mask = mask)
    #
    # cv2.imshow('frame', frame)
    # # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    # k = cv2.waitKey(5) & 0xFF
    # if k == 27:
    #     break
############# RGB detection

############# HSV detection
    frame_number = frame_number + 1
    time.sleep(0.1)
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape)

    framebgr = frame[:, :, :]
    # bgr = [140, 200, 220]
    bgr = [31, 203, 17] # B G R
    # bgr = [0, 128, 0]

    b, g, r = cv2.split(frame)
    # print("frame:", frame)
    # print("g:", g)
    # cv2.imshow("Green", g)

    thresh = 40

    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    maskBGR = cv2.inRange(framebgr,minBGR,maxBGR)
    resultBGR = cv2.bitwise_and(framebgr, framebgr, mask = maskBGR)

    # convert to HSV
    brightHSV = cv2.cvtColor(framebgr, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    # print("hsv:", hsv)



    # minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
    # maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

    huethresh = 40
    sthresh = 80
    minHSV = np.array([hsv[0] - huethresh, hsv[1] - sthresh, hsv[2] - sthresh])
    maxHSV = np.array([hsv[0] + huethresh, hsv[1] + sthresh, hsv[2] + sthresh])


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

    maskarea1 = [124, 180]
    maskarea2 = [170, 230]
    for i in range(maskarea1[0], maskarea2[0]):
        for j in range(maskarea1[1], maskarea2[1]):
            resultHSV[j, i] = 0

    cv2.imshow('Result HSV', resultHSV)

###################### Blob detection
# Read image
#     im = cv2.imread(resultHSV, cv2.IMREAD_GRAYSCALE)
    h, s, v1 = cv2.split(resultHSV)
    im = v1


###################### line detection
    # Read image
    # img = cv2.imread('C:/Users/IMLAB/Desktop/rect_test.jpg', cv2.IMREAD_COLOR) # road.png is the filename
    # # # Convert the image to gray-scale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("gray:", gray)
    # print("im:", im)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(im, 50, 200)
    # Detect points that form a line
    max_slider = 10
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, max_slider, minLineLength=1, maxLineGap=250)
    # lines = cv2.HoughLines(edges, 1, np.pi/180, max_slider)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, max_slider)
    print("lines: ", lines)
    print("line x1: ", min(lines[:,0,0]))
    print("line x1: ", max(lines[:,0,0]))
    print("line x2: ", lines[:,0,2])
    print("line y1: ", lines[:,0,1])
    # cv2.imshow("lines", lines)
    # Draw lines on the image

    minx1 = min(lines[:,0,0])
    maxx1 = max(lines[:,0,0])
    minx2 = min(lines[:,0,2])
    maxx2 = max(lines[:,0,2])
    miny1 = min(lines[:,0,1])
    maxy1 = max(lines[:,0,1])
    miny2 = min(lines[:,0,3])
    maxy2 = max(lines[:,0,3])

    if lines is not None :
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if ((x1-x2)**2 + (y1-y2)**2)**0.5 >= 3:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # position = open("C:/Users/IMLAB/Desktop/DetectionTestVideo/TargetPosition.txt", 'w')
            # position.write(np.array2string(lines))
            # position.close()
            # position = open("C:/Users/IMLAB/Desktop/DetectionTestVideo/TargetPosition.txt", 'w')
            # np.savetxt("TargetPosition.csv", line, delimiter=",")



    # Show result
    cv2.imshow("Result Image", frame)
###################### line detection

##########Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0.000001;
    params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 0.000001

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    # ver = (cv2.__version__).split('.')
    # if int(ver[0]) < 3:
    #     detector = cv2.SimpleBlobDetector(params)
    # else:
    #     detector = cv2.SimpleBlobDetector_create(params)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector.create()

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)

###################### Blob detection

    # # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
############# HSV detection


    if ret == True:
        if frame_number < stop_frame:
            # for i in range(1080):
            #     for j in range(1920):
            #         # print(frame[i,j])
            #          if (120 < frame[i,j,0] < 160) and (180 < frame[i,j,1] < 215) and (210 < frame[i,j,2] < 255):
            #             cv2.imshow('rgbdetection', frame)
            #             cv2.circle(frame, (i, j), 10, (0, 255, 255), thickness=10)
            print(frame_number)
            pixel = frame[:,:]

            average = pixel.mean(axis=0).mean(axis=0)
            # print (average)
            img = img + average
            # img = cv2.imread('frame', cv2.IMREAD_COLOR)[200,200,[2,1,0]]
            # print(img)

            continue

        elif frame_number >= stop_frame:
            # Display the resulting frame
            overallavg = img / length
            # print(overallavg)
            cv2.imshow('Frame', frame)
            # image_filtering(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print(frame_number)
                image_filtering(frame)

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

##########################################################################
# import numpy as np
# import cv2
#
# # the size of the image that fills the Color window
# COLOR_ROWS = 80
# COLOR_COLS = 250
#
# capture = cv2.VideoCapture("C:/Users/IMLAB/Desktop/video1.mp4")
# if not capture.isOpened():
#     raise RuntimeError('Error opening VideoCapture.')
#
# (grabbed, frame) = capture.read()
# snapshot = np.zeros(frame.shape, dtype=np.uint8)
# cv2.imshow('Snapshot', snapshot)
#
# colorArray = np.zeros((COLOR_ROWS, COLOR_COLS, 3), dtype=np.uint8)
# cv2.imshow('Color', colorArray)
#
#
# def on_mouse_click(event, x, y, flags, userParams):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         colorArray[:] = snapshot[y, x, :]
#         rgb = snapshot[y, x, [2, 1, 0]]
#
#         # From stackoverflow.com/questions/1855884/determine-font-color-based-on-background-color
#         luminance = 1 - (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
#         if luminance < 0.5:
#             textColor = [0, 0, 0]
#         else:
#             textColor = [255, 255, 255]
#
#         cv2.putText(colorArray, str(rgb), (20, COLOR_ROWS - 20),
#                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=textColor)
#         cv2.imshow('Color', colorArray)
#
# cv2.setMouseCallback('Snapshot', on_mouse_click)
#
# while True:
#     (grabbed, frame) = capture.read()
#     cv2.imshow('Video', frame)
#
#     if not grabbed:
#         break
#
#     keyVal = cv2.waitKey(1) & 0xFF
#     if keyVal == ord('q'):
#         break
#     elif keyVal == ord('t'):
#         snapshot = frame.copy()
#         cv2.imshow('Snapshot', snapshot)
#
# capture.release()
# cv2.destroyAllWindows()