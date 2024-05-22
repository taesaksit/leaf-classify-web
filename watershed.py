import cv2
import numpy as np


def detection(image):
    # read Gray
    image_bgr = cv2.imread(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)

    #  Color detection
    lower = np.array([30, 30, 30])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(image_hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    median = cv2.medianBlur(mask, 5)
    open = cv2.dilate(median, kernel)
    close = cv2.erode(open, kernel)

    # contours
    contours, hierachy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image_rgb.copy()

    for cnt in contours:

        if cv2.contourArea(cnt) < 10000 and cv2.contourArea(cnt) > 10:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_copy, (x, y), (x + w + 2, y + h + 2), (255, 0, 0), 1)

    return image_copy
