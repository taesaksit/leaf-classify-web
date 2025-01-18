import cv2
import numpy as np
from PIL import Image

def detection(path):
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image,  cv2.COLOR_BGR2RGB)  # แปลงเป็น RGB

    img_hsv = cv2.cvtColor(image_rgb , cv2.COLOR_RGB2HSV)
    lower = np.array([30,30,30])
    upper = np.array([90,255,255])
    mask = cv2.inRange(img_hsv , lower , upper)
    kernel = np.ones((3,3) , np.uint8)

    median = cv2.medianBlur(mask , 5)
    opening = cv2.dilate(median , kernel)
    closing = cv2.erode(opening , kernel)
    contours , hierachy = cv2.findContours(closing, mode=cv2.RETR_TREE , method=cv2.CHAIN_APPROX_SIMPLE)

    img_recog = image_rgb.copy()
    count = 0  
    for index , cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 10000 and cv2.contourArea(cnt) > 10:
            x,y,w,h = cv2.boundingRect(cnt)
            img_recog = cv2.rectangle(img_recog , (x,y) , (x+w,y+h) , (255,0,0) , 1)
            cv2.putText(img_recog, f'{index}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            count += 1  
    return img_recog, count



def detection2(image):
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
