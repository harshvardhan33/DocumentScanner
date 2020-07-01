# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:15:19 2020

@author: harshvardhan
"""

import numpy as np 
import cv2
import imutils

img = cv2.imread("test.jpg")
orig = img.copy()

if img.shape[0]>1500 and img.shape[1]>1200:

    scale_percent = 20
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Resized image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





cv2.imshow("Original Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayImageBlur = cv2.GaussianBlur(grayImage,(3,3),0)
edgedImage = cv2.Canny(grayImageBlur, 100, 100, 3)
cv2.imshow("gray", grayImage)
cv2.imwrite("gray.jpg",grayImage)
cv2.imshow("grayBlur", grayImageBlur)
cv2.imwrite("grayImageBlur.jpg",grayImageBlur)
cv2.imshow("Edge Detected Image", edgedImage)
cv2.imwrite("canny.jpg",edgedImage)
cv2.waitKey(0) 
cv2.destroyAllWindows()

allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
allContours = imutils.grab_contours(allContours)
# descending sort contours area and keep top 1
allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]
# approximate the contour
perimeter = cv2.arcLength(allContours[0], True) 
ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)
temp = ROIdimensions.copy()
cv2.drawContours(img, [ROIdimensions], -1, (0,255,0), 2)
cv2.imshow("Contour Outline", img)
cv2.imwrite("Contour.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()



ROIdimensions = ROIdimensions.reshape(4,2)
rect = np.zeros((4,2), dtype="float32")
s = np.sum(ROIdimensions, axis=1)
rect[0] = ROIdimensions[np.argmin(s)]
rect[2] = ROIdimensions[np.argmax(s)]
diff = np.diff(ROIdimensions, axis=1)
rect[1] = ROIdimensions[np.argmin(diff)]
rect[3] = ROIdimensions[np.argmax(diff)]
(tl, tr, br, bl) = rect

widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
maxWidth = max(int(widthA), int(widthB))
heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
    [0,0],
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight-1],
    [0, maxHeight-1]], dtype="float32")

transformMatrix = cv2.getPerspectiveTransform(rect, dst)
scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))
cv2.imshow("Scaned",scan)
cv2.imwrite("WarpPerspective.jpg",scan)
cv2.waitKey(0)
cv2.destroyAllWindows()



scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
cv2.imshow("scanGray", scanGray)
cv2.waitKey(0)
cv2.destroyAllWindows()


thresh1 = cv2.adaptiveThreshold(scanGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 199, 17)
thresh2 = cv2.adaptiveThreshold(scanGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 199, 17) 

cv2.imshow("Fthresh", thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Fthresh1", thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()






from skimage.filters import threshold_local
# increase contrast incase its document
T = threshold_local(scanGray, 9, offset=8)
scanBW = (scanGray > T).astype("uint8") * 255
# display final high-contrast image
cv2.imshow("scanBW", scanBW)
cv2.waitKey(0)
cv2.destroyAllWindows()