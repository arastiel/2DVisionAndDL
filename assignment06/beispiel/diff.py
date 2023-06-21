import cv2
import numpy as np


def nothing(x):
    pass


vid = cv2.VideoCapture(0)

cv2.namedWindow("win")
img0 = None
img1 = None
img2 = None

cv2.createTrackbar("Idiff or Iand", "win", 0, 1, nothing)
cv2.createTrackbar("s", "win", 20, 255, nothing)

while True:
    ret, img = vid.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if ret:
        # update buffer
        img0, img1, img2 = img1, img2, img

        # if buffer full
        if img0 is not None and img1 is not None and img2 is not None:
            if cv2.getTrackbarPos("Idiff or Iand", "win") == 0:
                diff = cv2.absdiff(img2, img1)
                s = cv2.getTrackbarPos("s", "win")
                diff[diff < s] = [0]
                diff[diff > s] = [255]
                cv2.imshow("win", diff)
            else:
                diff21 = cv2.absdiff(img2, img1)
                s = cv2.getTrackbarPos("s", "win")
                diff21[diff21 < s] = [0]
                diff21[diff21 > s] = [255]

                diff10 = cv2.absdiff(img1, img0)
                diff10[diff10 < s] = [0]
                diff10[diff10 > s] = [255]

                diff = cv2.multiply(diff21, diff10)/255

                cv2.imshow("win", diff)
    if cv2.waitKey(1) != -1:
        break

vid.release()
cv2.destroyAllWindows()
