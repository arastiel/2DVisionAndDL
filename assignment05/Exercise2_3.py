import cv2 as cv
import numpy as np

img1 = cv.imread("./Church/church_left.png")
img2 = cv.imread("./Church/church_right.png")

img3 = np.concatenate((img1, img2), axis=1)

orb = cv.ORB_create()

kp1 = orb.detect(img1, None)
kp2 = orb.detect(img2, None)

kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

cv.drawKeypoints(img1, kp1, img1, color=(0, 0, 255), flags=0)
cv.drawKeypoints(img2, kp2, img2, color=(0, 0, 255), flags=0)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

cv.drawMatches(img1, kp1, img2, kp2, matches[:180], img3, flags=2)

cv.imshow("combined", img3)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
