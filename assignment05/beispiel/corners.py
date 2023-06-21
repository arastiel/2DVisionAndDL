import cv2
import numpy as np
# https://docs.opencv.org/master/dc/d0d/tutorial_py_features_harris.html

win = "corners"

img = cv2.imread("Lenna.png")
cv2.namedWindow(win)


def update(val):
    scale = cv2.getTrackbarPos("Scaling", win)/100
    rot = cv2.getTrackbarPos("Rotation", win)
    kernel = cv2.getTrackbarPos("Kernel", win)
    nsize = cv2.getTrackbarPos("Neighborhood size", win)*2+1
    ssize = cv2.getTrackbarPos("Sobel size", win)*2+1
    hfree = cv2.getTrackbarPos("Harris free", win)
    threshold = cv2.getTrackbarPos("Threshold", win)/100

    A = cv2.getRotationMatrix2D((img.shape[0]//2, img.shape[1]//2), angle=rot, scale=scale)
    transformed = cv2.warpAffine(img, A, dsize=(img.shape[0], img.shape[1]))

    dst = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), nsize, ssize, hfree)
    dst_transformed = cv2.cornerHarris(cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY), nsize, ssize, hfree)

    # Threshold for an optimal value, it may vary depending on the image.
    harris = np.copy(img)
    harris[dst < -threshold] = [255, 0, 0]

    harris_transformed = np.copy(transformed)
    harris_transformed[dst_transformed < -threshold] = [255, 0, 0]


    out = np.hstack((harris, harris_transformed))
    out = np.hstack((img, out))
    cv2.imshow("corners", out)


cv2.createTrackbar("Rotation", win, 0, 360, update)
cv2.createTrackbar("Scaling", win, 100, 200, update)
cv2.createTrackbar("Kernel", win, 1, 5, update)
cv2.createTrackbar("Neighborhood size", win, 1, 5, update)
cv2.createTrackbar("Sobel size", win, 1, 5, update)
cv2.createTrackbar("Harris free", win, 50, 100, update)
cv2.createTrackbar("Threshold", win, 50, 100, update)
update(0)

cv2.waitKey(0)
cv2.destroyAllWindows()