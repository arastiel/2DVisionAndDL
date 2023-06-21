import cv2 as cv
import numpy as np

image = cv.imread("Lenna.png")

cv.namedWindow('Main', cv.WINDOW_FULLSCREEN)


image_center = tuple(np.array(image.shape[1::-1]) / 2)

def draw_img(x):
    #bild rotieren
    rot_mat = cv.getRotationMatrix2D(image_center, cv.getTrackbarPos("Rotation", "Main"), 1.0)
    new_image = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)

    #bild scaöem
    scale_width = int(new_image.shape[1] * cv.getTrackbarPos("Scale", "Main") / 100)
    scale_height = int(new_image.shape[0] * cv.getTrackbarPos("Scale", "Main") / 100)
    scale_dim = (scale_width, scale_height)
    new_image_2 = cv.resize(new_image, scale_dim, interpolation=cv.INTER_AREA) # warum nicht in RotationMatrix skalieren?

    #gaussianblur
    ksize = cv.getTrackbarPos("GaussianKernel", "Main")
    if ksize % 2 == 0:
        ksize = ksize+1

    new_image_result = cv.GaussianBlur(new_image_2, (ksize, ksize), 0)

    #image grayscale für corner Harris
    gray = cv.cvtColor(new_image_result, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    block_size = cv.getTrackbarPos("SobelNeighborSize", "Main")
    ksize_sobel = cv.getTrackbarPos("SobelKernel", "Main")
    if ksize_sobel % 2 == 0:
        ksize_sobel = ksize_sobel+1

    k = cv.getTrackbarPos("HarrisFree", "Main")
    k = k/100
    if k == 0:
        k = 0.01

    corners = cv.cornerHarris(gray, block_size, ksize_sobel, k)

    threshold = cv.getTrackbarPos("Threshold", "Main")/100
    if threshold == 0:
        threshold = 0.01

    #markieren, wo threshold erlaubt
    new_image_result[corners>threshold*corners.max()] = [0, 0, 255]


    cv.imshow("Main", new_image_result)


cv.createTrackbar("Rotation", 'Main', 0, 360, draw_img)
cv.createTrackbar("Scale", "Main", 100, 150, draw_img)
cv.createTrackbar("GaussianKernel", "Main", 0, 100, draw_img)
cv.createTrackbar("SobelNeighborSize", "Main", 1, 10, draw_img)
cv.createTrackbar("SobelKernel", "Main", 1, 30, draw_img)
cv.createTrackbar("HarrisFree", "Main", 0, 100, draw_img)
cv.createTrackbar("Threshold", "Main", 0, 100, draw_img)


cv.imshow("Main", image)


while True:
    if cv.waitKey(1) & 0xFF == 27:
        break