import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread('test.png', 0)

'''
für graubilder
img = cv.imread('test.png', 0)
'''

def draw_all(x):
    gaus_ksize = cv.getTrackbarPos('KernelSize(Gaus)', 'image')
    if gaus_ksize % 2 == 0:
        gaus_ksize = gaus_ksize+1

    sigma1 = cv.getTrackbarPos('Sigma1(Bil)', 'image')
    sigma2 = cv.getTrackbarPos('Sigma2(Bil)', 'image')
    diameter = cv.getTrackbarPos("Diameter(Bil)", 'image')

    median_ksize = cv.getTrackbarPos('KernelSize(Median)', 'image')
    if median_ksize % 2 == 0:
        median_ksize = median_ksize+1

    blurredGaus = cv.GaussianBlur(img, (gaus_ksize, gaus_ksize), 0)     # 1
    blurredBil = cv.bilateralFilter(img, diameter, sigma1, sigma2)      # 1
    blurredMed = cv.medianBlur(img, median_ksize)                       # 1


    cv.imshow("image", np.hstack((img, blurredGaus, blurredBil, blurredMed)))



def Exercise1():
    cv.namedWindow('image')

    blurredGaus = cv.GaussianBlur(img, (1, 1), 1)
    blurredBil = cv.bilateralFilter(img, 1, 1, 1)
    blurredMed = cv.medianBlur(img, 1)
    cv.imshow("image", np.hstack((img, blurredGaus, blurredBil, blurredMed)))

    cv.createTrackbar("KernelSize(Gaus)", 'image', 1, 25, draw_all)
    cv.createTrackbar("Diameter(Bil)", 'image', 1, 25, draw_all)
    cv.createTrackbar("Sigma1(Bil)", 'image', 1, 100, draw_all)
    cv.createTrackbar("Sigma2(Bil)", 'image', 1, 100, draw_all)
    cv.createTrackbar("KernelSize(Median)", 'image', 1, 25, draw_all)

    cv.waitKey(0)
    cv.destroyAllWindows()



def Exercise2():
    sobelx = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=3)
    sobely = cv.Sobel(img, cv.CV_8U, 0, 1, ksize=3)

    sobelMag = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)  # Not length

    plt.subplot(1, 4, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 2), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 3), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 4), plt.imshow(sobelMag, cmap='gray')
    plt.title('Length of image gradient'), plt.xticks([]), plt.yticks([])


    plt.show()


def Exercise3():
    canny = cv.Canny(img, 100, 200)

    sobelx = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=3)
    sobely = cv.Sobel(img, cv.CV_8U, 0, 1, ksize=3)

    sobelMag = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(canny, cmap='gray')
    plt.title('Canny'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(sobelMag, cmap='gray')
    plt.title('Length of the image gradient(sobel)'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    # uncomment subtask to be reviewed
    #Exercise1()    # 3
    #Exercise2()    # 1
    Exercise3()     # 1 not length of gradient, "Compare the result" requires text
