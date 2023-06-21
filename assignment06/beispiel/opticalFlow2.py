import cv2
import numpy as np


def nothing(x):
    pass

def downsamplePair(img0, img1):
    while img0.shape[0] > 200:
        img0 = cv2.pyrDown(img0)
        img1 = cv2.pyrDown(img1)
    return img0, img1

vid = cv2.VideoCapture(0)

cv2.namedWindow("win")
img0 = None
img1 = None
img2 = None

cv2.createTrackbar("s", "win", 80, 255, nothing)

kernel_x = np.array([[-1., 1.], [-1., 1.]])
kernel_y = np.array([[-1., -1.], [1., 1.]])
kernel_t = np.array([[1., 1.], [1., 1.]])


while True:
    ret, img = vid.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    s = cv2.getTrackbarPos("s", "win")/255

    if ret:
        # update buffer
        img0, img1, img2 = img1, img2, img

        # if buffer full
        if img0 is not None and img1 is not None and img2 is not None:
            dsimg1, dsimg2 = downsamplePair(img1, img2)
            # derivative of old image
            x = cv2.filter2D(dsimg1, cv2.CV_32F, kernel_x)
            y = cv2.filter2D(dsimg1, cv2.CV_32F, kernel_y)
            # datatype for new image
            dsimg2 = cv2.subtract(dsimg2, dsimg1, dtype=cv2.CV_32F)

            # solve [[x1, ..., xn], [y1, ..., yn]]*[u, v] = dsimg2
            ret, vec = cv2.solve(np.vstack((np.ravel(x), np.ravel(y))).transpose(), np.ravel(dsimg2).transpose(), flags=cv2.DECOMP_NORMAL)

            sensimage = cv2.resize(dsimg2/255, (1280, 720))
            response = cv2.resize(vec, (100, sensimage.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("win", np.hstack((sensimage, response)))

            direction = ""
            if vec[0] < -s:
                direction += "left "
            if vec[0] > s:
                direction += "right "
            if vec[1] < -s:
                direction += "down "
            if vec[1] > s:
                direction += "up "
            if direction == "":
                print("no motion")
            else:
                print(direction)

    if cv2.waitKey(1) != -1:
        break

vid.release()
cv2.destroyAllWindows()
