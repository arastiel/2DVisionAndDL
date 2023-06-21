import cv2 as cv
import numpy as np

image = cv.imread("koreanSigns.png")
img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

rect_x1 = 0
rect_y1 = 0
rect_x2 = 0
rect_y2 = 0
w = 0
h = 0


def draw_rectangle(event, x, y, flags, param):
    global rect_x1, rect_y1, rect_x2, rect_y2, w, h

    if event == cv.EVENT_LBUTTONDOWN:
        rect_x1 = x
        rect_y1 = y
        print("x1", rect_x1, "y1", rect_y1)
    elif event == cv.EVENT_LBUTTONUP:
        rect_x2 = x
        rect_y2 = y
        print("x2", rect_x2, "y2", rect_y2)
        cv.rectangle(img, pt1=(rect_x1, rect_y1), pt2=(rect_x2, rect_y2), color=(0, 255, 255), thickness=2,
                     lineType=cv.LINE_8)
        #
        w = abs(rect_x1 - rect_x2)
        h = abs(rect_y1 - rect_y2)
        template = create_template(rect_x1, rect_x2, rect_y1, rect_y2)
        res = norm_cross_cor(template)
        threshold = 0.9
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


def create_template(x1, x2, y1, y2):
    global img
    roi = img[y1:y2, x1:x2]
    return roi


def norm_cross_cor(template):
    res = cv.matchTemplate(img, template, cv.TM_CCORR_NORMED)
    return res


cv.namedWindow(winname="Normalized cross correlation")
cv.setMouseCallback("Normalized cross correlation", draw_rectangle)

while True:
    cv.imshow("Normalized cross correlation", img)
    if cv.waitKey(10) == 27:
        break

cv.destroyAllWindows()
# 5 Points