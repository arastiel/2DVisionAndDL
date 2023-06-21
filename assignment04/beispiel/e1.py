import numpy as np
import cv2


win = "window"

img = cv2.imread("koreanSigns.png")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
# mouse stuff

p1 = (0, 0)
p2 = (0, 0)
mx = img.shape[1]
my = img.shape[0]

marking = True
drawing = False

def mouse(event, x, y, flags, param):
    global p1, p2, drawing

    if event == cv2.EVENT_LBUTTONDOWN and marking:
        p = (max(0, min(x, mx)), max(0, min(y, my)))
        p1 = p
        p2 = p
        drawing = True
    if event == cv2.EVENT_MOUSEMOVE and drawing:
        p2 = (max(0, min(x, mx)), max(0, min(y, my)))

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow(win)
cv2.setMouseCallback(win, mouse)


# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
while(True):
    dst = cv2.resize(img, (img.shape[1], img.shape[0]))
    if marking:
        cv2.rectangle(dst, p1, p2, (255, 0, 0), thickness=2)
    else:
        for pt in zip(*loc[::-1]):
            cv2.rectangle(dst, pt, (pt[0]+crop.shape[1], pt[1]+crop.shape[0]), (0, 255, 0), 2)
        # cv2.rectangle(dst, p1, p2, (0, 255, 0), thickness=2)
        cv2.imshow("Crop", crop)
        cv2.imshow("Match", res)


    cv2.imshow(win, dst)
    if cv2.waitKey(5) != -1:
        if marking:
            marking = False
            p3 = (min(p1[0], p2[0]), min(p1[1], p2[1]))
            p4 = (max(p1[0], p2[0]), max(p1[1], p2[1]))
            p1 = p3
            p2 = p4
            crop = img[p1[1]:p2[1], p1[0]:p2[0]]
            p = (p1[0]+p2[0]//2, p1[1]+p2[1]//2)
            p3 = (p1[0]-p[0],p1[1]-p[1])
            p4 = (p2[0]-p[0],p2[1]-p[1])

            match = cv2.matchTemplate(img, crop, cv2.TM_CCORR_NORMED)
            res = cv2.matchTemplate(img, crop, cv2.TM_CCORR_NORMED)
            cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1)
            threshold = 0.9
            loc = np.where(res >= threshold)

        else:
            break

cv2.destroyAllWindows()



'''
tPoint = None
bPoint = None
cv2.cuda.DeviceInfo_
'''
