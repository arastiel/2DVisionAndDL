import cv2 as cv
import numpy as np

image1 = cv.imread("Church/church_left.png")
image2 = cv.imread("Church/church_right.png")

grosse = 20

gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray1 = np.float32(gray1)
corners1 = cv.goodFeaturesToTrack(gray1, 30, 0.1, 50)

gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
gray2 = np.float32(gray2)
corners2 = cv.goodFeaturesToTrack(gray2, 30, 0.1, 50)

for i in corners1:
    image1 = cv.circle(image1, tuple(i[0]), 5, [0, 0, 255])

for i in corners2:
    image2 = cv.circle(image2, tuple(i[0]), 5, [0, 0, 255])


columns1 = np.shape(image1)[0]
rows1 = np.shape(image1)[1]

columns2 = np.shape(image2)[0]
rows2 = np.shape(image2)[1]

templates1 = []
templates2 = []


for i in corners1:
    if grosse < (int(i[0, 0])) < (columns1 - grosse) and grosse < (int(i[0, 1])) < (rows1 - grosse):
        templates1.append([image1[(int(i[0, 0]))-grosse:(int(i[0, 0]))+grosse, (int(i[0, 1]))-grosse:(int(i[0, 1]))+grosse], (int(i[0, 0])), (int(i[0, 1]))])

for i in corners2:
    if grosse < (int(i[0, 0])) < (columns2 - grosse) and grosse < (int(i[0, 1])) < (rows2 - grosse):
        templates2.append([image2[(int(i[0, 0]))-grosse:(int(i[0, 0]))+grosse, (int(i[0, 1]))-grosse:(int(i[0, 1]))+grosse], (int(i[0, 0])), (int(i[0, 1]))])


stacked_img_ncc = np.hstack((image1, image2))
stacked_img_ssd = np.hstack((image1, image2))

for i in templates1:
    for j in templates2:
        res = cv.matchTemplate(j[0], i[0], cv.TM_CCORR_NORMED)
        threshold = 0.9
        if res >= threshold:
            print("HIER")
            cv.line(stacked_img_ncc, (i[1], i[2]), (j[1]+rows1, j[2]), [255,0,0])
            break

for i in templates1:
    for j in templates2:
        res = cv.matchTemplate(j[0], i[0], cv.TM_SQDIFF)
        threshold = 900000  # ???
        if res >= threshold:
            print("HIER")
            cv.line(stacked_img_ssd, (i[1], i[2]), (j[1]+rows1, j[2]), [255,0,0])
            break





cv.imshow("NCC method", stacked_img_ncc)
cv.imshow("SSD method", stacked_img_ssd)

cv.waitKey(0)  # "Explain the difference between Harris and goodFeaturesToTrack" fehlt (-1)