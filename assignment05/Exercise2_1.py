import cv2 as cv
import numpy as np

image1 = cv.imread("Church/church_left.png")
image2 = cv.imread("Church/church_right.png")

gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray1 = np.float32(gray1)
corners1 = cv.cornerHarris(gray1, 5, 5, 0.2)

image1[corners1 > 0.01 * corners1.max()] = [0, 0, 255]

gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
gray2 = np.float32(gray2)
corners2 = cv.cornerHarris(gray2, 5, 5, 0.2)

image2[corners2 > 0.01 * corners2.max()] = [0, 0, 255]

data1 = np.argwhere(image1 == [0, 0, 255])
data1 = np.delete(data1, 2, 1)
new_data = [tuple(row) for row in data1]
corner_points1 = (np.unique(new_data, axis=0))

shape1 = np.shape(image1)
rows1 = shape1[0]
columns1 = shape1[1]

data2 = np.argwhere(image2 == [0, 0, 255])
data2 = np.delete(data2, 2, 1)
new_data = [tuple(row) for row in data1]
corner_points2 = (np.unique(new_data, axis=0))

shape2 = np.shape(image2)
rows2 = shape2[0]
columns2 = shape2[1]

grosse = 5

templates1 = []


for i in corner_points1:
    if grosse < i[0] < (rows1 - grosse) and grosse < i[1] < (columns1 - grosse):
        templates1.append([image1[i[0]-grosse:i[0]+grosse, i[1]-grosse:i[1]+grosse], i[0], i[1]])


templates2 = []
for i in corner_points2:
    if grosse < i[0] < (rows2 - grosse) and grosse < i[1] < (columns2 - grosse):
        templates2.append([image2[i[0]-grosse:i[0]+grosse, i[1]-grosse:i[1]+grosse], i[0], i[1]])

stacked_img_ncc = np.hstack((image1, image2))
stacked_img_ssd = np.hstack((image1, image2))

for i in templates1:
    for j in templates2:
        res = cv.matchTemplate(j[0], i[0], cv.TM_CCORR_NORMED)
        threshold = 0.90
        if res >= threshold:  # macht es sinn das eine corner mehrmals im anderen bild makiert werden kann? (-1)
            cv.line(stacked_img_ncc, (i[2], i[1]), (j[2]+595, j[1]+5), [255,0,0])
            break

for i in templates1:
    for j in templates2:
        res = cv.matchTemplate(j[0], i[0], cv.TM_SQDIFF)
        threshold = 9000000
        if res >= threshold:
            cv.line(stacked_img_ssd, (i[2], i[1]), (j[2]+595, j[1]+5), [255,0,0])
            break

cv.imshow("NCC method", stacked_img_ncc)
cv.imshow("SSD method", stacked_img_ssd)
cv.waitKey(0)