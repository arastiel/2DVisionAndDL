import cv2 as cv
import numpy as np


######################################################################
# 1. 4 Punkte anklicken, wohin es gemappt werden soll (destination)  #
# 2. 4 Punkte anklicken, was dorthin gemappt werden soll (source)    #
######################################################################

img = cv.imread('test.png')
cv.namedWindow('image')

pointset_dst = []  #saving pointset1
pointset_src = [0, 0, 0, 0]  #saving pointset2
moved = []
moving = False
index = 0

def onMouse(event, x, y, flags, param):
    global pointset_dst, pointset_src, moving, img, index
    
    if event == cv.EVENT_LBUTTONDOWN:
        if len(pointset_dst) < 4:
            cv.circle(img, (x, y), 5, (0, 255, 0), -1)
            pointset_dst.append((x, y))
            print("pointset1: ", pointset_dst)
        else:
            if 0 in pointset_src:
                #get the nearest point which hasn't been moved yet
                min_dist = np.inf
                for coord in pointset_dst:
                    if coord not in moved:
                        curr_dist = np.sqrt((x - coord[0]) ** 2 + (y - coord[1]) ** 2)
                        if curr_dist < min_dist:
                            min_dist = curr_dist
                            min_coord = coord
                            index = pointset_dst.index(min_coord)

                moved.append(min_coord)
                moving = True

    elif event == cv.EVENT_LBUTTONUP:
        if moving:
            moving = False

            #redraw new image with new selected position
            img2 = cv.imread("test.png")
            pointset_src[index] = (x,y)    #add new coordinate on same index as old coordinate
            print("pointset2: ", pointset_src)

            [cv.circle(img2, coord, 5, (0, 255, 0), -1) for coord in pointset_dst + [points for points in pointset_src if points != 0] if coord not in moved]
            img = img2


cv.setMouseCallback('image', onMouse)


while True:     #loop until all points are set
    cv.imshow("image", img)
    if cv.waitKey(10) & 0xFF == 27:
        #press escape to break to close it
        break
    if 0 not in pointset_src:
        cv.imshow("image", img)
        break

#save pointsets as np.arrays for findHomography
pos1 = np.array(pointset_dst)
pos2 = np.array(pointset_src)
 
h, status = cv.findHomography(pos2, pos1)
im_out = cv.warpPerspective(img, h, (img.shape[1], img.shape[0])) # 10 Punkte

cv.imshow("image", img)
cv.imshow("image_out", im_out)
cv.waitKey(0)
cv.destroyAllWindows()
