import cv2 as cv
import numpy as np

# --------------------------- Exercise 3.1 --------------------------- #

cap = cv.VideoCapture('quadrotor.mp4')

# params for ShiTomasi corner detection as dictionary
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# params for lucas kanade optical flow as dictionary
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# create some random colors for the tracks
color = np.random.randint(0, 255, (100, 3))

# take first frame and find corners (= good features to track) in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# create a mask image to draw tracks on
mask = np.zeros_like(old_frame)

while(1):
    # extract single frames from video stream
    ret, frame = cap.read()

    # render grayscale image
    if ret:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        break

    # calculate image pyramids and optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # select best motion points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks of optical flow
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(dtype=np.int)
        c, d = old.ravel().astype(dtype=np.int)
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)

    # add mask with tracks on original frame
    img = cv.add(frame, mask)

    cv.imshow('frame', img)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Then update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
cap.release()  # yo! cite your sources!
