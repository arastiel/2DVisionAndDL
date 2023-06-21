import cv2 as cv
import numpy as np
import glob

# ----------------------------------------------- Setup -------------------------------------------------------------- #
# termination criteria (accuracy)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points for chessboard (object points = corners), we are using 6*7 corners,
# because this is the number of inner corners on the chessboard
objp = np.zeros((6 * 7, 3), np.float32)
# fills array with points like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d point in real world space
image_points = []  # 2d points in image plane.

images = glob.glob('data/calibrationImagesCheckerboard/*.JPG')

# main loop to calculate corner points and image points
counter = 0  # how many images we look at
good_img = 0  # how many images are good for calibrating
for file_name in images:
    if good_img < 15:
        print("[#] Calculating corner/image points for image:", counter)
        image = cv.imread(file_name)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 8-bit-grayscale image

        # Find the chess board corners
        # third parameter of cv.findChessboardCorners is an OutputArray which is None in our context
        # ret = returns a non-zero value if all of the corners are found and they are placed in a certain order
        # The detected coordinates are approximate, and to determine their positions more accurately,
        # the function calls cornerSubPix
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            print(" [+] Image", counter, "is good for calibrating")
            good_img += 1
            object_points.append(objp)
            """ We need this refinement only for drawing
            # gray = image, corners , winSize = searchwindow, zeroZone = (-1, -1) indicates there is no such size
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            """
            image_points.append(corners)
        else:
            print(" [-] Image", counter, "isn't good for calibrating")
        counter += 1

    else:
        break

# -------------------------------------------- Calibration ----------------------------------------------------------- #
image = cv.imread('data/calibrationImagesCheckerboard/P1000588.JPG')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 8-bit-grayscale image

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
print("<--- Camera Matrix --->")
print(mtx)

# -------------------------------------------- Undistortion ---------------------------------------------------------- #
# main loop to undistort all images
count = 0
undist_images = glob.glob('data/undistortedImages/')
for file_name in images:
    print("undistort image", count)
    img = cv.imread(file_name)
    h, w = img.shape[:2]
    # refine cameramatrix with alpha = 0.4 (all pixels are retained with some extra black images)
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.4, (w, h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    print('data/undistortedImages/' + file_name[-12:])
    cv.imwrite('data/undistortedImages/' + file_name[-12:], dst)
    count += 1
