import numpy as np
import cv2 as cv
import glob
import os

# ------------------- Converters -------------------- #


def vid2img(vid):
    vidcap = cv.VideoCapture(vid)
    success, image = vidcap.read()

    make_dir("vid_frames")
    path = "./testdata/vid_frames"

    vid_frames = []
    count = 0
    while success:
        # write images into separate folder
        if count < 10:
            cv.imwrite(os.path.join(path, "frame00%d.jpg" % count), image)
            vid_frames.append(cv.imread("./testdata/vid_frames/frame00%d.jpg" % count))
        elif 10 < count < 100:
            cv.imwrite(os.path.join(path, "frame0%d.jpg" % count), image)
            vid_frames.append(cv.imread("./testdata/vid_frames/frame0%d.jpg" % count))
        elif count >= 100:
            cv.imwrite(os.path.join(path, "frame%d.jpg" % count), image)
            vid_frames.append(cv.imread("./testdata/vid_frames/frame%d.jpg" % count))

        success, image = vidcap.read()
        print("Extracting frame", count)
        count += 1

    return vid_frames


def img2vid():
    img_array = []
    for filename in glob.glob('./testdata/vid_frames_gray/*.jpg'):
        img = cv.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv.VideoWriter('diff_vid.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def make_dir(title):
    try:
        os.mkdir("./testdata/" + title)
    except OSError:
        print("Creation of the directory %s failed" % "./testdata/" + title)
    else:
        print("Successfully created the directory %s " % "./testdata/" + title)

# ------------------ Exercise 1 a) ------------------ #


img1 = cv.imread("./testdata/waldo1.png")
img2 = cv.imread("./testdata/waldo2.png")


def img_diff(img1, img2, sigma):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    tmp = np.subtract(gray1, gray2) # missing absolute difference (-0.5)
    tmp[tmp > sigma] = 255
    tmp[tmp <= sigma] = 0
    return tmp

# testing
tmp1 = img_diff(img1, img2, 1)
cv.imshow("Exercise 1 a)", tmp1)
cv.waitKey(0)

# ------------------ Exercise 1 b) ------------------ #


def img_diff2(vid, sigma):
    if len(vid) < 3:
        raise TypeError("too little images")

    diff_vid = []
    height, width, channels = vid[0].shape
    count = 0
    make_dir("vid_frames_gray")

    for i in range(1, len(vid)-1):
        diff1 = img_diff(vid[i-1], vid[i], sigma)
        diff2 = img_diff(vid[i], vid[i+1], sigma)

        combined = np.zeros(shape=(height, width))
        for x in range(height):
            for y in range(width):
                if diff1[x][y] == 255 and diff2[x][y] == 255:
                    combined[x][y] = 255
        diff_vid.append(combined)
        if count < 10:
            cv.imwrite(os.path.join("./testdata/vid_frames_gray/", "frame00%d.jpg" % count), combined)
        elif 10 < count < 100:
            cv.imwrite(os.path.join("./testdata/vid_frames_gray/", "frame0%d.jpg" % count), combined)
        elif count >= 100:
            cv.imwrite(os.path.join("./testdata/vid_frames_gray/", "frame%d.jpg" % count), combined)
        print("Rendering differential frame", count)
        count += 1


# testing
test_vid = vid2img("./testdata/sample_vid.mp4")
img_diff2(test_vid, 100)
img2vid()


"""The advantage of img_diff2 over img_diff1 is that it also carries information 
about the object's direction of movement"""  # how? where is it stored?
# diff2 gets rid of the doubling artifact of simple diff

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
