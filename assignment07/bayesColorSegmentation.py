import numpy as np
import argparse
import cv2
import time

# Sources used
# https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/

# Usage
# python3 bayesColorSegmentation.py --video <video_path>

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])

# keep looping over the frames in the video
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # for every frame calculate P(Skin)
    converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # use chromaticity space
    skin_mask = cv2.inRange(converted_frame, lower, upper)  # use Bayes theorem

    # count white pixel in skin_mask for P(Skin)
    prob_skin = np.count_nonzero(skin_mask == 255) / skin_mask.size  # using the generated mask for P(skin) doesnt make sense

    # show the skin in the image along with the mask
    # cv2.imshow("images", np.hstack([frame, skin_mask]))

    # show only skin mask
    cv2.imshow("original", skin_mask)
    time.sleep(.03)
    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
