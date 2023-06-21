import cv2
import numpy as np
from copy import deepcopy

LEFT, TOP, RIGHT, BOTTOM = None, None, None, None
CURR_IMG = None
CAP = cv2.VideoCapture("vsauce_ahoy.mp4")
WIDTH = CAP.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = CAP.get(cv2.CAP_PROP_FRAME_HEIGHT)


def selectRectangle(evt, x, y, flags, userdata):
    """
    Callback that handles mousebutton events. Handles selection of the rectangle. Global vars LEFT, TOP, RIGHT, BOTTOM
    will be not None if the selection was completed successfully.
    """
    global LEFT, TOP, RIGHT, BOTTOM, CURR_IMG
    # Start template selection
    if evt == cv2.EVENT_LBUTTONDOWN:
        LEFT = x
        TOP = y
        RIGHT = None
        BOTTOM = None
        return
    # While dragging: show current selected region
    elif evt == cv2.EVENT_MOUSEMOVE and LEFT is not None and TOP is not None and RIGHT is None and BOTTOM is None:
        # Draw rectangle
        img = deepcopy(CURR_IMG)
        cv2.rectangle(img, (LEFT, TOP), (x, y), color=(255, 0, 0), thickness=2)
        cv2.imshow("bayesianColorSegmentation", img)
        return
    # End template selection and do matching
    elif evt == cv2.EVENT_LBUTTONUP:
        # Safety check in case Mouse is already down before the callback is registered
        if LEFT is None or TOP is None:
            return
        else:
            # Reset if selection has no area or is too small
            if abs(LEFT - x) * abs(TOP - y) < 400:
                cv2.imshow("bayesianColorSegmentation", CURR_IMG)
                LEFT, TOP, RIGHT, BOTTOM = None, None, None, None
                return
            else:
                # Set parameters of selected rectangle
                if x < LEFT:
                    LEFT, RIGHT = x, LEFT
                else:
                    RIGHT = x
                if y < TOP:
                    TOP, BOTTOM = y, TOP
                else:
                    BOTTOM = y
                return


def calculateCondProb(pixel_matrix):
    """
    :param pixel_matrix: ndarray of shape (height, width, 3) and dtype uint8 (each pixel has RGB colour channel)
    :return: ndarray of shape (256, 256) and dtype float where the element at index (i, j) is the probability of
    observing the rg color space value r=(i/256), g=(j/256) under the condition that the pixel is in the given
    pixel_matrix.
    """
    prob_matrix = np.zeros(shape=(256, 256), dtype=np.float32)
    sum_per_pixel = pixel_matrix.sum(axis=2)
    # Divide each color by sum of color values of its pixel, do not divide (i.e. return 0) if sum is 0
    pixel_matrix = pixel_matrix.astype(np.float32)
    pixel_matrix = np.divide(
        pixel_matrix, sum_per_pixel[:, :, np.newaxis], out=pixel_matrix, where=(sum_per_pixel[:, :, np.newaxis] != 0)
    )
    # Create histogram on 256 * 256 grid of each rg color mix (insensitive to overall intensity in original image)
    pixel_matrix = (pixel_matrix * 255).astype(np.uint8)
    for row in pixel_matrix:
        for el in row:
            prob_matrix[el[2], el[1]] += 1
    # Normalize so probabilities sum to one
    prob_matrix = prob_matrix / (prob_matrix.shape[0] * prob_matrix.shape[1])
    return prob_matrix


def bayesianSegmentation(img, cond_prob, prior_prob):
    """

    :param img: Image with BGR color channel
    :param cond_prob: P(color|skin) - ndarray of shape (256, 256) and dtype float where the element at index (i, j) is
    probability of observing the rg color space value r=(i/256), g=(j/256) under the condition that the pixel is skin.
    :param prior_prob: Float value of P(skin)
    :return: ndarray of the same shape as img and dtype uint8, picture where pixels segmented as skin are white and all
    others are black
    """
    # Intensity normalization
    sum_per_pixel = img.sum(axis=2)
    pixel_matrix = img.astype(np.float32)
    pixel_matrix = np.divide(
        img, sum_per_pixel[:, :, np.newaxis], out=pixel_matrix, where=(sum_per_pixel[:, :, np.newaxis] != 0)
    )
    pixel_matrix = (pixel_matrix * 255).astype(np.uint8)
    # Look up cond probabilities (very slow but hey - it's python...)
    lookup = lambda x: cond_prob[x[2], x[1]]
    p = np.apply_along_axis(lookup, 2, pixel_matrix) * prior_prob
    # White pixel wherever P(skin|color) > P(not skin|color)
    return np.full(p.shape, 255, dtype=np.uint8) * (p > (1 / 256**2 * (1 - prior_prob)))


def main():
    global LEFT, RIGHT, TOP, BOTTOM, CURR_IMG, CAP
    cv2.namedWindow("bayesianColorSegmentation")

    # Show instructions
    ctrls = cv2.imread("controls_exercise1.png")
    cv2.imshow("bayesianColorSegmentation", ctrls)
    cv2.waitKey(0)

    # Start selection of face
    while True:
        ret, CURR_IMG = CAP.read()
        # Start clip again if it is over before selection was done
        if not ret:
            CAP.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            cv2.imshow("bayesianColorSegmentation", CURR_IMG)
            key = cv2.waitKey(20)
            # If any key was pressed
            if key > -1:
                cv2.setMouseCallback("bayesianColorSegmentation", selectRectangle)
                cv2.waitKey(0)
                # If user has selected a valid region go to next phase
                if LEFT is not None and RIGHT is not None and TOP is not None and BOTTOM is not None:
                    break

    # Calculate P(color|skin) and P(skin)
    selection = CURR_IMG[TOP:BOTTOM, LEFT:RIGHT]
    cond_prob = calculateCondProb(selection)
    prior_prob = selection.shape[0] * selection.shape[1] / (CURR_IMG.shape[0] * CURR_IMG.shape[1])

    # Show segmentation on live video
    CAP.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret = True
    while ret:
        ret, img = CAP.read()
        # As the entire process is done in CPU and contains an expensive lookup operation, this will
        # have low framerate (~ 2 fps)
        seg_img = bayesianSegmentation(img, cond_prob, prior_prob)
        cv2.imshow("bayesianColorSegmentation", seg_img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()