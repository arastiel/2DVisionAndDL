import cv2
import numpy as np

def gaussPyr(img):
    pyr = [img]
    while (True):
        img = cv2.pyrDown(img)
        pyr.append(img)
        shape = img.shape
        if min(shape[0], shape[1]) == 1:
            return pyr

def laplacePyr(pyr):
    lapyr = []
    for i in range(len(pyr)-1):
        high = pyr[i]
        low = cv2.resize(pyr[i+1], (high.shape[1], high.shape[0]), interpolation=cv2.INTER_LINEAR)
        lp = high - low
        # cv2.imshow(str(i), cv2.resize((lp+128)//2, (1024, 768)))
        lapyr.append(lp)
    lapyr.append(pyr[-1])
    return lapyr


def reconstructLaplace(lapyr, level):
    depth = len(lapyr)
    level = min(level, depth)
    acc = lapyr[-1]
    for i in range(1, depth-level):
        lp = lapyr[depth-i-1]
        acc = lp + cv2.resize(acc, (lp.shape[1], lp.shape[0]))
    return acc


def merge(a, b):
    return np.hstack((a[:, :(a.shape[1]//2)], b[:, b.shape[1]//2:]))


def mergeLaplace(a_lapyr, b_lapyr):
    depth = len(a_lapyr)
    acc = a_lapyr[-1] # merge(a_lapyr[-3], b_lapyr[-3])
    for i in range(1, depth):
        lp = merge(a_lapyr[depth - i - 1], b_lapyr[depth - i - 1])
        cv2.imshow(str(i), cv2.resize((lp+128)//2, (1024, 768)))
        acc = lp + cv2.resize(acc, (lp.shape[1], lp.shape[0]))
    return acc


horse = cv2.imread("images/horse.png")

zebra = cv2.imread("images/zebra.png")
horse_pyr = gaussPyr(horse)

zebra_pyr = gaussPyr(zebra)
horse_lapyr = laplacePyr(horse_pyr)

zebra_lapyr = laplacePyr(zebra_pyr)

cv2.imshow("lp", reconstructLaplace(zebra_lapyr, 0))
cv2.imshow("a", mergeLaplace(zebra_lapyr, horse_lapyr))

cv2.waitKey(0)
cv2.destroyAllWindows()