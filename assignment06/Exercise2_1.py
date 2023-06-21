import cv2 as cv
import numpy as np, sys

A = cv.imread('../assignment04/horse.png')
B = cv.imread('../assignment04/zebra.png')

# -------------------- Exercise 2.1 -------------------- #


def create_pyramids(prev, nxt):
    # generate Gaussian pyramid for prev
    G = prev.copy()
    gp_prev = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gp_prev.append(G)

    # generate Gaussian pyramid for next
    G = nxt.copy()
    gp_nxt = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gp_nxt.append(G)

    # generate Laplacian Pyramid for prev
    lp_prev = [gp_prev[5]]
    for i in range(5, 0, -1):
        GE = cv.pyrUp(gp_prev[i])
        L = cv.subtract(gp_prev[i - 1], GE)
        lp_prev.append(L)

    # generate Laplacian Pyramid for next
    lp_nxt = [gp_nxt[5]]
    for i in range(5, 0, -1):
        GE = cv.pyrUp(gp_nxt[i])
        L = cv.subtract(gp_nxt[i - 1], GE)
        lp_nxt.append(L)

    return lp_prev, lp_nxt


a, b = create_pyramids(A, B)  # Points
print(len(a), len(b))

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()