import cv2 as cv
import numpy as np

img = cv.imread('test.png', 0)

cv.namedWindow('test')

def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=1000):

    # m = height, n = width
    # m = len(img)
    # n = len(img[0])

    m, n = np.shape(im)

    U = U_init
    Px = im
    Py = im
    error = 1

    while (error > tolerance):
        Uold = U

        GradUx = np.roll(U, -1, axis=1) - U
        GradUy = np.roll(U, -1, axis=0) - U

        PxNew = Px + (tau/tv_weight) * GradUx
        PyNew = Py + (tau/tv_weight) * GradUy
        #print(PxNew)
        NormNew = np.maximum(1, np.sqrt(PxNew**2 + PyNew**2))

        #print(NormNew)

        Px = PxNew / NormNew
        Py = PyNew / NormNew

        RxPx = np.roll(Px, 1, axis=1)
        RyPy = np.roll(Py, 1, axis=0)

        DivP = (Px - RxPx) + (Py - RyPy)
        U = im + tv_weight * DivP

        error = np.linalg.norm(U-Uold)/np.sqrt(n*m)

    return U, im-U


denoised = denoise(img, np.zeros_like(img))

cv.imshow('test', np.hstack((denoised[0], denoised[1]))) # 1 Punkt,  das bild passt nicht

cv.waitKey(0)

cv.destroyAllWindows()