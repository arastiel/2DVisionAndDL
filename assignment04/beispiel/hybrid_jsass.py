import cv2
import numpy as np

# load images, get image height and width, set sigma
i1 = cv2.imread('mm.png')
i2 = cv2.imread('jfk.png')
(h, w) = i1.shape[:2]
sigma = 5.5


# get filter size from sigma, invert expression from
#   https://huningxin.github.io/opencv_docs/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
# where it states
#   sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
size = int((20 * sigma + 10)/3 + 1)
if not size % 2:
    size += 1

# fill filter with coefficients
g = np.zeros((h, w))
padding_h = (h - size) // 2
padding_w = (w - size) // 2

# compute value for actual filter region as given in assignment sheet
for x in range(size):
    for y in range(size):
        dist = (x - size // 2) ** 2 + (y - size // 2) ** 2  # dist is computed w.r.t. center of image
        g[padding_h + y][padding_w + x] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-dist / (2 * sigma ** 2))
g = g / np.sum(g)

# low pass target images
l1 = np.zeros(i1.shape)
l2 = np.zeros(i2.shape)

# kernel in frequency space
G = np.fft.fft2(g)

# for every channel compute low pass images using FFT, then apply filter and transform back.
# NB: scaling and shifting is important here, abs removes complex and a warning as bonus
for channel in range(3):
    I1_channel = np.fft.fft2(i1[:, :, channel])
    l1[:, :, channel] = np.abs(np.fft.ifftshift(np.fft.ifft2(I1_channel * G) / 255))
    I2_channel = np.fft.fft2(i2[:, :, channel])
    l2[:, :, channel] = np.abs(np.fft.ifftshift(np.fft.ifft2(I2_channel * G) / 255))

# compute high pass images as complement to low pass
h1 = i1 / 255 - l1
h2 = i2 / 255 - l2

# the actual hybrid image, is just the sum of low and high of i1 and i2 respectively
H = l1 + h2

# show, save
cv2.imshow("l1", np.abs(l1))
cv2.imshow("h2", np.abs(h2))
cv2.imshow("H", H)
cv2.imwrite('l1.png', l1 * 255)
cv2.imwrite('l2.png', l2 * 255)
cv2.imwrite('h1.png', h1 * 255 + 127)
cv2.imwrite('h2.png', h2 * 255 + 127)
cv2.imwrite('H.png', H * 255)

# done.
cv2.waitKey(0)
cv2.destroyAllWindows()
# 10 Punkte