import numpy as np
import math
import cv2

sqrt = math.sqrt(2)
mat = [[490,    -390,   -1500,   1300],
       [-590,    1400,   -600,   1300],
       [-0.5*sqrt, -0.3*sqrt, -0.4*sqrt, 5]]

P = np.array(mat)
print("P", P)
U, D, V = np.linalg.svd(P)
print("U", U)
print("D", D)
print("V", V)
V_c = np.asarray(V[3, :]).reshape((4, 1))
print("V_c", V_c/V_c[3])
print("--------------------")

# print(np.asarray([P[:, 1], P[:, 2], P[:, 3]]))
c1 = np.linalg.det([P[:, 1], P[:, 2], P[:, 3]])
# print(np.asarray([P[:, 0], P[:, 2], P[:, 3]]))
c2 = -np.linalg.det([P[:, 0], P[:, 2], P[:, 3]])
# print(np.asarray([P[:, 0], P[:, 1], P[:, 3]]))
c3 = np.linalg.det([P[:, 0], P[:, 1], P[:, 3]])
# print(np.asarray([P[:, 0], P[:, 1], P[:, 2]]))
c4 = -np.linalg.det([P[:, 0], P[:, 1], P[:, 2]])

c = np.asarray([[c1], [c2], [c3], [c4]])
print("c~", c)
print("c", c/c[3])

print("--------------------")

K, R, t, RX, RY, RZ, eulerAngles = cv2.decomposeProjectionMatrix(P)
print("K:", K)
print("R:", R)
print("t:", t/t[3])

print("--------------------")

print("res1", np.matmul(P, V_c), np.linalg.norm(np.matmul(P, V_c)))
print("res2", np.matmul(P, c), np.linalg.norm(np.matmul(P, c)))
print("res3", np.matmul(P, t), np.linalg.norm(np.matmul(P, t)))
