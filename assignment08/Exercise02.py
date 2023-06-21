import matplotlib.mlab
import matplotlib.pyplot as plt
import numpy as np


def z_func(x, y):
    return (x - 2) ** 2 / 4 + (y - 4) ** 2


if __name__ == '__main__':
    x = np.linspace(-50, 50, 1000)
    y = np.linspace(-50, 50, 1000)
    X, Y = np.meshgrid(x, y)
    Z = z_func(X, Y)

    plt.suptitle(r"$f (x,y) = \frac{(x - 2)^2}{4} + (y - 4)^2$")
    plt.xlabel("x-values")
    plt.ylabel("y-values")
    plt.contourf(X, Y, Z, cmap='YlOrRd')
    plt.colorbar()
    plt.contour(X, Y, Z, colors='purple')
    plt.show()

# 2.1
# In this case the zero isoline of f(x,y) describes the minimum of the elliptic paraboloid
# Mathematically speaking, it is the area where 0 <= f(x,y) < 500 # it's als the points where f(x, y) = 0, which is a point in this case
# 1p

# 2.2
# We calculate the gradient of f(x,y) by partially deriving with respects to x and y
# f_x' = 2x - 4
# f_y' = 2y - 8
# Pass x = 2.4 into f_x'(2.4) = 0.8
# Pass y = 4.2 into f_y'(4.2) = 0.4
# Which leads to gradient vector (0.8, 0.4) indicating the direction of maximum increase of f(x,y)
# 1p

# 2.3
# 0p
