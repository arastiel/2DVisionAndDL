import numpy as np


# In the max pooling layer during forward propagation, only the maximum value of a window is pushed to the next layer.
# Because of this, values besides the maximum of a window don't play any role in the output value, so during backward
# propagation their gradient should be 0. The value for the maximum value is the value of the derivative received from
# upper layers.

def max_pool_back(d_out, cache):
    """
    Compute the backward propagation of the max pooling layer
    :param d_out: Upstream derivative
    :param cache: Tuple of (x, pool_param), as in the forward pass
    :return: dx - Gradient with respect to x
    """

    x, pool_param = cache
    stride = 2
    p_height = 2
    p_width = 2
    H, W = d_out.shape  # height and width of derivative
    dx = np.zeros(x.shape)

    # Calculate dx(mask * d_out)
    # mask = matrix with 1 for the max value and 0 otherwise
    for h in range(H):      # Slide vertically
        for w in range(W):  # Slide horizontally
            # Get window and calculate mask
            x_pool = x[h*stride:h*stride+p_height, w*stride:w*stride+p_width]
            mask = (x_pool == np.max(x_pool))
            # Calculate mask * d_out
            dx[h*stride:h*stride+p_height, w*stride:w*stride+p_width] = mask * d_out[h, w]

    return dx
