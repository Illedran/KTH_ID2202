import numpy as np


def ordinal_thresholds_exp(beta, alpha=0, n_classes=100, eps=1):
    """
    Returns a list of thresholds for ordinal regression given output space [alpha, beta] and number of classes.
    The space is exponential.
    :param beta: Maximum value of output space.
    :param alpha: Minimum value of output space (default = 0).
    :param n_classes: Number of classes (should equal output of last convolution, default = 100).
    :param eps: Smoothing parameter (from paper, default = 1). The bigger, the less rapid growth.
    :return: List of thresholds for classes, including the left edge (=alpha) for the first class.
    """
    eps = eps - alpha
    a_star = alpha + eps
    b_star = beta + eps
    thresholds = [alpha]
    for i in range(1, n_classes):
        thresholds.append(np.power(np.e, np.log(a_star) + (np.log(b_star / a_star) * i) / n_classes) - eps)

    thresholds.append(beta)
    return thresholds


def depth_to_ordinal(matrix, thresholds):
    """
    :param matrix: (N, H, W, 1) depth matrix.
    :return: a (N, H, W) matrix with the corresponding ordinal label.
    """
    ordinal_labels = np.empty_like(matrix, dtype=int)
    prev = thresholds[0]
    for i, c in enumerate(thresholds[1:]):
        ordinal_labels[np.bitwise_and(matrix >= prev, matrix < c)] = i
        prev = c
    return ordinal_labels.squeeze()

def pixel_ordinal_loss(pixel):
    pass

beta = 100
K = 25
Y = np.random.uniform(high=beta, size=(10, 9, 12, 1))
thresh = ordinal_thresholds_exp(beta=beta, n_classes=K)

Y_hat = depth_to_ordinal(Y, thresh)