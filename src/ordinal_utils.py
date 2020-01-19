import h5py
import numpy as np


def ordinal_thresholds_uniform(beta, n_classes, alpha=0):
    """
    Returns a list of thresholds for ordinal regression given output space [alpha, beta] and number of classes.
    The space is uniform.
    :param beta: Maximum value of output space.
    :param alpha: Minimum value of output space (default = 0).
    :param n_classes: Number of classes (should equal output of last convolution, default = 100)
    :return: List of thresholds for classes, including the left edge (=alpha) for the first class.
    """
    thresholds = [alpha]
    for i in range(1, n_classes):
        thresholds.append(alpha + (beta - alpha) / n_classes * i)

    thresholds.append(beta)
    return thresholds


def ordinal_thresholds_exp(beta, n_classes, alpha=0, eps=1):
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


def ordinal_thresholds_power(beta, n_classes, alpha=0, power=2):
    """
    Returns a list of thresholds for ordinal regression given output space [alpha, beta] and number of classes.
    The space depends on a power-law.
    :param beta: Maximum value of output space.
    :param alpha: Minimum value of output space (default = 0).
    :param n_classes: Number of classes (should equal output of last convolution, default = 100)
    :param power: Exponent used in the power law. When p=1, equal to uniform.
    :return: List of thresholds for classes, including the left edge (=alpha) for the first class.
    """
    thresholds = [alpha]
    for i in range(1, n_classes):
        thresholds.append(alpha + ((i / n_classes) ** power) * (beta - alpha))

    thresholds.append(beta)
    return thresholds


def depth_to_ordinal(matrix, thresholds):
    """
    :param matrix: (N, H, W) depth matrix.
    :return: a (N, H, W, C) matrix with the corresponding ordinal labels.
    """
    ordinal_labels = np.zeros(shape=matrix.shape + (len(thresholds) - 1,), dtype=int)
    for i, c in enumerate(thresholds[1:]):
        membership_matrix = matrix > c
        if np.any(membership_matrix):
            ordinal_labels[..., i] += membership_matrix
        else:
            break
    return ordinal_labels


def mean_backtranslation(y_pred, thresholds):
    """

    :param y_pred: (N,H,W,C) matrix.
    :param thresholds:
    :return: (N,H,W) depth matrix.
    """
    n, h, w, c = y_pred.shape
    backtranslated_depth = np.zeros((n, h, w), dtype=np.float32)
    ordinal_class = np.sum(y_pred >= 0.5, axis=-1)
    for i in range(n):
        for y in range(h):
            for x in range(w):
                class_idx = ordinal_class[i, y, x]
                backtranslated_depth[i, y, x] = (thresholds[class_idx] + thresholds[class_idx + 1]) / 2

    return backtranslated_depth


def uniform_backtranslation(y_pred, thresholds):
    """

    :param y_pred: (N,H,W,C) matrix.
    :param thresholds:
    :return: (N,H,W) depth matrix.
    """
    n, h, w, c = y_pred.shape
    backtranslated_depth = np.zeros((n, h, w), dtype=np.float32)
    ordinal_class = np.sum(y_pred >= 0.5, axis=-1)
    for i in range(n):
        for y in range(h):
            for x in range(w):
                class_idx = ordinal_class[i, y, x]
                p = y_pred[i, y, x, class_idx]
                backtranslated_depth[i, y, x] = (1 - p) * thresholds[class_idx] + p * thresholds[class_idx + 1]

    return backtranslated_depth


def relative_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (y_true + np.finfo(np.float32).tiny))


def train_generator(hdf_archive_path, batch_size=4):
    with h5py.File(hdf_archive_path, 'r') as archive:
        X = archive['X_train']
        y = archive['y_train_ordinal']
        assert len(X) == len(y)
        i = 0
        while i < len(X):
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size


def test_generator(hdf_archive_path, batch_size=4):
    with h5py.File(hdf_archive_path, 'r') as archive:
        X = archive['X_test']
        y = archive['y_test_numerical']
        assert len(X) == len(y)
        i = 0
        while i < len(X):
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size


def val_generator(hdf_archive_path, batch_size=4):
    with h5py.File(hdf_archive_path, 'r') as archive:
        X = archive['X_test']
        y = archive['y_test_ordinal']
        assert len(X) == len(y)
        i = 0
        while i < len(X):
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size

