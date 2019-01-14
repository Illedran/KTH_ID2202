import tensorflow as tf
from keras import backend as K


def Upsample2DToSize(input_tensor, size):
    """
    Custom Keras layer (using TF) that upsamples to a specific size instead of a factor.
    :param input_tensor: (N, H, W, C) tensor.
    :param size: (new_H, new_W).
    :return: (N, new_H, new_W, C) tensor.
    """
    return tf.image.resize_bilinear(input_tensor, size)


def OrdinalSoftmax(input_tensor, n_classes):
    """
    Custom Keras layer that computes probability of belonging to each ordinal class using a softmax on two values.
    (belong to class k or do not belong to class k for each class k).
    :param input_tensor:  (N, H, W, 2C) tensor.
    :return: (N, H, W, C) tensor.
    """
    class_probs = []
    for i in range(n_classes):
        is_in = K.exp(input_tensor[..., 2 * i + 1])
        is_out = K.exp(input_tensor[..., 2 * i])
        p_i = is_in / (is_in + is_out)
        class_probs.append(p_i)

    return K.stack(class_probs, axis=-1)


def ordinal_loss(y_true, y_pred):
    """
    Custom function to compute loss of DORN.
    :param true_classes: A (H,W,C) binary matrix that contains a vector of size C for the classes of pixel (H, W)
    of image N. The vector contains 1 up until the ordinal class of the data point.
    :param softmax_probs: A (H,W,C) matrix where each value corresponds to the probability of being
    in class C for pixel (H,W) of image N.
    :return: Loss of the network.
    """
    loss_on_prev_classes = K.sum(K.log(y_pred) * y_true, axis=-1)
    loss_on_next_classes = K.sum((1 - K.log(y_pred)) * (1 - y_true), axis=-1)

    return K.abs(K.mean(loss_on_prev_classes + loss_on_next_classes))

