import keras
from keras import backend as K


def DORN_ResNet50_NYUV2(input_shape=(257, 353, 3), n_classes=68):
    # (Dense) feature extractor originally ResNet36 in implementation, we use ResNet50 here
    feature_extractor = keras.applications.resnet50.ResNet50(input_shape=input_shape, include_top=False,
                                                             weights='imagenet', pooling='avg')
    feature_extractor.layers.pop()
    feature_extractor.layers[-1].outbound_nodes = []
    feature_extractor.outputs = [feature_extractor.layers[-1].output]

    # Components of Scene Understanding Modular
    # 1) Full-image encoder
    enc = keras.layers.AveragePooling2D(pool_size=(3, 3))(feature_extractor.outputs[0])
    enc = keras.layers.Dropout(0.5)(enc)
    enc = keras.layers.Flatten()(enc)
    enc = keras.layers.Dense(512, activation='relu')(enc)
    enc = keras.layers.Reshape(target_shape=(1, 1, 512))(enc)
    enc = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(enc)
    enc = keras.layers.UpSampling2D(size=feature_extractor.layers[-1].output_shape[1:3], interpolation='bilinear')(
        enc)

    # 2-5) ASPPs i.e. Atrous convolutions)
    aspp1 = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(feature_extractor.outputs[0])
    aspp1 = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(aspp1)

    aspp2 = keras.layers.Conv2D(512, padding='same', dilation_rate=3, kernel_size=3, strides=1, activation='relu')(
        feature_extractor.outputs[0])
    aspp2 = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(aspp2)

    aspp3 = keras.layers.Conv2D(512, padding='same', dilation_rate=6, kernel_size=3, strides=1, activation='relu')(
        feature_extractor.outputs[0])
    aspp3 = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(aspp3)

    aspp4 = keras.layers.Conv2D(512, padding='same', dilation_rate=9, kernel_size=3, strides=1, activation='relu')(
        feature_extractor.outputs[0])
    aspp4 = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(aspp4)

    x = keras.layers.Concatenate()([enc, aspp1, aspp2, aspp3, aspp4])
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(2048, kernel_size=1, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Conv2D(n_classes * 2, kernel_size=1, activation='relu')  # Depth is positive anyway...
    x = output(x)
    ih, iw = feature_extractor.input_shape[1:3]
    oh, ow = output.output_shape[1:3]
    x = keras.layers.UpSampling2D(size=map(round, (ih / oh, iw / ow)),
                                  interpolation='bilinear')(x)

    def ordinal_layer(input_tensor):
        decode_label = K.zeros(shape=K.shape(input_tensor)[:-1], dtype=K.floatx())
        for i in range(n_classes):
            ord_i = K.argmax(input_tensor[:, :, :, 2 * i:2 * i + 2], axis=-1)
            decode_label += K.cast(ord_i, K.floatx())
        return decode_label

    x = keras.layers.Lambda(ordinal_layer)(x)  # Return the max in each prediction class
    model = keras.models.Model(inputs=feature_extractor.input, outputs=x)

    return model


def DORN_ResNet50_KITTI(input_shape=(385, 513, 3), n_classes=71):
    # (Dense) feature extractor originally ResNet36 in implementation, we use ResNet50 here
    feature_extractor = keras.applications.resnet50.ResNet50(input_shape=input_shape, include_top=False,
                                                             weights='imagenet', pooling='avg')
    feature_extractor.layers.pop()
    feature_extractor.layers[-1].outbound_nodes = []
    feature_extractor.outputs = [feature_extractor.layers[-1].output]

    # Components of Scene Understanding Modular
    # 1) Full-image encoder
    enc = keras.layers.AveragePooling2D(pool_size=(4, 4))(feature_extractor.outputs[0])
    enc = keras.layers.Dropout(0.5)(enc)
    enc = keras.layers.Flatten()(enc)
    enc = keras.layers.Dense(512, activation='relu')(enc)
    enc = keras.layers.Reshape(target_shape=(1, 1, 512))(enc)
    enc = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(enc)
    enc = keras.layers.UpSampling2D(size=feature_extractor.layers[-1].output_shape[1:3], interpolation='bilinear')(
        enc)

    # 2-5) ASPPs i.e. Atrous convolutions)
    aspp1 = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(feature_extractor.outputs[0])
    aspp1 = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(aspp1)

    aspp2 = keras.layers.Conv2D(512, padding='same', dilation_rate=4, kernel_size=3, strides=1, activation='relu')(
        feature_extractor.outputs[0])
    aspp2 = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(aspp2)

    aspp3 = keras.layers.Conv2D(512, padding='same', dilation_rate=8, kernel_size=3, strides=1, activation='relu')(
        feature_extractor.outputs[0])
    aspp3 = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(aspp3)

    aspp4 = keras.layers.Conv2D(512, padding='same', dilation_rate=12, kernel_size=3, strides=1, activation='relu')(
        feature_extractor.outputs[0])
    aspp4 = keras.layers.Conv2D(512, kernel_size=1, activation='relu')(aspp4)

    x = keras.layers.Concatenate()([enc, aspp1, aspp2, aspp3, aspp4])
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(2048, kernel_size=1, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Conv2D(n_classes * 2, kernel_size=1, activation='relu')  # Depth is positive anyway...
    x = output(x)
    ih, iw = feature_extractor.input_shape[1:3]
    oh, ow = output.output_shape[1:3]
    x = keras.layers.UpSampling2D(size=map(round, (ih / oh, iw / ow)),
                                  interpolation='bilinear')(x)

    def ordinal_layer(input_tensor):
        decode_label = K.zeros(shape=K.shape(input_tensor)[:-1], dtype=K.floatx())
        for i in range(n_classes):
            ord_i = K.argmax(input_tensor[:, :, :, 2 * i:2 * i + 2], axis=-1)
            decode_label += K.cast(ord_i, K.floatx())
        return decode_label

    x = keras.layers.Lambda(ordinal_layer)(x)  # Return the max in each prediction class
    model = keras.models.Model(inputs=feature_extractor.input, outputs=x)

    return model
