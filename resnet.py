from keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    ZeroPadding2D,
)
from keras.models import Model


def convolutional_block(x, filters, strides=(2, 2)):
    x_shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding="same")(x)
    x_shortcut = BatchNormalization(axis=3)(x_shortcut)

    x = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization(axis=3)(x)

    x = Add()([x, x_shortcut])
    x = Activation("relu")(x)
    return x


def identity_block(x, filters):
    x_shortcut = x

    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization(axis=3)(x)

    x = Add()([x, x_shortcut])
    x = Activation("relu")(x)
    return x


def ResNet34(input_shape=(28, 28, 1), classes=345):
    x_input = Input(input_shape)
    x = ZeroPadding2D((3, 3))(x_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    stages = [(64, 3), (128, 4), (256, 6), (512, 3)]

    for filters, blocks in stages:
        strides = (1, 1) if filters == 64 else (2, 2)
        x = convolutional_block(x, filters, strides=strides)

        for _ in range(blocks - 1):
            x = identity_block(x, filters)

    x = AveragePooling2D((2, 2), padding="same")(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(classes, activation="softmax")(x)

    model = Model(inputs=x_input, outputs=x, name="ResNet34")
    return model
