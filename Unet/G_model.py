from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers import multiply


def GC_block(input_1, n_kernels):
    conv_1 = Conv2D(n_kernels, 1, padding='same', kernel_initializer='he_normal')(input_1)
    pool_1 = GlobalMaxPooling2D()(conv_1)
    conv_2 = Conv2D(n_kernels, 1, padding='same', kernel_initializer='he_normal')(pool_1)
    conv_2 = BatchNormalization(axis=3)(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_2 = Activation('sigmoid')(conv_2)
    conv_2 = multiply([conv_1, conv_2])
    conv_2 = Conv2D(n_kernels, 1, padding='same', kernel_initializer='he_normal')(conv_2)

    conv_3 = Conv2D(n_kernels, 1, padding='same', kernel_initializer='he_normal')(input_1)
    conv_3 = BatchNormalization(axis=3)(conv_3)
    conv_3 = Add()([conv_2, conv_3])

    return conv_3