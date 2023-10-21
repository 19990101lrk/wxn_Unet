from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers import multiply


def ASPP(input, out_channel):
    aspp_list = []
    x =Conv2D(out_channel, (1, 1), strides=(1, 1), kernel_initializer='glorot_normal', padding="same",dilation_rate=(1,1), name='as_conv1_1' )(input)
    x = BatchNormalization(name='as_conv1_1_bn')(x)
    x1 = Activation('relu',name='as_conv1_1_act')(x)

    x = Conv2D(out_channel, (3, 3), strides=(1, 1), kernel_initializer='glorot_normal', padding="same", dilation_rate=(6, 6), name='as_conv2_1')(input)
    x = BatchNormalization(name='as_conv2_1_bn')(x)
    x2 = Activation('relu', name='as_conv2_1_act')(x)

    x = Conv2D(out_channel, (3, 3), strides=(1, 1), kernel_initializer='glorot_normal', padding="same", dilation_rate=(12, 12),name='as_conv3_1')(input)
    x = BatchNormalization(name='as_conv3_1_bn')(x)
    x3 = Activation('relu', name='as_conv3_1_act')(x)

    x = Conv2D(out_channel, (3, 3), strides=(1, 1), kernel_initializer='glorot_normal', padding="same", dilation_rate=(18, 18), name='as_conv4_1')(input)
    x = BatchNormalization(name='as_conv4_1_bn')(x)
    x4 = Activation('relu', name='as_conv4_1_act')(x)

    x = AveragePooling2D((1, 1))(input)
    x = Conv2D(out_channel, (1, 1), strides=(1, 1), kernel_initializer='glorot_normal', padding="same", name='as_conv5_1')(x)
    x = BatchNormalization(name='as_conv5_1_bn')(x)
    x5 = Activation('relu', name='as_conv5_1_act')(x)

    x = Concatenate(axis=3)([x1, x2, x3, x4, x5])

    x = Conv2D(out_channel, (1, 1), strides=(1, 1), kernel_initializer='glorot_normal', padding="same", name='as_conv6_1')(x)
    x = BatchNormalization(name='as_conv6_1_bn')(x)
    x = Activation('relu', name='as_conv6_1_act')(x)

    x = Dropout(0.5)(x)

    return x