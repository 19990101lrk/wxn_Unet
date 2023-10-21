from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers import multiply
from G_model import GC_block

# from keras.utils import plot_model


IMG_SIZE = 512


def res_block(in_put, n_kernels):
    conv = BatchNormalization(axis=3)(in_put)
    conv = Activation('relu')(conv)
    conv = Conv2D(n_kernels, 3, padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(n_kernels, 3, padding='same', kernel_initializer='he_normal')(conv)

    conv_shortcut = Conv2D(n_kernels, 3, padding='same', kernel_initializer='he_normal')(in_put)
    conv_shortcut = BatchNormalization(axis=3)(conv_shortcut)
    conv = Add()([conv, conv_shortcut])
    return conv


def unet(pretrained_weights=None, input_size=(IMG_SIZE, IMG_SIZE, 3), num_class=12):
    inputs = Input(input_size)
    # inputs_segBN = BatchNormalization(axis=3)(inputs)

    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    conv1_shortcut = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1_shortcut = BatchNormalization(axis=3)(conv1_shortcut)
    conv1 = Add()([conv1, conv1_shortcut])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = res_block(pool1, n_kernels=128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = res_block(pool2, n_kernels=256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = res_block(pool3, n_kernels=512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = res_block(pool4, n_kernels=1024)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    scale6 = multiply([conv4, up6])
    conv6_GC = GC_block(scale6, n_kernels=512)
    merge6 = concatenate([conv6_GC, up6], axis=3)
    conv6 = res_block(merge6, n_kernels=512)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    scale7 = multiply([conv3, up7])
    conv7_GC = GC_block(scale7, n_kernels=512)
    merge7 = concatenate([conv7_GC, up7], axis=3)
    conv7 = res_block(merge7, n_kernels=256)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    scale8 = multiply([conv2, up8])
    conv8_GC = GC_block(scale8, n_kernels=512)
    merge8 = concatenate([conv8_GC, up8], axis=3)
    conv8 = res_block(merge8, n_kernels=128)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    scale9 = multiply([conv1, up9])
    conv9_GC = GC_block(scale9, n_kernels=512)
    merge9 = concatenate([conv9_GC, up9], axis=3)
    conv9 = res_block(merge9, n_kernels=64)
    if num_class == 2:
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        loss_function = 'binary_crossentropy'
    else:
        conv10 = Conv2D(num_class, 1, activation='softmax')(conv9)
        loss_function = 'categorical_crossentropy'
    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss=loss_function, metrics=["accuracy"])
    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model