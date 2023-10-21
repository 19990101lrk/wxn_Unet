from keras.models import *
from keras.layers import *
from keras.optimizers import *
from yolo3.model import yolo_head, box_iou, DarknetConv2D_BN_Leaky, DarknetConv2D, resblock_body

def spp_block(x):
    '''Create SPP block'''

    #	x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(512, (1, 1), strides=(1, 1))(x)
    x = DarknetConv2D_BN_Leaky(1024, (3, 3), strides=(1, 1))(x)
    x = DarknetConv2D_BN_Leaky(512, (1, 1), strides=(1, 1))(x)

    mp5 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    mp9 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    mp13 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, mp13, mp9, mp5])

    #	x = DarknetConv2D_BN_Leaky(512, (1,1), strides=(1,1))(x)
    #	x = DarknetConv2D_BN_Leaky(1024, (3,3), strides=(1,1))(x)
    #	x = DarknetConv2D_BN_Leaky(512, (1,1), strides=(1,1))(x)
    #	x = DarknetConv2D_BN_Leaky(1024, (3,3), strides=(1,1))(x)

    return x