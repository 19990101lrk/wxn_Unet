import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers import *
from keras import layers
from keras.optimizers import *
from keras.layers import LeakyReLU
from keras.layers import multiply
def SKConv(M=2, r=16, L=32, G=32):

  def wrapper(inputs):
    inputs_shape =K.int_shape(inputs)
    b, h, w = inputs_shape[0], inputs_shape[1], inputs_shape[2]
    filters = inputs.get_shape().as_list()[-1]
    d = max(filters//r, L)

    x = inputs
    xs = []
    for m in range(1, M+1):
      if G == 1:
        _x = Conv2D(filters, 3, dilation_rate=m, padding='same', use_bias=False)(x)
      else:
        c = filters //G
        _x = DepthwiseConv2D(3, dilation_rate=m, depth_multiplier=c, padding='same', use_bias=False)(x)

        _x = Reshape([h, w, G, c, c])(_x)
        _x = Lambda(lambda x: np.sum(_x, axis=-1), output_shape=[b, h, w, G, c])(_x)
        _x = Reshape([h, w, filters])(_x)

      _x = BatchNormalization(axis=3)(_x)
      _x = Activation('relu')(_x)

      xs.append(_x)

    U = Add()(xs)
    s = Lambda(lambda x: K.mean(x, axis=None, keepdims=False), output_shape=[b, 1, 1, filters])(U)

    z = Conv2D(d, 1)(s)
    z = BatchNormalization(axis=3)(z)
    z = Activation('relu')(z)

    x = Conv2D(filters*M, 1)(z)
    x = Reshape([1, 1, filters, M])(x)
    scale = Softmax()(x)

    x = Lambda(lambda x: K.stack(x, axis=-1), output_shape=[b, h, w, filters, M])(xs) # b, h, w, c, M
    x = Axpby()([scale, x])

    return x
  return wrapper

class Axpby(layers.Layer):
  """
  Do this:
    F = a * X + b * Y + ...
    Shape info:
      a:  B x 1 x 1 x C
      X:  B x H x W x C
      b:  B x 1 x 1 x C
      Y:  B x H x W x C
      ...
      F:  B x H x W x C
  """
  def __init__(self, **kwargs):
        super(Axpby, self).__init__(**kwargs)

  def build(self, input_shape):
        super(Axpby, self).build(input_shape)  # Be sure to call this at the end

  def call(self, inputs, **kwargs):
    """ scale: [B, 1, 1, C, M]
        x: [B, H, W, C, M]
    """
    scale, x = inputs
    f = multiply([scale, x])
    f = np.sum(f, axis=-1, name='sum')
    return f

  def compute_output_shape(self, input_shape):
    return input_shape[0:4]


# if __name__ == '__main__':
#   from tensorflow.keras.layers import Input
#   from tensorflow.keras.models import Model
#
#   inputs = Input([None, None, 32])
#   x = SKConv(3, G=1)(inputs)
#
#   m = Model(inputs, x)
#   m.summary()
#
#   import numpy as np
#
#   X = np.random.random([2, 224, 224, 32]).astype(np.float32)