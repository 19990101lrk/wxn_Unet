import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
import numpy as np

def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        network = Relu(network)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Relu(x):
    return tf.nn.relu(x)


def Sigmoid(x):
    return tf.nn.sigmoid(x)


def Avg_pooling(x, pool_size=[3,3], stride=1, padding='SAME') :
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers) :
    return tf.concat(layers, axis=3)



def Fully_connected(x, units, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=False, units=units)


def Inception_B(x, training, scope):
    with tf.name_scope(scope):
        # init = x

        split_conv_x1 = Avg_pooling(x)
        split_conv_x1 = conv_layer(split_conv_x1, filter=128, kernel=[1, 1], layer_name=scope + '_split_conv1')

        split_conv_x2 = conv_layer(x, filter=384, kernel=[1, 1], layer_name=scope + '_split_conv2')

        split_conv_x3 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv3')
        split_conv_x3 = conv_layer(split_conv_x3, filter=224, kernel=[1, 7], layer_name=scope + '_split_conv4')
        split_conv_x3 = conv_layer(split_conv_x3, filter=256, kernel=[1, 7], layer_name=scope + '_split_conv5')

        split_conv_x4 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv6')
        split_conv_x4 = conv_layer(split_conv_x4, filter=192, kernel=[1, 7], layer_name=scope + '_split_conv7')
        split_conv_x4 = conv_layer(split_conv_x4, filter=224, kernel=[7, 1], layer_name=scope + '_split_conv8')
        split_conv_x4 = conv_layer(split_conv_x4, filter=224, kernel=[1, 7], layer_name=scope + '_split_conv9')
        split_conv_x4 = conv_layer(split_conv_x4, filter=256, kernel=[7, 1], layer_name=scope + '_split_connv10')

        x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])

        x = Batch_Normalization(x, training=training, scope=scope + '_batch1')
        x = Relu(x)

        return x

def squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input_x * excitation

        return scale




def build_inception(inputs, istraining):
    x = Inception_B(inputs, istraining, scope='build_Inception')
    channel = int(np.shape(x)[-1])
    x = squeeze_excitation_layer(x, out_dim=channel, ratio=16, layer_name='SE_Inception')

    return x