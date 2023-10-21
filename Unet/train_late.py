#encoding:utf-8

from model_v9 import *
import data
import os
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == '__main__':

    # path to images which are prepared to train a model
    train_path = "CamVid"
    image_folder = "train"
    label_folder = "trainannot"
    valid_path = "CamVid"
    valid_image_folder = "val"
    valid_label_folder = "valannot"
    log_filepath = "./logs"
    flag_multi_class = True
    num_classes = 12
    dp = data.data_preprocess(train_path=train_path, image_folder=image_folder, label_folder=label_folder,
                         valid_path=valid_path, valid_image_folder=valid_image_folder, valid_label_folder=valid_label_folder,
                         flag_multi_class=flag_multi_class,
                         num_classes=num_classes)

    # train your own model
    train_data = dp.trainGenerator(batch_size=2)
    valid_data = dp.validLoad(batch_size=2)
    test_data = dp.testGenerator()
    model = unet(num_class=num_classes)

    tb_cb = TensorBoard(log_dir=log_filepath)
    model_checkpoint = keras.callbacks.ModelCheckpoint('./model9/CamVid_model_v2.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    history = model.fit_generator(train_data,
                                  steps_per_epoch=200, epochs=30,
                                  validation_steps=10,
                                  validation_data=valid_data,
                                  callbacks=[model_checkpoint, tb_cb])

    # draw the loss and accuracy curve
    plt.figure(12, figsize=(6, 6), dpi=60)
    plt.subplot(211)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('loss')
    plt.legend()
    plt.subplot(212)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('acc')
    plt.legend()
    plt.show()
