#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import os
import shutil
import sys
import tensorflow as tf


# 指定第一块GPU可用
import keras

config = tf.ConfigProto( device_count = {'GPU': 1})
sess = tf.Session(config=config)
keras.backend.set_session(sess)
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # Parameters
    if len(sys.argv) == 4:
        superclass = sys.argv[1]
        imgmove = sys.argv[2]
        if imgmove == 'False':
            imgmove = False
        else:
            imgmove = True
        lr = float(sys.argv[3])
    else:
        print('Parameters error')
        exit()


    trainpath = 'trainval_'+superclass+'/train'
    valpath = 'trainval_'+superclass+'/val'

    if not os.path.exists('model'):
        os.mkdir('model')

    # Train/validation data preparation
    if imgmove:
        os.mkdir('trainval_'+superclass)
        os.mkdir(trainpath)
        os.mkdir(valpath)
        sourcepath = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/train224'
        categories = os.listdir(sourcepath)
        for eachclass in categories:
            print(eachclass)
            # os.mkdir(trainpath)
            # os.mkdir(valpath)
            imgs = os.listdir(sourcepath)
            idx = 0
            for im in imgs:
                if idx%8 == 0:
                    shutil.copyfile(sourcepath+'/'+im, valpath+'/'+im)
                else:
                    shutil.copyfile(sourcepath+'/'+im, trainpath+'/'+im)
                idx += 1

    # Train and validation ImageDataGenerator
    batchsize = 8

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=5,
        height_shift_range=5,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        trainpath,
        target_size=(224, 224),
        batch_size=batchsize)
    
    print(train_generator.n)
    valid_generator = test_datagen.flow_from_directory(
        valpath,
        target_size=(224, 224),
        batch_size=batchsize)
    print(valid_generator.n)
   

    # Train MobileNet
    model = MobileNet(include_top=True, weights=None,
                      input_tensor=None, input_shape=None,
                      pooling=None, classes=190)

    model.summary()
    model.compile(optimizer=SGD(lr=lr, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    steps_per_epoch = int(train_generator.n/batchsize)
    validation_steps = int(valid_generator.n/batchsize)

    weightname = 'model/mobile_'+superclass+'_wgt.h5'

    checkpointer = ModelCheckpoint(weightname, monitor='val_loss', verbose=0,
                        save_best_only=True, save_weights_only=True, mode='auto', period=2)
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=200,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=[checkpointer])


if __name__ == "__main__":
    main()
