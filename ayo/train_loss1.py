#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Baseline codes for zero-shot learning task.
This python script is train a deep feature extractor (CNN).
The command is:     python train_CNN.py mobile True 0.05
T
The third parameter is the learning rate of the deep network (MobileNet).
The trained model will be saved at 'model/mobile_wgt.h5'
"""

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
from keras.models import Model
import keras.backend as K
# ?????GPU??
import keras

config = tf.ConfigProto(device_count={'GPU': 1})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """

    pos_dist = K.sum(K.square(anchor - positive), axis=1, keepdims=True)
    neg_dist = K.sum(K.square(anchor - negative), axis=1, keepdims=True)
    dis_pos = K.sqrt(pos_dist)
    dis_neg =  K.sqrt(neg_dist)
    loss = dis_pos + K.maximum(0.0, dis_pos - dis_neg + alpha)

    return K.mean(loss)
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

    # The constants
    # classNum = {'A': 40, 'F': 40, 'V': 40, 'E': 40, 'H': 24}
    # testName = {'A': 'a', 'F': 'a', 'V': 'b', 'E': 'b', 'H': 'b'}
    # date = '20180321'

    trainpath = 'trainval_' + superclass + '/train'
    valpath = 'trainval_' + superclass + '/val'

    if not os.path.exists('model'):
        os.mkdir('model')

    # Train/validation data preparation
    if imgmove:
        os.mkdir('trainval_' + superclass)
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
                if idx % 8 == 0:
                    shutil.copyfile(sourcepath + '/' + im, valpath + '/' + im)
                else:
                    shutil.copyfile(sourcepath + '/' + im, trainpath + '/' + im)
                idx += 1

    # Train and validation ImageDataGenerator
    batchsize = 8

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=5,
        height_shift_range=5,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

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
    # print('valattr',dir(valid_generator))
    print('out',train_generator.next()[1])
    # Train MobileNet
    model = MobileNet(include_top=True, weights=None,
                      input_tensor=None, input_shape=None,
                      pooling=None, classes=190)
    # model = VGG19(include_top=True, weights=None,
    #                             input_tensor=None, input_shape=None,
    #                             pooling=None,
    #                             classes=190)

    layer_name = 'reshape_1'
    intermediate_layer_model = Model(inputs= model.input,
                    outputs = model.get_layer(layer_name).output)
    embeddings = intermediate_layer_model.predict(train_generator.next()[1])
    print('about embedding',type(embeddings))
    anchor = embeddings[512:int(1024/3)]
    positive = embeddings[int(1024/3)+1:int(1024/3)*2]
    negative = embeddings[int(1024/3)+1*2:1024]
    usetriplet_loss = triplet_loss(anchor, positive, negative, 1)
    model.summary()
    model.compile(optimizer=SGD(lr=lr, momentum=0.9),
                  loss='categorical_crossentropy'+usetriplet_loss, metrics=['accuracy'])

    steps_per_epoch = int(train_generator.n / batchsize)
    validation_steps = int(valid_generator.n / batchsize)

    weightname = 'model/mobile_' + superclass + '_wgt.h5'

    checkpointer = ModelCheckpoint(weightname, monitor='val_loss', verbose=0,
                                   save_best_only=True, save_weights_only=True, mode='auto', period=2)
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=150,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=[checkpointer])


if __name__ == "__main__":
    main()
