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
This python script is the baseline to implement zero-shot learning on each super-class.
The command is:     python MDP.py mobile

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sklearn.linear_model as models
import numpy as np
import pickle
import sys
import re
import random
import keras.backend as K

import keras

# import tensorflow as tf
# config = tf.ConfigProto( device_count = {'GPU': 1})
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

def main():
    if len(sys.argv) == 2:
        superclass = sys.argv[1]
    else:
        print('Parameters error')
        exit()

    file_feature = 'features_'+superclass+'.pickle'

    # The constants
    classNum = 190
    # testName = {'A': 'a', 'F': 'a', 'V': 'b', 'E': 'b', 'H': 'b'}
    # date = '20180321'
    # Load seen/unseen split
    all_label_list_path = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/label_list.txt'
    allfsplit = open(all_label_list_path, 'r')
    lines_alllabel = allfsplit.readlines()
    allfsplit.close()
    list_all = list()
    names_all = list()
    for each in lines_alllabel:
        tokens = each.split('\t')
        list_all.append(tokens[0])
        names_all.append(tokens[1])


    label_list_path='/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/train_label.txt'
    fsplit = open(label_list_path, 'r')
    lines_label = fsplit.readlines()
    fsplit.close()
    list_train = list()
    names_train = list()
    for each in lines_label:
        tokens = each.split('\t')
        list_train.append(tokens[0])
        names_train.append(tokens[1])
    list_test = list()
    for label in list_all:
        # for i in range(classNum):
        if label not in list_train:
            list_test.append(label)
    print('test list',len(list_test))
    # Load attributes
    attrnum = {'m':30}
    wordvectornum = {'m':300}

    attributes_per_class_path = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/attributes_per_class.txt'
    fattr = open(attributes_per_class_path, 'r')
    lines_attr = fattr.readlines()
    fattr.close()
    attributes = dict()
    for each in lines_attr:
        tokens = each.split('\t')
        label = tokens[0]
        attr = tokens[1:]
        # print(type(label))
        # print(label)
        if not (len(attr) == attrnum[superclass[0]]):
            print('attributes number error\n')
            exit()
        attributes[label] = attr

    # load word_vector
    word_vector_per_class = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/class_wordembeddings.txt'
    wvector = open(word_vector_per_class, 'r')
    lines_vector = wvector.readlines()
    wvector.close()
    wvectors =dict()
    for every in lines_vector:
        token = every.split(' ')
        wname = token[0]
        # print('wname', wname)
        for kk in range(len(names_all)):
            # print(names_all[kk])
            if wname in names_all[kk]:
                wlabel = list_all[kk]
                print('wlabel',wlabel)
                vect =token[1:]
                # print('vect',vect)
                wvectors[wlabel] = vect

    # print('wvectors:', wvectors)

    # Load image features
    fdata = open(file_feature, 'rb')
    features_dict = pickle.load(fdata)  # variables come out in the order you put them in
    fdata.close()
    features_all = features_dict['features_all']
    # print('features_all',features_all)
    labels_all = features_dict['labels_all']
    print('labels_all',len(labels_all))
    images_all = features_dict['images_all']
    print('images_all',len(images_all))

    # Calculate prototypes (cluster centers)
    features_all = features_all/np.max(abs(features_all))
    dim_f = features_all.shape[1]
    prototypes_train = np.ndarray((int(classNum), dim_f))
    print('prototypes_train',prototypes_train.shape)
    dim_a = attrnum[superclass[0]]
    dim_w = wordvectornum[superclass[0]]
    attributes_train = np.ndarray((int(classNum), dim_a))
    wordv_train = np.ndarray((int(classNum), dim_w))
    attributes_test = np.ndarray((40, dim_a))
    wordv_test = np.ndarray((40, dim_w))

    print('list', len(list_train))
    for i in range(len(list_train)):
        label = list_train[i]
        idx = [pos for pos, lab in enumerate(labels_all) if lab == label]

        prototypes_train[i, :] = np.mean(features_all[idx, :], axis=0)
        attributes_train[i, :] = np.asarray(attributes[label])
        wordv_train[i, :] = np.asarray(wvectors[label])

    for i in range(len(list_test)):
        label = list_test[i]
        attributes_test[i, :] = np.asarray(attributes[label])
        wordv_test[i, :] = np.asarray(wvectors[label])
    #
    newzero1 = np.zeros((attributes_train.shape[0], 270))
    newzero2 = np.zeros((attributes_test.shape[0], 270))
    newattr_train = np.c_[attributes_train, newzero1]
    newattr_test = np.c_[attributes_test,newzero2]
    define_train = newattr_train + wordv_train
    define_test = newattr_test + wordv_test
    # Structure learning
    LASSO = models.Lasso(alpha=0.01)

    # attr+word
    LASSO.fit(define_train.transpose(), define_test.transpose())

    attrLASSO = LASSO.fit(newattr_train.transpose(),newattr_test.transpose())
    wordLASSO = LASSO.fit(wordv_train.transpose(),wordv_test.transpose())
    W = attrLASSO.coef_ +  wordLASSO.coef_
    # Image prototype synthesis
    prototypes_test = (np.dot(prototypes_train.transpose(), W.transpose())).transpose()

    # Prediction
    label = 'test'
    idx = [pos for pos, lab in enumerate(labels_all) if lab == label]
    features_test = features_all[idx, :]
    images_test = [images_all[i] for i in idx]
    prediction = list()

    for i in range(len(idx)):
        temp = np.repeat(np.reshape((features_test[i, :]), (1, dim_f)), len(list_test), axis=0)
        distance = np.sum((temp - prototypes_test)**2, axis=1)
        pos = np.argmin(distance)
        prediction.append(list_test[pos])

    # Write prediction
    fpred = open('pred_'+ superclass + '.txt', 'w')

    for i in range(len(images_test)):
        fpred.write(str(images_test[i])+'\t'+prediction[i]+'\n')
    fpred.close()


if __name__ == "__main__":
    main()
