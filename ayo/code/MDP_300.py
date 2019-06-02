#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sklearn.linear_model as models
import numpy as np
import pickle
import time
import sys
import re
import random


def main():

    file_feature = '/../pkl/2018-10-30-10_07_21.pickle'

    # The constants
    classNum = 2

    # Load seen/unseen split
    all_label_list_path = '/data/round2B/semifinal_image_phase2/label_list.txt'
    allfsplit = open(all_label_list_path, 'r')
    lines_alllabel = allfsplit.readlines()
    allfsplit.close()
    list_all = list()
    names_all = list()
    for each in lines_alllabel:
        tokens = each.split('\t')
        list_all.append(tokens[0])
        names_all.append(tokens[1])


    label_list_path='/data/round2B/semifinal_image_phase2/use/train_label.txt'
    fsplit = open(label_list_path, 'r')
    lines_label = fsplit.readlines()
    fsplit.close()
    list_train = list()
    # names_train = list()
    for each in lines_label:
        tokens = each.split('\n')
        list_train.append(tokens[0])
    print ('list_train',list_train)
        # names_train.append(tokens[1])
    list_test = list()
    for label in list_all:
        # for i in range(classNum):
        if label not in list_train:
            list_test.append(label)
    print('test list',len(list_test))
    # Load attributes
    attrnum = 50
    wordvectornum = 300

    attributes_per_class_path = '/data/round2B/semifinal_image_phase2/attributes_per_class.txt'
    fattr = open(attributes_per_class_path, 'r')
    lines_attr = fattr.readlines()
    fattr.close()
    attributes = dict()
    for each in lines_attr:
        tokens = each.split('\t')
        label = tokens[0]
        attr = tokens[1:]
        # attr.append(tokens[29])
        # print(type(label))
        # print(label)
        if not (len(attr) == attrnum):
            print('attributes number error\n')
            exit()
        attributes[label] = attr

    print(type(attr))


    # load word_vector
    word_vector_per_class = '/data/round2B/semifinal_image_phase2/class_wordembeddings.txt'
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

    print('wvectors:', wvectors)

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
    dim_a = attrnum
    dim_w = wordvectornum
    attributes_train = np.ndarray((int(classNum), dim_a))
    wordv_train = np.ndarray((int(classNum), dim_w))
    attributes_test = np.ndarray((len(list_test), dim_a))
    wordv_test = np.ndarray((len(list_test), dim_w))

    # print('list', len(list_train))
    for i in range(len(list_train)):
        label = list_train[i]
        idx = [pos for pos, lab in enumerate(labels_all) if lab == label]
        label = label.strip('\n')

        prototypes_train[i, :] = np.mean(features_all[idx, :], axis=0)
        attributes_train[i, :] = np.asarray(attributes[label])
        wordv_train[i, :] = np.asarray(wvectors[label])

    for i in range(len(list_test)):
        labeltest = list_test[i]
        print('labeltest',labeltest)
        attributes_test[i, :] = np.asarray(attributes[labeltest])
        wordv_test[i, :] = np.asarray(wvectors[labeltest])
    #
    newzero1 = np.zeros((attributes_train.shape[0], 300))
    newzero2 = np.zeros((attributes_test.shape[0], 300))
    newattr_train = np.c_[attributes_train, newzero1]
    newattr_test = np.c_[attributes_test, newzero2]
    newzero3 = np.zeros((wordv_train.shape[0], 50))
    newzero4 = np.zeros((wordv_test.shape[0], 50))
    newword_train = np.c_[newzero3, wordv_train]
    newword_test = np.c_[newzero4, wordv_test]
    define_train = newattr_train + newword_train
    define_test = newattr_test + newword_test
    # Structure learning
    LASSO = models.Lasso(alpha=0.01)
    # attr only
    # LASSO.fit(attributes_train.transpose(), attributes_test.transpose())
    # attr+word
    LASSO.fit(define_train.transpose(), define_test.transpose())

    # word only
    # LASSO.fit(wordv_train.transpose(), wordv_test.transpose())
    W = LASSO.coef_

    prototypes_test = (np.dot(prototypes_train.transpose(), W.transpose())).transpose()


    # Prediction
    label = 'test'
    idx = [pos for pos, lab in enumerate(labels_all) if lab == label]
    features_test = features_all[idx, :]
    images_test = [images_all[i] for i in idx]
    prediction1 = list()
    prediction2 = list()
    prediction3 = list()
    print(len(idx))
    print('list_test',list_test)
    print('list_test', len(list_test))
    for i in range(len(idx)):
        temp = np.repeat(np.reshape((features_test[i, :]), (1, dim_f)), len(list_test), axis=0)
        distance = np.sum((temp - prototypes_test)**2, axis=1)
        # pos = np.argmin(distance)
        pos1,pos2,pos3 = np.argsort(distance)[0], np.argsort(distance)[1],np.argsort(distance)[2]
        prediction1.append(list_test[pos1])
        prediction2.append(list_test[pos2])
        prediction3.append(list_test[pos3])
        # print('prediction',prediction)

    # Write prediction
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    fpred = open('/submit/pred_MDP'+ now + '.txt', 'w')

    for i in range(len(images_test)):
        fpred.write(str(images_test[i])+'\t'+prediction1[i]+'\t'+prediction2[i]+'\t'+prediction3[i]+'\n')
    fpred.close()


if __name__ == "__main__":
    main()
