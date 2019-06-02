# /usr/bin/env python
#  -*- coding: UTF-8 -*-

import pickle
import re
import os
import matplotlib.pyplot as plt
import cv2
import time
def imgfeature_reader(file_feature):
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
    return features_all,labels_all,images_all

def test_label_reader(path):
    label_list_path = path
    fsplit = open(label_list_path, 'r')
    lines_label = fsplit.readlines()
    fsplit.close()
    list_test = list()
    # names_train = list()
    for each in lines_label:
        # tokens = each.split('\t')
        tokens = re.split(r'[\t\n]', each)
        list_test.append(tokens[0])
    return list_test
list_test = test_label_reader('/data/round2_DatasetA_20180927/use/test_label.txt')
list_train =test_label_reader('/data/round2_DatasetA_20180927/use/train_label.txt')

# print ('list_train',list_train)

def attribute_reader(flag):
    attributes_per_class_path = '/data/round2_DatasetA_20180927/attributes_per_class.txt'
    fattr = open(attributes_per_class_path, 'r')
    lines_attr = fattr.readlines()
    fattr.close()
    attributes_train = dict()
    attributes_test = dict()
    for each in lines_attr:
        tokens = each.split('\t')

        #  need to select 0-1 attr !!!
        label = tokens[0]
        # attr = tokens[15]
        attr = tokens[1:51]
        # attr.append(tokens[19])
        # attr.append(tokens[20])
        # attr.append(tokens[21])
        # attr.append(tokens[22])
        # attr.append(tokens[23])
        # attr.append(tokens[24])
        # attr.append(tokens[25])
        # attr.append(tokens[37])
        # attr.append(tokens[46])
        # attr.append(tokens[48])
        #  adjust attribute to 0-1
        # for index in range(len(attr)):
        #     if attr[index] < 0.5:
        #         attr[index] = 0
        #     else:
        #         attr[index] = 1
        if label in list_train:
            attributes_train[label] = attr
        elif label in list_test:
            attributes_test[label] = attr
        else:
            print ('fun error in attr_reader: something is wrong with label reader')
    if flag == 'train':
        return attributes_train
    elif flag == 'test':
        return attributes_test

def img_show(image_name):

    image_path = '/data/DatasetB_20180919/test224/'
    img_path = os.path.join(image_path, image_name)
    img = cv2.imread(img_path)
    plt.imshow(img[:, :, ::-1])
    plt.show()

