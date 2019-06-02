# -*- coding: UTF-8 -*-

import numpy as np
from sklearn import svm, naive_bayes
import time

def svm_m_train(need,num, train_img, train_label, train_attr):
    svm_m = []
    m = num
    count = 0

    # m attr m SVM
    for attr_index in range(m):
        clf = svm.SVC()
        batch_index = 0
        if need == 'True':
            for index, batch in enumerate(train_img):
                attr = train_attr[train_label[index]][attr_index] # 获取的是单一值
                # print('attr',train_label[index],attr)

                if attr == '0':

                    attr_temp = np.zeros((1,), dtype=int)
                    # print('attr_temp',attr_temp.shape)
                else:

                    attr_temp = np.ones((1,), dtype=int)
                    # print('attr_temp',attr_temp.shape)


                if batch_index == 0:
                    img_for_train = train_img[index]
                    attr_for_train = attr_temp
                    # print('shape img', img_for_train.shape)
                    # print('shape attr', attr_for_train.shape)
                else:

                    img_for_train = np.vstack((img_for_train, train_img[index]))
                    attr_for_train = np.concatenate((attr_for_train, attr_temp))

                batch_index += 1
            # print('shape_img', img_for_train)
            # print('shape_attr', attr_for_train)
            f1 = ('/data/round2B/semifinal_image_phase2/use/img_for_train' + str(count) + '.npy')
            f2 = ('/data/round2B/semifinal_image_phase2/use/attr_for_train' + str(count) + '.npy')

            np.save(f1, img_for_train)
            np.save(f2, attr_for_train)
        elif need == 'False':
            print('begin to read .npy'+str(count))
            img_for_train = np.load('/data/round2B/semifinal_image_phase2/use/img_for_train' + str(count) + '.npy')
            attr_for_train = np.load('/data/round2B/semifinal_image_phase2/use/attr_for_train' + str(count) + '.npy')

        clf.fit(img_for_train, attr_for_train)
        svm_m.append(clf)
        count = count + 1
        print('count: ===>', count , '/ 50')
        # print('attr:',attr_for_train)
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        print('task time: ',now)


    return svm_m


def bayes_train(test_attr,test_cls):

    clf = naive_bayes.BernoulliNB()
    clf.fit(test_attr.reshape(-1,50), test_cls)

    return clf