# -*- coding: UTF-8 -*-
import time
import os
from classifer import svm_m_train, bayes_train
from fun import imgfeature_reader,test_label_reader,attribute_reader,img_show
import numpy as np
from model import attr_predict
    # , cls_predict

################################################################################

print("====================")
print("begin to read data")
print("====================\n")

# Load image features
file_feature = '/pkl/2018-10-27-08_35_07.pickle'
features_all,labels_all,images_all = imgfeature_reader(file_feature)
# pick out train/test

# test data
# test label
test_label_path = '/data/round2B/semifinal_image_phase2/use/test_label.txt'
test_cls = test_label_reader(test_label_path)
print ('list_test',test_cls)
print('len_test_label',len(test_cls))

label = 'test'
idx = [pos for pos, lab in enumerate(labels_all) if lab == label]

features_test = features_all[idx, :]
print('features_test',features_test.shape)
for alll in features_test:
    print(type(alll),alll)
images_test = [images_all[i] for i in idx]
#  test_attr contains test_label--test_attr, it means test_attr[label] = attr
test_attr_temp = attribute_reader('test')
print('test_arry_ori',test_attr_temp)
print('test_arry_ori',type(test_attr_temp))
test_attr = []
# xiugai!!!
batch_index = 0
for uu in range(50):

    for convert in test_attr_temp.keys():

        if test_attr_temp[convert][uu] == '0':
            testattr_temp = np.zeros((1,), dtype=int)
        else:
            testattr_temp = np.ones((1,), dtype=int)
        if batch_index == 0:

            test_attr = testattr_temp

        else:


            test_attr = np.concatenate((test_attr, testattr_temp))

        batch_index += 1
print('convert test attr',len(test_attr),'!!!')

# train data
train_index = len(idx)
t_idx = []
for count in range(train_index,len(features_all)):
    t_idx.append(count)

features_train = features_all[t_idx,:]
images_train = [images_all[j] for j in range(train_index,len(images_all))]
train_attr = attribute_reader('train')
train_label = labels_all[train_index:]


print("====================")
print("data has been prepared!")
print("====================\n")


print("train svm")

svm_m = svm_m_train('True',50, features_train, train_label, train_attr)
print("about svm",len(svm_m),svm_m)
print("____SVM has trained well____")




# print("____start_bayes_train____")

# bayes_clf = bayes_train(test_attr,test_cls)

# print("____bayes_train_has_compeleted____")



# print("____test once____")
#
# predict_img = features_test[0]
# print("test_img_class")
#
# attribute = attr_predict(predict_img, svm_m)
# cls_name = cls_predict(attribute, bayes_clf)
#
# print("test_label:"+ str(cls_name))
# img_predict = images_test[0]
# im_path = '/data/round2_DatasetA_20180927/test/'
# img_show(os.path.join(im_path + img_predict))


# str = raw_input('Will we go on test?   (Yes/No)')
# print('Received input is :' , str)
# if str == 'yes':
#     print('ok')

    # test_all
def distance(attr,test_attr):
    result = []
    print('predict_attr',len(test_attr),'real_attr',len(attr))
    # if len(attr) != len(test_attr):
    #     raise ValueError,'predict attr and test attr len is not equal'
    # else:
    test_attr_use = np.array(test_attr).reshape(-1,50)
    for i in range(test_attr_use.shape[0]):
        dislist = []
        for j in range(attr.shape[0]):
            dist = sum(abs(attr[j]-test_attr_use[i]))
            # print('dist',dist)
            dislist.append(dist)
        print('dislist',dislist)
        result.append(dislist.index(min(dislist)))
        print(result)

    return result
prediction = []
attribute = []
for all in features_test:
    attribute_temp = attr_predict(all, svm_m)
    attribute.append(attribute_temp)
attribute = np.array(attribute).reshape(-1, 50)
indexx = distance(test_attr, attribute)
for test_cls_pred in indexx:
    prediction.append(test_cls[test_cls_pred])


now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
fpred = open('/result/pred_attr_classify' + now + '.txt', 'w')

for i in range(len(features_test)):
    fpred.write(str(images_test[i]) + '\t' + prediction[i] + '\n')
fpred.close()
print('Well done!')

#
# else:
#     print('End')
