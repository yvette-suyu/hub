# -*- coding: UTF-8 -*-

import numpy as np

# attr_pred
def attr_predict(img, svm_list):
    attr = []
    for svm_id in range(len(svm_list)):
        attr_temp = svm_list[svm_id].predict(img.reshape(-1,2048))
        attr.append(attr_temp)

    return np.array(attr).reshape(-1,50)




