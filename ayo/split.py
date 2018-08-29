import os

import shutil

dirpath='/media/hszc/data1/syh/zhijiang/AI_Challenger_2018/Baselines/zero_shot_learning_baseline/trainval_mobile/train'

for parent, dirnames, filenames in os.walk(dirpath):
    filenamelist=filenames
    # print(filenames)
    eachdirname = os.path.split(parent)[1]
    # print(eachdirname)
    savefilepath = '/media/hszc/data1/syh/zhijiang/AI_Challenger_2018/Baselines/zero_shot_learning_baseline/trainval_mobile/val/' + eachdirname + '/'
    if filenamelist:
        imgpath=parent
        for i in range(int(len(filenamelist)*0.2)):

            print('Fname',filenamelist[i])
            img=imgpath+'/'+filenamelist[i]
            if not os.path.exists(savefilepath):
                os.makedirs(savefilepath)
            else:
                print('savepath is existed')
            print('--ok--')

            shutil.move(img,savefilepath+filenamelist[i])

