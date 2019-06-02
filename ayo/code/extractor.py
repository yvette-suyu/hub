import os
import numpy as np
import pandas as pd
from dataset.dataset import dataset, collate_fn
import torch
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from math import ceil
from  torch.nn.functional import softmax
from dataset.data_aug import *
import pickle
from models.inception_v4 import inceptionv4
from models.senet import se_resnet101_xuelang
import time
test_labelencoder = ['ZJL296','ZJL298','ZJL301','ZJL302','ZJL303','ZJL312','ZJL320','ZJL322','ZJL326','ZJL330','ZJL349','ZJL351',
                'ZJL352','ZJL354','ZJL361','ZJL364','ZJL377','ZJL393','ZJL395','ZJL404','ZJL406','ZJL408','ZJL410','ZJL411',
                'ZJL416','ZJL420','ZJL422','ZJL431','ZJL434','ZJL441','ZJL447','ZJL451','ZJL462','ZJL469','ZJL474','ZJL481',
                'ZJL485','ZJL486','ZJL489','ZJL491','ZJL496','ZJL497','ZJL498','ZJL499','ZJL500','ZJL501','ZJL502','ZJL503',
                'ZJL504','ZJL505','ZJL506','ZJL507','ZJL508','ZJL509','ZJL510','ZJL511','ZJL512','ZJL513','ZJL514','ZJL515',
                'ZJL516','ZJL517','ZJL518','ZJL519','ZJL520']

test_encoder = {}

for i in range(len(test_labelencoder)):
    test_encoder[test_labelencoder[i]] = i



test_transforms= Compose([
        ExpandBorder(size=(180,180),resize=True),
        # ExpandBorder(size=(336,336),resize=True),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
mode ="test"

rawdata_root = '/data/round2B/test'
true_test_pb = pd.read_csv("/data/round2B/semifinal_image_phase2/image.txt",
                       header=None, names=['ImageName'])

print(true_test_pb.head(10))

true_test_pb['label'] = 'ZJL296'

test_pd =true_test_pb if mode=="test" else val_pd
print(test_pd.head())

data_set = {}
data_set['test'] = dataset(imgroot=rawdata_root, anno_pd=test_pd,encoder=test_encoder,
                             transforms=test_transforms,
                             )
data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=8, num_workers=4,
                                           shuffle=False, pin_memory=True, collate_fn=collate_fn)

# model_name = 'resnet50-out'
resume = None

model = se_resnet101_xuelang(num_classes=160)
# model =resnet50(pretrained=False)
# model.avgpool =  torch.nn.AdaptiveAvgPool2d(output_size=1)
# model.fc = torch.nn.Linear(model.fc.in_features,2)
# model = inceptionv4(num_classes=160)

print('resuming finetune from %s'%resume)
model.load_state_dict(torch.load(resume),strict=False)
model = model.cuda()
model.eval()

criterion = CrossEntropyLoss()

test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
test_scores = np.zeros((len(data_set['test'])), dtype=np.float32)
true_label = np.zeros((len(data_set['test'])), dtype=np.int)
idx = 0
test_loss = 0
test_corrects = 0

features_all = np.ndarray((40003, 2048))
labels_all = list()


images_all = list(test_pd['ImageName'].values)
for i in range(len(images_all)):
    labels_all.append('test')


for batch_cnt_test, data_test in enumerate(data_loader['test']):
    # print data
    print("{0}/{1}".format(batch_cnt_test, int(test_size)))
    inputs, labels = data_test
    inputs = Variable(inputs.cuda())
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
    # forward
    outputs2 = model(inputs)

    features_all[idx:(idx + labels.size(0)), :] = outputs2.data.cpu().numpy()
    idx += labels.size(0)


train_labelencoder = ['ZJL455','ZJL446','ZJL479','ZJL362','ZJL297','ZJL402','ZJL306','ZJL341','ZJL307','ZJL346','ZJL436',
                      'ZJL412','ZJL429','ZJL338','ZJL329','ZJL443','ZJL311','ZJL444','ZJL313','ZJL370','ZJL324','ZJL399',
                      'ZJL415','ZJL305','ZJL392','ZJL493','ZJL367','ZJL331','ZJL337','ZJL468','ZJL428','ZJL321','ZJL466',
                      'ZJL389','ZJL363','ZJL432','ZJL332','ZJL435','ZJL423','ZJL403','ZJL394','ZJL476','ZJL426','ZJL300',
                      'ZJL384','ZJL348','ZJL310','ZJL473','ZJL472','ZJL299','ZJL381','ZJL480','ZJL365','ZJL396','ZJL342',
                      'ZJL495','ZJL492','ZJL471','ZJL448','ZJL359','ZJL458','ZJL452','ZJL388','ZJL328','ZJL385','ZJL323',
                      'ZJL409','ZJL390','ZJL400','ZJL380','ZJL465','ZJL344','ZJL487','ZJL369','ZJL374','ZJL419','ZJL318',
                      'ZJL368','ZJL425','ZJL371','ZJL482','ZJL333','ZJL413','ZJL414','ZJL405','ZJL445','ZJL437','ZJL339',
                      'ZJL315','ZJL387','ZJL439','ZJL350','ZJL421','ZJL477','ZJL347','ZJL449','ZJL378','ZJL375','ZJL433',
                      'ZJL360','ZJL453','ZJL382','ZJL450','ZJL427','ZJL356','ZJL417','ZJL424','ZJL470','ZJL454','ZJL343',
                      'ZJL438','ZJL386','ZJL460','ZJL366','ZJL464','ZJL440','ZJL314','ZJL316','ZJL459','ZJL483','ZJL463',
                      'ZJL484','ZJL430','ZJL309','ZJL325','ZJL353','ZJL357','ZJL358','ZJL345','ZJL379','ZJL334','ZJL475',
                      'ZJL478','ZJL391','ZJL355','ZJL340','ZJL373','ZJL383','ZJL372','ZJL488','ZJL304','ZJL401','ZJL442',
                      'ZJL327','ZJL335','ZJL418','ZJL456','ZJL397','ZJL319','ZJL407','ZJL376','ZJL317','ZJL308','ZJL490',
                      'ZJL461','ZJL467','ZJL336','ZJL398','ZJL457','ZJL494',]

train_encoder = {}

for i in range(len(train_labelencoder)):
    train_encoder[train_labelencoder[i]] = i

rawdata_root = '/data/round2B/semifinal_image_phase2/train'
true_test_pb = pd.read_csv("/data/round2B/semifinal_image_phase2/train.txt",sep='\t',
                       header=None, names=['ImageName','label'])

print(true_test_pb.tail(10))


test_pd =true_test_pb if mode=="test" else val_pd


data_set = {}
data_set['test'] = dataset(imgroot=rawdata_root, anno_pd=test_pd,encoder=train_encoder,
                             transforms=test_transforms,
                             )
data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=8, num_workers=4,
                                           shuffle=False, pin_memory=True, collate_fn=collate_fn)

# model_name = 'resnet50-out'
resume = None

model = se_resnet101_xuelang(num_classes=160)
# model = inceptionv4(num_classes=160)
# model =resnet50(pretrained=False)
# model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
model.fc = torch.nn.Linear(model.fc.in_features,2)

print('resuming finetune from %s'%resume)
model.load_state_dict(torch.load(resume),strict=False)
model = model.cuda()
model.eval()

criterion = CrossEntropyLoss()

test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
test_scores = np.zeros((len(data_set['test'])), dtype=np.float32)
true_label = np.zeros((len(data_set['test'])), dtype=np.int)
test_loss = 0
test_corrects = 0


for i in range(len(test_pd['ImageName'].values)):
    images_all.append(test_pd['ImageName'].iloc[i])
    labels_all.append(test_pd['label'].iloc[i])

print(len(images_all))

for batch_cnt_test, data_test in enumerate(data_loader['test']):
    # print data
    print("{0}/{1}".format(batch_cnt_test, int(test_size)))
    inputs, labels = data_test
    inputs = Variable(inputs.cuda())
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
    # forward
    outputs2 = model(inputs)

    features_all[idx:(idx + labels.size(0)), :] = outputs2.data.cpu().numpy()
    idx += labels.size(0)


data_all = {'features_all':features_all, 'labels_all':labels_all,
            'images_all':images_all}

# Save features
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
savename = '/../pkl/'+ now +'.pickle'
fsave = open(savename, 'wb')
pickle.dump(data_all, fsave)
fsave.close()
