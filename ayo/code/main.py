#coding=utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.dataset import collate_fn, dataset
import torch
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50,inception_v3
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util import train, trainlog
from  torch.nn import CrossEntropyLoss
import logging
from dataset.data_aug import *
from models.inception_v4 import inceptionv4
from models.senet import se_resnet50_xuelang,se_resnet101_xuelang


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
labelencoder = ['ZJL455','ZJL446','ZJL479','ZJL362','ZJL297','ZJL402','ZJL306','ZJL341','ZJL307','ZJL346','ZJL436',
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
print(len(labelencoder))
encoder = {}

for i in range(len(labelencoder)):
    encoder[labelencoder[i]] = i
print(encoder)


save_dir = '/../genermodel'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = '%s/trainlog.log'%save_dir
trainlog(logfile)

rawdata_root = '/data/round2B/semifinal_image_phase2/train'
all_pd = pd.read_csv("/data/round2B/semifinal_image_phase2/train.txt",sep="\t",
                     header=None, names=['ImageName', 'label'])

print(all_pd.head(10))

train_pd, val_pd = train_test_split(all_pd, test_size=0.1, random_state=43,
                                    stratify=all_pd['label'])



'''数据扩增'''
data_transforms = {
    'train': Compose([
        RandomRotate(angles=(-15,15)),
        ExpandBorder(size=(210,210),resize=True),
        # ExpandBorder(size=(368,368),resize=True),
        RandomResizedCrop(size=(200, 200)),
        # RandomResizedCrop(size=(336, 336)),
        RandomHflip(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': Compose([
        # ExpandBorder(size=(336,336),resize=True),
        ExpandBorder(size=(200,200),resize=True),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_set = {}
data_set['train'] = dataset(imgroot=rawdata_root,anno_pd=train_pd,encoder=encoder,
                           transforms=data_transforms["train"],
                           )
data_set['val'] = dataset(imgroot=rawdata_root,anno_pd=val_pd,encoder=encoder,
                           transforms=data_transforms["val"],
                           )
dataloader = {}
dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=16,
                                               shuffle=True, num_workers=8,collate_fn=collate_fn)
dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=16,
                                               shuffle=True, num_workers=8,collate_fn=collate_fn)



'''model'''

model = se_resnet101_xuelang(num_classes=160)
# model =resnet50(pretrained=False)
# model.avgpool =  torch.nn.AdaptiveAvgPool2d(output_size=1)
# model.fc = torch.nn.Linear(model.fc.in_features,2)
# model = inceptionv4(num_classes=160)
model = torch.nn.DataParallel(model)
base_lr =0.001
resume = None

if resume:
    logging.info('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-5)
criterion = CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

train(model,
      epoch_num=50,
      start_epoch=0,
      optimizer=optimizer,
      criterion=criterion,
      exp_lr_scheduler=exp_lr_scheduler,
      data_set=data_set,
      data_loader=dataloader,
      save_dir=save_dir,
      print_inter=50,
      val_inter=400)