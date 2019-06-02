import pandas as pd
import numpy as np
import time
def doit(label1,label2,label3,label4,label5):
    lis = []
    lis.append(int(label1[3:]))
    lis.append(int(label2[3:]))
    lis.append(int(label3[3:]))
    lis.append(int(label4[3:]))
    lis.append(int(label5[3:]))
    return "ZSL"+str(np.argmax(np.bincount(lis)))



res1= pd.read_csv("/result/pred_mobile_inception.txt",sep="\t",header=None,names=["id","label1"])

res2= pd.read_csv("/result/pred_mobile_inception1.txt",sep="\t",header=None,names=["id","label2"])

res3= pd.read_csv("/result/pred_mobile_resnet.txt",sep="\t",header=None,names=["id","label3"])

res4= pd.read_csv("/result/pred_mobile_inre.txt",sep="\t",header=None,names=["id","label4"])

res5= pd.read_csv("/result/pred_mobile_nas.txt",sep="\t",header=None,names=["id","label5"])

res = pd.merge(res1,res2,on='id',how="left")
res = pd.merge(res,res3,on='id',how="left")
res = pd.merge(res,res4,on='id',how="left")
res = pd.merge(res,res5,on='id',how="left")
print(res.head(10))
res['label'] = res[["label1","label2","label3","label4","label5"]].apply(lambda x:doit(x.label1,x.label2,x.label3,x.label4,x.label5),axis=1)
print(res.head(10))

now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
res[["id","label"]].to_csv("submit_"+now+".txt",header=None,sep="\t",index=False)
