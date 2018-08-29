## 训练网络  
  
train_loss.py   
  
  The command is:
  ```python
  /anaconda3/bin/python3.6 train_loss.py mobile False 0.05

  ```
如果数据集分好了就使用 False/没划分好要单独运行划分文件
  数据集下载：https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.225b47d0sMilNQ&raceId=231677
  
use split.py
  
  
## 提取特征
  
feature_extract.py
  
  The command is:
      

    python feature_extract.py mobile model/mobile_mobile_wgt.h5

   
## MDP
  
MDP.py
  
  The command is:     
    

    python MDP.py mobile

