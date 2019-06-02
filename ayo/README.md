python2.7
 GeForce GTX 1080
1、train_main.py训练给出的160类图片
2、pred.py提取出最后一层FC的特征
3、dap/dap_main.py对每个属性训练直接属性分类器
4、lasso_map.py利用lasso根据词向量求出seen对于unseen的映射关系W
5、count.py对于五类分类器网络得到的结果进行投票


