import numpy as np
#导入画图工具
import matplotlib.pyplot as plt
#导入支持向量机svm
from sklearn import svm
#导入数据集生成工具
from sklearn.datasets import make_blobs

#创建50个数据点，分为2类
x,y=make_blobs(n_samples=50,centers=2,random_state=6)
# #创建一个线性内核的支持向量机模型
# clf=svm.SVC(kernel='linear',C=1000)
# clf.fit(x,y)

#创建一个RBF内核的支持向量机模型
clf_rbf=svm.SVC(kernel='rbf',C=10)
clf_rbf.fit(x,y)
#把数据点画出来
plt.scatter(x[:,0],x[:,1],c=y,s=30,cmap=plt.cm.Paired)

#建立图像坐标
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()


#生成两个等差数列
xx = np.linspace(xlim[0],xlim[1],30)
yy = np.linspace(ylim[0],ylim[1],30)
YY,XX = np.meshgrid(yy,xx)

xy = np.vstack([XX.ravel(),YY.ravel()]).T

# Z = clf.decision_function(xy).reshape(XX.shape)

Z = clf_rbf.decision_function(xy).reshape(XX.shape)
#把分类的决定边界画出来
ax.contour(XX,YY,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])

# ax.scatter(clf.support_vectors_[:, 0],clf.support_vectors_[:, 1],s=100,linewidth=1,facecolors='none')

ax.scatter(clf_rbf.support_vectors_[:, 0],clf_rbf.support_vectors_[:, 1],s=100,linewidth=1,facecolors='none')
plt.show()

