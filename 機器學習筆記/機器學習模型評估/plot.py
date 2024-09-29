import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

def plot_classifer(classifer,X,y):
    plt.figure()
    x_min,x_max=min(X[:,0])-1.0,max(X[:,0])+1
    y_min, y_max = min(X[:,1]) - 1.0, max(X[:,1]) + 1
    step_size=0.01
    x_vaules,y_vaules=np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))
    meshoutput=classifer.predict(np.c_[x_vaules.ravel(),y_vaules.ravel()])
    print(meshoutput)
    meshoutput=meshoutput.reshape(x_vaules.shape)
    print(meshoutput)
    plt.pcolormesh(x_vaules,y_vaules,meshoutput,cmap=plt.cm.gray)
    plt.scatter(X[:,0],X[:,1],c=y,s=80,edgecolors="black",linewidths=1,cmap=plt.cm.Paired)
    plt.xlim(x_vaules.min(),x_vaules.max())
    plt.ylim(y_vaules.min(), y_vaules.max())
    plt.xticks(np.arange(int(min(X[:,0]-1)),int(max(X[:,0]+1)),1.0))
    plt.yticks(np.arange(int(min(X[:, 1] - 1)), int(max(X[:, 1] + 1)), 1.0))
    plt.show()
def plot_confusion_matrix(confusion_mat):
    # 將混淆矩陣作為影像輸出
    plt.imshow(confusion_mat,interpolation="nearest",cmap="binary")
    plt.title("confusion matrix")
    # 設定色階
    plt.colorbar()
    # 設定刻度
    tick_number=confusion_mat.shape[0]
    tick_marks=np.arange(tick_number)
    # 設定x軸、y軸標題
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.show()
