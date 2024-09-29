from sklearn.naive_bayes import GaussianNB
from plot import plot_classifer,plot_confusion_matrix
from sklearn import datasets,model_selection
from sklearn.metrics import confusion_matrix
data=datasets.load_iris()
X_train,X_test,y_train,y_test=model_selection.train_test_split(data.data[:,0:2],data.target,test_size=0.25,random_state=1)
# print(X_train)
classifer_gausiannb=GaussianNB()
classifer_gausiannb.fit(X_train,y_train)
y_pred=classifer_gausiannb.predict(X_test)
confusion_mat=confusion_matrix(y_test,y_pred)
print(confusion_mat)
# accuracy=100.0*(y_test==y_pred).sum()/len(y_test)
# print("準確度：",accuracy,"%")
plot_confusion_matrix(confusion_mat)
plot_classifer(classifer_gausiannb,data.data[:,0:2],data.target)
accuracy=model_selection.cross_val_score(classifer_gausiannb,data.data[:,0:2],data.target,scoring="accuracy",cv=5)
precision=model_selection.cross_val_score(classifer_gausiannb,data.data[:,0:2],data.target,scoring="precision_weighted",cv=5)
recall=model_selection.cross_val_score(classifer_gausiannb,data.data[:,0:2],data.target,scoring="recall_weighted",cv=5)
print("準確度為:",round(100*accuracy.mean(),2),"%")
print("精確度為:",round(100*precision.mean(),2),"%")
print("召回率為:",round(100*recall.mean(),2),"%")
