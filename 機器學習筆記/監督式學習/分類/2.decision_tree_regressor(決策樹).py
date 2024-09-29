import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from tool_metric import  show_metrics
from sklearn.tree import plot_tree
housing_data=datasets.fetch_california_housing()
# print(housing_data.target)
# print(housing_data.data)
X,y=shuffle(housing_data.data,housing_data.target,random_state=7)
num_training=int(0.8*len(X))
X_train,y_train=X[:num_training],y[:num_training]
X_test,y_test=X[num_training:],y[num_training:]
dt_regresoor=DecisionTreeRegressor(max_depth=4)
dt_regresoor.fit(X_train,y_train)
y_pred_dt=dt_regresoor.predict(X_test)
# print(len(X_test))
# print(y_pred_dt)
show_metrics(y_test,y_pred_dt)
print(dt_regresoor.feature_importances_)
feature_importance=100.0*dt_regresoor.feature_importances_/max(dt_regresoor.feature_importances_)
print(feature_importance)
print(np.flipud(np.argsort(feature_importance)))
index_sorted=np.flipud(np.argsort(feature_importance))
feature_names=[housing_data.feature_names[index_sorted[i]] for i in range(len(index_sorted))]
print(feature_names)
pos=np.arange(index_sorted.shape[0])+0.5
print(pos)
# plt.figure()
# plt.bar(pos,feature_importance[index_sorted],align="center")
# plt.xticks(pos,feature_names)
# plt.ylabel("relative importance")
# plt.title("DT-regressor")

plt.figure(figsize=(15,20))
plot_tree(dt_regresoor,filled=True,feature_names=housing_data.feature_names)
plt.show()