# 多項式回歸器
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from tool_metric import show_metrics
from sklearn.preprocessing import PolynomialFeatures
import pickle
# 抓取cmd輸入的參數
filename = sys.argv[1]
print(sys.argv)
# 定義x, y空列表
x = []
y = []
# 讀取檔案並把每一行寫入x,y
with open(filename, 'r') as f:
    for line in f.readlines():
        #print(line.split(','))
        xt, yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)
num_training=int(0.8*len(x))
num_test=len(x)-num_training
x_train=np.array(x[:num_training]).reshape(num_training,1)
y_train=np.array(y[:num_training])
x_test=np.array(x[num_training:]).reshape(num_test,1)
y_test=np.array((y[num_training:]))
polynomail=PolynomialFeatures(degree=3)
x_train_transformed=polynomail.fit_transform(x_train)
x_test_transformed=polynomail.fit_transform(x_test)

linear_regressor=linear_model.LinearRegression()
linear_regressor.fit(x_train_transformed,y_train)
y_test_predict=linear_regressor.predict(x_test_transformed)
show_metrics(y_test,y_test_predict)
output_file="saved_model.pkl"
# with open(output_file,"wb")as f:
#     pickle.dump(linear_regressor,f)

# 產生空白畫布
plt.figure()
# 畫散布圖
plt.scatter(x_train, y_train, color='green')
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test, y_test_predict, color='blue', label='Prediction')
# 設定標題
plt.title('linear_regression')
# 展示數據圖
plt.show()