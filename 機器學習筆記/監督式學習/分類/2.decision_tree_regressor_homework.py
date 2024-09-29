import numpy as np
def decision(X_data,y_data):
    X_array=np.array(X_data)
    y_array = np.array(y_data)
    min_MSE=len(y_array)*(y_array.std()**2)
    for i in range(X_array.shape[1]):
        X_clip=X_array[:,i]
        X_sorted=np.sort(X_clip)
        y_sorted=y_array[np.argsort(X_clip)]
        for j in range(len(X_sorted)-1):
            y_left=y_sorted[:j+1]
            y_right=y_sorted[j+1:]
            MSE_total=len(y_left)*(y_left.std()**2)+len(y_right)*(y_right.std()**2)
            if min_MSE>MSE_total:
                min_MSE=MSE_total
                feature_index=i
                feature_value=X_sorted[j+1]
        return feature_index,float(feature_value)

X_data=[[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
y_data=[0,1,2,3]
print(decision(X_data,y_data))












