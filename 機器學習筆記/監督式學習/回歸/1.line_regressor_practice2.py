import numpy as np
class LR:
    def __init__(self, regularization_factor=0.0):
        # 初始化係數，設為 None
        self.cof = None
        self.regularization_factor = regularization_factor

    def fit(self, x_data, y_data):
        # 將 x_data 和 y_data 轉換為 numpy array 格式
        X = np.array(x_data)
        y = np.array(y_data)

        # 加一列 1 來表示常數項 b
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # 正則化矩陣 (X^T * X + regularization_factor * I)
        I = np.eye(X.shape[1])  # 單位矩陣
        I[0, 0] = 0  # 不正則化常數項
        regularized_term = self.regularization_factor * I

        # 使用正規方程計算係數
        self.cof = np.linalg.inv(X.T @ X + regularized_term) @ (X.T @ y)

    def predict(self, x_data):
        # 將 x_data 轉換為 numpy array
        X = np.array(x_data)

        # 加一列 1 來表示常數項 b
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # 使用已經訓練出的係數計算預測值
        return X @ self.cof

# # 測試範例
# if __name__ == "__main__":
#     # 輸入資料範例
#     x_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 每組數據包含 3 個特徵
#     y_data = [10, 20, 30]  # 每組數據對應的 y 值
#
#     # 創建線性回歸模型，加入正則化項
#     model = LR(regularization_factor=1e-5)
#
#     # 訓練模型
#     model.fit(x_data, y_data)
#
#     # 使用模型進行預測
#     x_test = [[2, 3, 4], [5, 6, 7]]  # 測試用的輸入數據
#     predictions = model.predict(x_test)
#
#     # 輸出結果
#     print("係數:", model.cof)
#     print("預測值:", predictions)


