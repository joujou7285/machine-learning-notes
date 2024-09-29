class SLR:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        n = len(x_data)

        # Calculate the slope self.a
        sum_x = sum(self.x_data)
        sum_y = sum(self.y_data)
        sum_xy = sum([x * y for x, y in zip(self.x_data, self.y_data)])
        sum_x_squared = sum([x ** 2 for x in self.x_data])

        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x_squared - sum_x ** 2
        self.a = numerator / denominator

        # Calculate the intercept self.b
        self.b = (sum_y - self.a * sum_x) / n

    def predict(self, x_data):
        # Predict using the formula y = a * x + b
        output = [self.a * x + self.b for x in x_data]
        return output

# Test data
x_data = [1, 2, 3, 4, 5]
y_data = [2, 4, 5, 4, 5]

# Create an instance of SLR and fit the model
model = SLR(x_data, y_data)

# Predict using the same x_data
predict_y = model.predict(x_data)
print(predict_y)


# # 創建 SLR 類別的實例
# model = SLR(x_data, y_data)
#
# # 測試 self.a 和 self.b 的輸出
# print(f"斜率 (self.a): {model.a}")
# print(f"截距 (self.b): {model.b}")
# # 測試另一組數據
# x_data = [10, 20, 30, 40, 50]
# y_data = [15, 25, 35, 45, 55]
#
# model = SLR(x_data, y_data)
# print(f"斜率 (self.a): {model.a}")
# print(f"截距 (self.b): {model.b}")
