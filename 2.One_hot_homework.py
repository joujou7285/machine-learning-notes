class One_hot:
    def __init__(self):
        # 初始化標籤列表
        self.label_list = []

    def encode(self, label_data):
        # 根據標籤資料，建立獨熱編碼
        self.label_list = list(dict.fromkeys(label_data))  # 去重且保持順序
        one_hot_encoded = []

        for label in label_data:
            # 建立一個與標籤列表長度相同的 0 向量
            encoding = [0] * len(self.label_list)
            # 將對應位置的值設為 1
            encoding[self.label_list.index(label)]=1
            one_hot_encoded.append(encoding)


        return one_hot_encoded

    def decode(self, encode_data):
        # 將獨熱編碼轉回標籤資料
        decoded_data = []

        for encoding in encode_data:
            # 找到 1 所在的位置，並對應標籤
            index = encoding.index(1)
            decoded_data.append(self.label_list[index])

        return decoded_data


# # 測試範例
# data = ['a', 'a', 'c', 'b', 'c']
# one_hot_encoder = One_hot()
#
# # 進行編碼
# encoded_data = one_hot_encoder.encode(data)
# print("Encoded Data: ", encoded_data)
#
# # 進行解碼
# decoded_data = one_hot_encoder.decode(encoded_data)
# print("Decoded Data: ", decoded_data)
