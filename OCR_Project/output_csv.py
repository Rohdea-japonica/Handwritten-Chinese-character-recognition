import torch
import pandas as pd
from torch.optim import optimizer
from torch.utils.data import DataLoader

from OCR_Project import OCR_Model, MyDataset

if torch.cuda.is_available():  # 训练处理器
    device = "cuda"
else:
    device = "cpu"
batch_size = 512
module = input("请输入训练模式：")

# 获取训练数据集
train_dataset = MyDataset()
train_dataset.getdata("../data", module)  # 注意，此处使用test模式，需要在data文件夹下有一个test文件夹用于存放图片数据
train_loader = DataLoader(train_dataset, batch_size, drop_last=False)

# 加载模型并用于预测
model = OCR_Model()
model.to(device)
position = torch.load("./model.pt", map_location=device)
model.load_state_dict(position["model"])
model.eval()
labels = []
# 开始预测
for x, y in train_loader:
    pred = model(x.to(device))
    label = pred.argmax(1)
    labels = labels + label.tolist()

df = pd.read_csv(module + "_dict.csv")
df["label"] = labels
df.to_csv(module + "_dict.csv", index=None)
