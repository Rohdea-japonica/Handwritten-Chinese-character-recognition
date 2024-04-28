import torch
from torch import optim
from MyDataset import MyDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from OCR_Model import OCR_Model

if __name__ == "__main__":
    # 一些变量
    lr = 0.01  # 学习率
    batch_size = 512  # batch中的数据量
    device = ""
    epochs =int(input("请输入训练轮数："))

    if torch.cuda.is_available():  # 训练处理器
        device = "cuda"
    else:
        device = "cpu"

    # 获取训练数据集
    train_dataset = MyDataset()
    train_dataset.getdata("../data", "train")
    train_loader = DataLoader(train_dataset, batch_size, drop_last=False)

    # 加载模型
    model = OCR_Model().to(device)
    try:
        model.load_state_dict(torch.load("./model.pt", map_location=device))
    except FileNotFoundError:
        pass

    params = filter(lambda p: p.requires_grad, model.parameters())  # 设置模型基础参数
    criterion = nn.CrossEntropyLoss()  # 设置误差函数
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)  # 优化器
    model.train()  # 进入训练模式

    # 开始训练
    for epoch in range(epochs):
        count = 0
        correct = 0
        step = 0
        for x, y in train_loader:
            x.to(device)
            y.to(device)
            count += len(y)
            step += 1
            pred = model(x)
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()    # 参数修改
            label = pred.argmax(1)
            print("step:", step)
            for i in range(len(y)):
                if y[i] == label[i]:
                    correct += 1
        print("Current epoch is :", epoch, " Accuracy is :", correct / count)
        torch.save(model.state_dict(), "./model.pt")


