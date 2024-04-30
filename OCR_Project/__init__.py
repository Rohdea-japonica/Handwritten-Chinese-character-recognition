import torch
from torch import optim
from MyDataset import MyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from OCR_Model import OCR_Model

if __name__ == "__main__":
    # 一些变量
    device = ""
    lr = 0.01  # 学习率
    pre_epoch = 0  # 前一次训练的轮数
    batch_size = 512  # batch中的数据量
    writer = SummaryWriter("log")
    module = input("请输入训练模式：")
    epochs = int(input("请输入训练轮数："))

    if torch.cuda.is_available():  # 训练处理器
        device = "cuda"
    else:
        device = "cpu"

    # 获取训练数据集
    train_dataset = MyDataset()
    train_dataset.getdata("../data", module)
    train_loader = DataLoader(train_dataset, batch_size, drop_last=False)

    # 加载模型
    model = OCR_Model().to(device)
    criterion = nn.CrossEntropyLoss()  # 设置误差函数
    params = filter(lambda p: p.requires_grad, model.parameters())  # 设置模型参数跟踪
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)  # 优化器
    try:
        position = torch.load("./model.pt", map_location=device)
        model.load_state_dict(position["model"])
        pre_epoch = position["epoch"]
        optimizer.load_state_dict(position["optimizer"])
    except FileNotFoundError:
        print("Not download model!")

    if module == "train":
        model.train()  # 进入训练模式
    elif module == "dev":
        model.eval()   # 进入测试模式

    # 开始训练
    for epoch in range(epochs):
        count = 0
        correct = 0
        for x, y in train_loader:
            count += len(y)
            pred = model(x.to(device))
            optimizer.zero_grad()
            loss = criterion(pred, y.to(device))
            if module == "train":
                loss.backward()
                optimizer.step()  # 参数修改
            label = pred.argmax(1)
            for i in range(len(y)):
                if y[i] == label[i]:
                    correct += 1
        writer.add_scalar("Accuracy/Train", correct / count, epoch)  # 用于tensorboard的数据写入
        print("Current epoch is :", epoch + pre_epoch + 1, " Accuracy is :", correct / count)
        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1 + pre_epoch}
        if module == "train":
            torch.save(state_dict, "./model.pt")
    print("Finished!!!")
    writer.close()
