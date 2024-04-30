import torch
from PIL import Image
import torch.nn as nn
from torch import optim
from torchvision import transforms
from OCR_Project import OCR_Model


def show_image_label(img_path, model):
    img = Image.open(img_path).convert("RGB")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    tensor_img = tf(img)
    label = model(tensor_img)
    print(img, label)


lr = 0.01
if torch.cuda.is_available():  # 训练处理器
    device = "cuda"
else:
    device = "cpu"
img_path = input("请输入想要预测的图片的地址：")
model = OCR_Model().to(device)
criterion = nn.CrossEntropyLoss()  # 设置误差函数
params = filter(lambda p: p.requires_grad, model.parameters())  # 设置模型参数跟踪
optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)  # 优化器
position = torch.load("./model.pt", map_location=device)
model.load_state_dict(position["model"])
show_image_label(img_path, model)
