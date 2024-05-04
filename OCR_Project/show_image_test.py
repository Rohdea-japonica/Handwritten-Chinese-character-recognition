import torch
from PIL import Image
from torchvision import transforms
from OCR_Model import OCR_Model


def show_image_label(img_path):
    lr = 0.01
    if torch.cuda.is_available():  # 训练处理器
        device = "cuda"
    else:
        device = "cpu"
    model = OCR_Model()
    model.to(device)
    position = torch.load("./model.pt", map_location=device)
    model.load_state_dict(position["model"])
    model.eval()
    img = Image.open(img_path).convert("RGB")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    tensor_img = tf(img)
    tensor_img.unsqueeze_(0)
    label = model(tensor_img.to(device)).argmax(1).item()
    img.show()
    print(label)


img_path = input("请输入想要预测的图片的地址：")
show_image_label(img_path)
