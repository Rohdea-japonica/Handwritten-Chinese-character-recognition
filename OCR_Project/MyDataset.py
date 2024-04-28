import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class MyDataset(Dataset):
    def __init__(self):
        self.images = []
        self.labels = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img = Image.open(img).convert("RGB")
        # ImageNet推荐的RGB通道均值、方差
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        img = tf(img)
        label = torch.tensor(label)
        return img, label

    def getdata(self, data_path, mode):
        dict_path = os.path.join(data_path, mode + "_dict.csv")
        img_root = os.path.join(data_path, mode)
        content = pd.read_csv(dict_path)
        suite_id = content["suite_id"]
        sample_id = content["sample_id"]
        code = content["code"]
        label = content["label"]
        labels = []
        image_names = []
        for i in range(len(content)):
            file_path = "input_" + str(suite_id[i]) + "_" + str(sample_id[i]) + "_" + str(code[i]) + ".jpg"
            labels.append(label[i])
            image_names.append(os.path.join(img_root, file_path))
        self.images = image_names
        self.labels = labels

