from OCR_Model import OCR_Model
import torch

if torch.cuda.is_available():  # 训练处理器
    device = "cuda"
else:
    device = "cpu"


model = OCR_Model()
model.to(device)
position = torch.load("./model.pt", map_location=device)
print(position["model"])
with open("model_weight", 'w') as f:
    f.write(position["model"])
