import torch.nn as nn


# N=(W-F+2P)/S+1
class OCR_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, 2, 1, 1),  # size = 64*64*3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4, 2, 1),  # size = 64*64*64
            # Conv2
            nn.Conv2d(64, 128, 2, 1, 1),  # size = 32*32*64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(4, 2, 1),  # size = 32*32*128
            # Conv3
            nn.Conv2d(128, 256, 2, 1, 1),  # size = 16*16*128
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 1, 1),  # size = 16*16*256
            # Conv4
            nn.Conv2d(256, 512, 2, 1, 1),  # size = 8*8*256
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(4, 2, 1),  # size = 8*8*512
            # Conv5
            nn.Conv2d(512, 1024, 2, 2, 1),  # size = 4*4**512
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(4, 2, 1),  # size = 4*4*1024  ->  output = 2*2*1024
        )
        self.dense = nn.Sequential(
            # FC1
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            # FC2
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            # FC3
            nn.Linear(1024, 15),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x
