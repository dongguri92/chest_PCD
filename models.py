import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

# efficientnet model
class Custom_Efficientnet(nn.Module):
    def __init__(self):
        super(Custom_Efficientnet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4', in_channels=3)
        self.output_layer = nn.Linear(1000, 2)

    def forward(self, x):
        x = F.relu(self.efficientnet(x))
        x = F.softmax(self.output_layer(x), dim=1)
        return x

def modeltype(model):
    if model == "efficientnet":
        return Custom_Efficientnet()