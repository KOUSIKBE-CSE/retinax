import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def backbone(pretrained=False):
    m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    return nn.Sequential(*list(m.children())[:-2]), 512

class BaselineCNN(nn.Module):
    def __init__(self, classes=5, pretrained=False):
        super().__init__(); self.features, c = backbone(pretrained); self.pool=nn.AdaptiveAvgPool2d(1); self.classifier=nn.Linear(c, classes)
    def forward(self, x): return self.classifier(self.pool(self.features(x)).flatten(1))

class CNNLSTM(nn.Module):
    def __init__(self, classes=5, hidden=256, pretrained=False):
        super().__init__(); self.features, c=backbone(pretrained); self.proj=nn.Linear(c, hidden); self.lstm=nn.LSTM(hidden, hidden, batch_first=True); self.head=nn.Linear(hidden, classes)
    def forward(self,x):
        f=self.features(x).flatten(2).transpose(1,2); out,_=self.lstm(self.proj(f)); return self.head(out[:,-1])

class CNNLSTMMoE(nn.Module):
    def __init__(self, classes=5, experts=3, hidden=256, pretrained=False):
        super().__init__(); self.features,c=backbone(pretrained); self.proj=nn.Linear(c,hidden); self.lstm=nn.LSTM(hidden,hidden,batch_first=True); self.gate=nn.Linear(hidden,experts); self.experts=nn.ModuleList([nn.Linear(hidden,classes) for _ in range(experts)])
    def forward(self,x, return_gates=False):
        f=self.features(x).flatten(2).transpose(1,2); o,_=self.lstm(self.proj(f)); h=o[:,-1]; g=self.gate(h).softmax(-1); logits=torch.stack([e(h) for e in self.experts],1); y=(g.unsqueeze(-1)*logits).sum(1); return (y,g) if return_gates else y
