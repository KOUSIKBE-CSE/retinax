import torch
from torch import nn

class AttentionGate(nn.Module):
    def __init__(self, channels, gate):
        super().__init__(); self.x=nn.Conv2d(channels,gate,1); self.g=nn.Conv2d(gate,gate,1); self.out=nn.Sequential(nn.ReLU(),nn.Conv2d(gate,1,1),nn.Sigmoid())
    def forward(self,x,g): return x*self.out(self.x(x)+self.g(g))

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__(); self.e1=self.block(in_channels,base); self.e2=self.block(base,base*2); self.e3=self.block(base*2,base*4); self.pool=nn.MaxPool2d(2); self.mid=self.block(base*4,base*8); self.a3=AttentionGate(base*4,base*8); self.a2=AttentionGate(base*2,base*4); self.a1=AttentionGate(base,base*2); self.d3=self.block(base*8+base*4,base*4); self.d2=self.block(base*4+base*2,base*2); self.d1=self.block(base*2+base,base); self.out=nn.Conv2d(base,out_channels,1)
    @staticmethod
    def block(a,b): return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); m=self.mid(self.pool(e3)); u3=nn.functional.interpolate(m,size=e3.shape[-2:],mode='bilinear',align_corners=False); u3=self.d3(torch.cat([u3,self.a3(e3,u3)],1)); u2=nn.functional.interpolate(u3,size=e2.shape[-2:],mode='bilinear',align_corners=False); u2=self.d2(torch.cat([u2,self.a2(e2,u2)],1)); u1=nn.functional.interpolate(u2,size=e1.shape[-2:],mode='bilinear',align_corners=False); return self.out(self.d1(torch.cat([u1,self.a1(e1,u1)],1)))

def bce_dice_loss(logits, target):
    bce=nn.functional.binary_cross_entropy_with_logits(logits,target); p=torch.sigmoid(logits); inter=(p*target).sum((1,2,3)); dice=1-((2*inter+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)).mean(); return bce+dice
