import torch
from retinax.models import BaselineCNN, CNNLSTM, CNNLSTMMoE
from retinax.segmentation import AttentionUNet

def test_shapes():
    x=torch.randn(2,3,224,224)
    assert BaselineCNN()(x).shape==(2,5); assert CNNLSTM()(x).shape==(2,5); assert CNNLSTMMoE()(x).shape==(2,5); assert AttentionUNet()(x).shape==(2,1,224,224)
