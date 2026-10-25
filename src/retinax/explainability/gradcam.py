import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, layer): self.model,self.layer=model,layer; self.a=self.g=None; layer.register_forward_hook(lambda _,__,o: setattr(self,'a',o)); layer.register_full_backward_hook(lambda _,__,go: setattr(self,'g',go[0]))
    def __call__(self, x, class_index=None):
        self.model.zero_grad(set_to_none=True); y=self.model(x); class_index=y.argmax(1) if class_index is None else torch.tensor([class_index],device=x.device); y[torch.arange(len(y),device=x.device),class_index].sum().backward(); w=self.g.mean((2,3),keepdim=True); heat=(w*self.a).sum(1).relu(); heat=heat/(heat.amax((1,2),keepdim=True)+1e-8); return y.detach(),heat.detach()

def overlay(image, heatmap):
    im=np.asarray(image.convert('RGB').resize((heatmap.shape[-1],heatmap.shape[-2])))/255.; color=cm.jet(heatmap)[...,:3]; return Image.fromarray(np.uint8((.55*im+.45*color)*255))
