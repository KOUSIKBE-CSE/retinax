"""Small Flower-compatible building blocks; raw datasets remain client-local."""
import torch
from collections import OrderedDict

def get_parameters(model): return [v.detach().cpu().numpy() for _,v in model.state_dict().items()]
def set_parameters(model, parameters): model.load_state_dict(OrderedDict((k,torch.tensor(v)) for k,v in zip(model.state_dict(),parameters)), strict=True)
def fedprox_loss(loss, model, global_parameters, mu=.01):
    penalty=sum((p-g.to(p.device)).pow(2).sum() for p,g in zip(model.parameters(),global_parameters)); return loss + mu/2*penalty

def split_indices(n, clients=5, non_iid=False, labels=None, seed=42):
    import numpy as np
    rng=np.random.default_rng(seed); idx=np.arange(n)
    if not non_iid or labels is None: return [x.tolist() for x in np.array_split(rng.permutation(idx),clients)]
    order=np.argsort(labels); return [x.tolist() for x in np.array_split(order,clients)]
