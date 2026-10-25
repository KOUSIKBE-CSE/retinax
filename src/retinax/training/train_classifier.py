import torch
from ..models import BaselineCNN, CNNLSTM, CNNLSTMMoE
from ..evaluation.metrics import classification_report

def run_epoch(model, loader, optimizer=None, device='cpu'):
    train=optimizer is not None; model.train(train); loss_fn=torch.nn.CrossEntropyLoss(); ys=[]; ls=[]; total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device); out=model(x); loss=loss_fn(out,y)
        if train: optimizer.zero_grad(); loss.backward(); optimizer.step()
        total+=loss.item()*len(y); ys.extend(y.cpu().tolist()); ls.append(out.detach().cpu())
    return total/len(loader.dataset), classification_report(ys,torch.cat(ls).numpy())

def train(data_root, model_name='baseline', epochs=1, batch_size=8, lr=1e-3, device=None):
    from ..data.classification_dataset import make_loaders
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); loaders=make_loaders(data_root,batch_size=batch_size)
    cls={'baseline':BaselineCNN,'cnn_lstm':CNNLSTM,'moe':CNNLSTMMoE}[model_name]().to(device); opt=torch.optim.AdamW(cls.parameters(),lr=lr); history=[]
    for epoch in range(epochs):
        tr=run_epoch(cls,loaders[0],opt,device); va=run_epoch(cls,loaders[1],None,device); history.append({'epoch':epoch+1,'train':tr,'val':va})
    test=run_epoch(cls,loaders[2],None,device)
    return cls, history, test
