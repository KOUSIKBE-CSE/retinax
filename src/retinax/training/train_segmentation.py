import torch
from torch.utils.data import DataLoader, random_split
from ..data.segmentation_dataset import LesionDataset
from ..models.attention_unet import AttentionUNet, bce_dice_loss

def train_segmentation(image_dir, mask_dir, lesion='MA', epochs=10, batch_size=4, checkpoint='checkpoints/retinax_unet.pt', device=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); ds=LesionDataset(image_dir,mask_dir,lesion=lesion); n=max(1,int(len(ds)*.8)); tr,va=random_split(ds,[n,len(ds)-n]); model=AttentionUNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); history=[]
    for epoch in range(epochs):
        model.train(); total=0
        for x,y in DataLoader(tr,batch_size=batch_size,shuffle=True):
            x,y=x.to(device),y.to(device); loss=bce_dice_loss(model(x),y); opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()*len(y)
        model.eval(); val=0
        with torch.no_grad():
            for x,y in DataLoader(va,batch_size=batch_size): val+=bce_dice_loss(model(x.to(device)),y.to(device)).item()*len(y)
        history.append({'epoch':epoch+1,'train_loss':total/len(tr),'val_loss':val/max(1,len(va))})
    path=__import__('pathlib').Path(checkpoint); path.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'history':history},path); return history
