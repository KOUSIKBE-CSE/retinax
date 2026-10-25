from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LesionDataset(Dataset):
    """Loads paired images/masks with identical filename stems."""
    def __init__(self, image_dir, mask_dir, size=224, lesion='MA'):
        self.images=sorted(p for p in Path(image_dir).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})
        masks=sorted(p for p in Path(mask_dir).rglob('*') if p.suffix.lower() in {'.tif','.tiff','.png'})
        self.pairs=[]
        for image in self.images:
            candidates=[p for p in masks if p.stem.startswith(image.stem+'_') and p.stem.endswith('_'+lesion)]
            if candidates: self.pairs.append((image,candidates[0]))
        if not self.pairs: raise ValueError('No matching image/mask filename pairs found.')
        self.image_tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor()])
        self.mask_tf=transforms.Compose([transforms.Resize((size,size),transforms.InterpolationMode.NEAREST),transforms.ToTensor()])
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        image=self.image_tf(Image.open(self.pairs[i][0]).convert('RGB'))
        mask=(self.mask_tf(Image.open(self.pairs[i][1]).convert('L'))>0).float()
        return image,mask
