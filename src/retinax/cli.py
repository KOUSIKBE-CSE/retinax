import argparse, json
import random
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision.datasets import ImageFolder

def main():
    p=argparse.ArgumentParser('retinax'); sub=p.add_subparsers(dest='cmd',required=True); i=sub.add_parser('inspect-data'); i.add_argument('--data-root',required=True); d=sub.add_parser('make-demo-data'); d.add_argument('--output',default='./demo_data'); d.add_argument('--images-per-class',type=int,default=4); t=sub.add_parser('train'); t.add_argument('--data-root',required=True); t.add_argument('--model',choices=['baseline','cnn_lstm','moe'],default='baseline'); t.add_argument('--epochs',type=int,default=10); t.add_argument('--batch-size',type=int,default=16); t.add_argument('--learning-rate',type=float,default=1e-3); t.add_argument('--checkpoint',default='checkpoints/retinax_model.pt'); s=sub.add_parser('train-segmentation'); s.add_argument('--images',required=True); s.add_argument('--masks',required=True); s.add_argument('--lesion',choices=['MA','HE','EX','SE'],default='MA'); s.add_argument('--epochs',type=int,default=10); s.add_argument('--batch-size',type=int,default=4); s.add_argument('--checkpoint',default='checkpoints/retinax_unet.pt')
    a=p.parse_args()
    if a.cmd=='train-segmentation':
        from .training.train_segmentation import train_segmentation
        print(json.dumps(train_segmentation(a.images,a.masks,a.lesion,a.epochs,a.batch_size,a.checkpoint),indent=2))
    elif a.cmd=='make-demo-data':
        classes=['No_DR','Mild','Moderate','Severe','Proliferate_DR']; root=Path(a.output); rng=random.Random(42)
        for label in classes:
            folder=root/label; folder.mkdir(parents=True,exist_ok=True)
            for n in range(a.images_per_class):
                image=Image.new('RGB',(224,224),(rng.randrange(20,60),rng.randrange(20,60),rng.randrange(20,60))); draw=ImageDraw.Draw(image)
                draw.ellipse((28,28,196,196),outline=(130,80,40),width=3); draw.ellipse((75,75,149,149),fill=(15,15,15)); draw.ellipse((95+n,95,101+n,101),fill=(200,30,30))
                image.save(folder/f'demo_{n:04d}.jpg')
        print(f'Created {len(classes)*a.images_per_class} synthetic images at {root}. Not for medical training or evaluation.')
    elif a.cmd=='inspect-data':
        ds=ImageFolder(Path(a.data_root)); print(json.dumps({'classes':ds.classes,'images':len(ds),'counts':Counter(y for _,y in ds.samples)},indent=2))
    else:
        from .training.train_classifier import train
        import torch
        model, history, test = train(a.data_root,a.model,a.epochs,a.batch_size,a.learning_rate)
        checkpoint=Path(a.checkpoint); checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'model':model.state_dict(),'model_name':a.model,'classes':model_classes(),'history':history,'test':test},checkpoint)
        print(json.dumps({'checkpoint':str(checkpoint),'history':history,'test':test},indent=2))

def model_classes():
    from . import CLASS_NAMES
    return CLASS_NAMES
if __name__=='__main__': main()
