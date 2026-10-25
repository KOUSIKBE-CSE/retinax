import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

def classification_report(y_true, logits):
    y_true=np.asarray(y_true); p=np.asarray(logits); pred=p.argmax(1); pr,re,f1,_=precision_recall_fscore_support(y_true,pred,average=None,zero_division=0); out={'accuracy':float(accuracy_score(y_true,pred)),'macro_precision':float(pr.mean()),'macro_recall':float(re.mean()),'macro_f1':float(f1.mean()),'per_class':{'precision':pr.tolist(),'recall':re.tolist(),'f1':f1.tolist()},'confusion_matrix':confusion_matrix(y_true,pred).tolist()}
    try: out['ovr_roc_auc']=float(roc_auc_score(y_true,p, multi_class='ovr'))
    except ValueError: pass
    return out

def dice_iou(pred, target, eps=1e-7):
    pred, target = pred.float().flatten(), target.float().flatten(); inter=(pred*target).sum(); union=pred.sum()+target.sum()
    return {'dice':float((2*inter+eps)/(union+eps)), 'iou':float((inter+eps)/(union-inter+eps))}
