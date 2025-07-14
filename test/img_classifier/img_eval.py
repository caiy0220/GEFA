import torch
import shap
import numpy as np
from gefa.utils import log
from gefa.gefa import GEFA4Image
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
from torchvision import models
from torch import nn
from torch.nn import functional as F
import random

PIX_MEAN_IMAGENET, PIX_STD_IMAGENET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PV_FLOOR_IMAGENET = (-torch.tensor(PIX_MEAN_IMAGENET) / torch.tensor(PIX_STD_IMAGENET)).reshape(-1, 1, 1)

# Get image size by model name
IMG_SIZE_DICT = {
    'inception': (3, 299, 299),
    'vit': (3, 224, 224),

}

# Get normalization statistics by dataset name
PIX_STATS_DICT = {
    'imagenet': (PIX_MEAN_IMAGENET, PIX_STD_IMAGENET)
}

M_NAME_DICT = {
    'vit': models.vit_b_16,
    'resnet': models.resnet50,
    'inception': models.inception_v3,
}

M_WEIGHTS_DICT = {
    'vit': models.ViT_B_16_Weights.IMAGENET1K_V1,
    'resnet': models.ResNet50_Weights.IMAGENET1K_V1,
    'inception': models.Inception_V3_Weights.IMAGENET1K_V1,
}

def unified_explain(expl, m, x, args):
    lbl = m(x.cuda()).cpu().argmax().item()
    if isinstance(expl, shap.Explainer):
        res = expl(x.permute(0,2,3,1), 
                   max_evals=args.num_samples, 
                   batch_size=64,
                   outputs=shap.Explanation.argsort.flip[:1])
        attr = np.mean(res.values[0,:,:,:,0], axis=-1)
        return torch.tensor(attr)
    else:
        return expl.explain(m, x)

# Sorting features (pixels) in descending order
def get_pix_ranking(rel):
    rel = rel.flatten()
    # pairs = zip(rel, list(range(len(rel))))
    # return sorted(pairs, key=lambda x:x[0], reverse=True)
    res = rel.sort(descending=True)
    return list(zip(res[0].tolist(), res[1].tolist()))

def expl_guided_edition(m, img_org, rel, lbl, num=100, v_default=0, stride=1, batch_size=64, pbar=None, do_random=False):
    """
    Parameters:
        m: model
        img_org: torch.Tensor in shape [1, C, H, W]
        rel: attribution matrix in shape [H, W]
    """
    img = img_org.clone()
    if do_random:
        pix_ranking = list(range(rel.shape.numel()))
        pix_ranking = list(zip(pix_ranking, pix_ranking))
        random.shuffle(list(pix_ranking))
    else:
        pix_ranking = get_pix_ranking(rel)

    pred_manipul = [m(img.cuda()).detach().cpu()[0][lbl]]
    ptr = 0
    # flag_subtask = pbar is not None
    # pbar = pbar if pbar is not None else tqdm(total=num)

    while ptr < num:
        buff = []
        while len(buff) < batch_size:
            start, end = ptr, min(ptr + stride, num)
            img = mask_out_pix(img, pix_ranking[start: end], v_default)
            buff.append(img.clone())
            ptr += stride
            if ptr >= num: break
        preds = list(m(torch.cat(buff).to('cuda')).cpu().detach()[:, lbl])
        pred_manipul += preds

        if pbar: 
            pbar.set_postfix_str('Current edition: [{}/{}]'.format(ptr, num))
        # else:
        #     pbar.update(batch_size)
    return pred_manipul, img

def evaluate_via_deletion(expl, m, test_loader, input_shape, del_ratio=1.0, v_default=0.1307, 
                          num_instances=None, stride=10, batch_size=64):
    """
    Parameters:
        expl: explainer instance
        m: the to-be-explained model
        input_shape:    [tuple] required for the preparation of the default value matrix
        del_ratio:      [float] the ratio of features to be removed, 1.0 indicates total removal
        v_default: the value for replacement, supported values:
            1. [int/float] identical default value for all features with single constant
            2. [torch.Tensor] of shape [C, H, W], Gaussian noise 
            3. [str] 'blur' specifying a blurred version of explicand as absence values
        stride: [int] the period (in steps of deletion) of querying the model for confidence drops 
    """
    v_default = prepare_default(v_default, input_shape)
    num_del = int(input_shape[-1] * input_shape[-2] * del_ratio)
    test_handler = iter(test_loader) 
    records, count = [], 0 
    num_instances = num_instances if num_instances else len(test_loader)*test_loader.batch_size
    log(f'Num of instances for evaluation: {num_instances}')

    blur_m = GaussianBlur((31, 31), 5.0)
    with tqdm(total=num_instances, desc='#Instances') as pbar:
        for batch in test_handler:
            imgs, _ = batch
            for img in imgs:
                img = img.unsqueeze(dim=0)
                v = blur_m(img) if v_default =='blur' else v_default
                v = v[0] if len(v.shape) == 4 else v
                logits = m(img.cuda()).detach().cpu()
                num_classes = len(logits[0])

                # Get the initally predicted class, whose confidence drop is supervised
                _, lbl = torch.max(logits, dim=1)
                components = [img, lbl, m] + [expl] 
                rel = unified_explain(components)

                # Start of the explanation-guided deletion process for the current explicand
                trend, _ = expl_guided_edition(m, img, rel, lbl, v_default=v, 
                                            num=num_del, stride=stride, pbar=pbar,
                                            batch_size=batch_size)
                records.append(trend)
                count += 1
                pbar.update(1)
                if count >= num_instances:
                    return records
    return records

def prepare_default(v, shape):
    if isinstance(v, str):
        v = v.lower()
    elif isinstance(v, int) or isinstance(v, float):
        v = torch.tile(torch.tensor(v), (1,)+shape[-2:])
    return v

def mask_out_pix(img, idxs, v_default):
    for _, idx in idxs:
        rid, cid = idx2pos(idx, img.shape[-1]) 
        v = v_default if isinstance(v_default, (int, float)) else v_default[:, rid, cid]
        img[0,:,rid,cid] = v
    return img

def compute_aopc(trend):
    return 1. - torch.mean(trend) / trend[0]

def compute_auc(trend):
    return torch.mean(trend) 

def idx2pos(idx, col=28):
    row_id = idx // col 
    col_id = idx % col 
    return row_id, col_id

def attr_quantization(x):
    x = x / x.abs().max() * 1000
    return x.to(torch.int16)

class SoftmaxWrapping(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def parameters(self, recurse: bool = True):
        return self.m.parameters(recurse)

    def forward(self, x):
        return F.softmax(self.m(x), dim=1)
    
class ImageNetSampler_OnePerClass(torch.utils.data.Sampler):
    def __init__(self, ds):
        self.indices = [[] for _ in range(len(ds.classes))]
        for i, (_, l) in enumerate(ds.samples):
            self.indices[l].append(i)

        self.batches = []
        for i in range(len(self.indices)):
            self.batches.append(self.indices[i][0])

    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.indices)
