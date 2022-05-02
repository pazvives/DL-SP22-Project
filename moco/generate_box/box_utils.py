
import torch.nn.functional as F
import math
import os
import yaml
import random
from   PIL import Image
import torch
import torchvision.transforms as T
import os
import cv2
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from gradcam import GradCAM


def save_boxes(d, path):
    """"Save boxes from dictionary"""
    with open(path, "w") as f:
        for img_id, boxes in d.items():
            line = [str(img_id)]
            for box in boxes:
                line.append(",".join(map("{:.6f}".format, box)))
            print('line')
            print(line)
            f.write(" ".join(line) + '\n')


def get_normalization(normalize):
    """Get normalization values for dataset"""
    if normalize == 'redo':
        return T.Normalize(mean=0.5, std=0.5)
    else:  # default: ImageNet
        return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class FinetuneTransform:
    """Base transformations for fine-tuning"""
    def __init__(self, image_size=224, crop='center', normalize=None):
        assert crop in ['random', 'center', 'none']

        transforms = []

        if crop == 'random':
            transforms += [
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        elif crop == 'center':
            transforms += [
                T.Resize(image_size) if image_size != 224 else transforms.Resize(256),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        else:
            transforms += [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]

        if normalize is not None:
            transforms.append(normalize)

        self.transform = T.Compose(transforms)

    def __call__(self, sample):
        return self.transform(sample)



class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def reset_parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()
        if isinstance(m, nn.Linear):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -bound, bound)

def load_projection(n_in, n_hidden, n_out, num_layers=2, last_bn=False):
    layers = []
    for i in range(num_layers - 1):
        layers.append(nn.Linear(n_in, n_hidden, bias=False))
        layers.append(nn.BatchNorm1d(n_hidden))
        layers.append(nn.ReLU())
        n_in = n_hidden
    layers.append(nn.Linear(n_hidden, n_out, bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(n_out))
    layers.append(Lambda(F.normalize))  # normalize projection
    projection = nn.Sequential(*layers)
    reset_parameters(projection)
    return projection



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



def collate_fn(batch):
    return tuple(zip(*batch))




def clean_mask(masks, single=False, min_obj_scale=0.001):
    """Clean small noisy segmentations"""
    #b, c, h, w = masks.size()
    #masks = masks.detach().cpu()

    outs = []
    for mask in tqdm(masks, desc='Cleaning masks...'):
        # winnie add this squeeze  
        mask = mask.squeeze(0)
        # print('clean mask after squeeze')
        # print(mask.size())
        c, h, w = mask.size()

        mask = mask.detach().cpu()
        print('to cpu done')
        mask = np.array(T.ToPILImage()(mask))
        print('to PILimage done')
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

        if single:
            idxs = [stats[1:, -1].argmax() + 1] if len(stats) > 1 else []
        else:
            min_obj_size = h * w * min_obj_scale
            idxs = (stats[:, -1] > min_obj_size).nonzero()[0][1:]

        out = torch.zeros(output.shape)
        for idx in idxs:
            out[output == idx] = 1
        outs.append(out.view(1, h, w))

    return torch.stack(outs, dim=0)


def extract_boxes(masks, image_size=224, threshold=0.5, margin=0, largest_only=False):
    """Extract all boxes from masks"""
    all_boxes = []
    for mask in masks:
        boxes = _extract_box(mask, threshold=threshold)

        if largest_only:
            areas = [box[2] * box[3] for box in boxes]
            boxes = [boxes[np.argmax(areas)]]

        for i, box in enumerate(boxes):
            box = xywh_to_xyxy(box)
            box = expand_box(box, image_size, margin=margin)
            boxes[i] = box

        all_boxes.append(boxes)

    return all_boxes


def _extract_box(mask, min_obj_scale=0.01, threshold=0.5, quantile=1.):
    """Extract boxes from mask"""
    # print('Extract boxes from mask')
    # winnie add this squeeze  
    mask = mask.squeeze(0)
    # print(mask.size())
    # potential bug fixed by Winnie ==> check with others if they agree with the fix

    _, h, w = mask.size()
    #_,_,h,w = mask.size()
    mask = mask.detach().cpu()

    threshold *= mask.view(1, -1).quantile(q=quantile, dim=1)

    mask = (mask > threshold).float()
    mask = np.array(T.ToPILImage()(mask))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        box = cv2.boundingRect(cnt)
        min_obj_size = h * w * min_obj_scale
        if box[2] * box[3] > min_obj_size:
            boxes.append(box)
    return boxes

def xywh_to_xyxy(box):
    """Convert box from xywh to xyxy format"""
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    box = (x1, y1, x2, y2)
    return box


def xyxy_to_xywh(box):
    """Convert box from xyxy to xywh format"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    box = (x1, y1, w, h)
    return box


 
def expand_box(box, image_size=(1, 1), margin=0.2):
    """Expand box with margin and downscale"""
    x1, y1, x2, y2 = box
    #print(‘inside expand box’)
    #print(box)
    W, H = get_image_size(image_size)
    #print(‘image_size’)
    #print(W)
    #print(H)
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * margin) / W
    y1 = max(0, y1 - h * margin) / H
    x2 = min(W, x2 + w * margin) / W
    y2 = min(H, y2 + h * margin) / H
    box = (x1, y1, x2, y2)
    #print(‘after exand’)
    #print(box)
    return box




def get_image_size(image_size):
    """Convert image_size to (H,W) format"""
    if isinstance(image_size, (list, tuple)):
        W, H = image_size
    else:
        W = H = image_size
    return W, H



def collect_outputs(model, loader, device='cuda', **kwargs):
    outs, labels = [], []
    ii = 0
    for (x, _) in loader:#tqdm(loader):
        #TODO: why do we have 3 images here! ==> NOT THREE IMAGES,THREE CHANNELS
       
        # len(x[0]) --> 3
        # x[0][0], x[0][1], x[0][2] --> are all tensors of shape [500, 375]) ==> NO LONGER TRUE
        # for me all tensosr of shape [3,224,224] ==> I think 3 is because 3 RGB channel , 224 IS BECAUSE CROP 224 AT CENTER
        # Keeping the first only here
        #x = x[0][2]
        
        # print('before get to model')
        # print(x.size()) # shnape [1,3,224,224]
        #print(x[0].size()) # shape [3,224,224]


        ################   PLOT TO VISUALIZE
        #print(ii)
        #fig = plt.figure()
        #plt.imshow(np.array(x[0][0]))
        #plt.title(ii)
        ################
                
        out = model(x.to(device)).cpu()#, **kwargs).cpu() # this is the key part! here the input is feed in to the gradCAM model to comput CAM 
        outs.append(out)
        ii =ii +1
        #labels.append(y)
    print("Exit")
    #print(outs)
    outs_changed = []
 
    #for out in outs:
      #This solves it
      #print(out.size())
      #out_changed = out.view(-1, 500, 1,1)
      #print(out_changed.size())
      #outs_changed.append(out_changed)
  #outs = torch.cat(outs_changed)
 
    #labels = torch.cat(labels)

    return outs #, labels

# copied form the original code for debugging, delete later
def collect_outputs_original(model, loader, device='cuda', **kwargs):
    outs, labels = [], []
    for x, y in tqdm(loader):
        out = model(x.to(device), **kwargs).cpu()
        outs.append(out)
        labels.append(y)
    outs = torch.cat(outs)
    labels = torch.cat(labels)
    return outs, labels

def accuracy(X, Y, classifier):
    with torch.no_grad():
        preds = classifier(X).argmax(1)
    acc = (preds == Y).float().mean().item()
    return acc

def compute_masks(model, loader, cam_iters = 1):
    forward_kwargs = {}
    if isinstance(model, GradCAM):
        forward_kwargs['score_type'] = 'con'
        forward_kwargs['n_iters'] = cam_iters

    x = collect_outputs(model, loader, **forward_kwargs)
    return x


