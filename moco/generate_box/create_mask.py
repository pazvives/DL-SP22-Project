# ALL BOX RELATED UTILS
import os
import yaml
import random
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import time
from collections import OrderedDict
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm

import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
#import transforms as T

# Moco related imports
import moco
import moco.builder as moco_builder
import moco.loader as moco_loader

# Demo related imports
#from demo_mocov2 import get_transform, get_model
# import utils
from dataset import class_dict

from gradcam import GradCAM

import box_utils

from box_utils import get_normalization, FinetuneTransform, save_boxes, compute_masks, clean_mask, extract_boxes, load_projection




class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.transform = transform

        self.image_dir  = root
        self.num_images = len(os.listdir(self.image_dir))
        self.offset = 37490       # To change
    
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # the idx of labeled image is from 0 (not, is 1!) #TODO: review in real dataset
        with open(os.path.join(self.image_dir, f"{idx+self.offset}.PNG"), 'rb') as f:
            img = Image.open(f).convert('RGB')

        return self.transform(img)





def get_image_ids(dataset):
    #TODO: correct is send dataset and put length (or even get the real filenames)
    dir_list = os.listdir('DL22SP/train/unlabeled')
    for i in range(len(dir_list)):
        dir_list[i] = int(dir_list[i][0:-4])
    return dir_list



def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    backbone_path = "checkpoints/checkpoint_0100.pth.tar"

    model      = moco.builder.MoCo(models.__dict__['resnet50'], 128, 65536, 0.999, 0.07, True)

    checkpoint = torch.load(backbone_path, map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k]=v

    model.load_state_dict(new_state_dict)

    normalize = get_normalization("coco") # this just get normalization value ==> it is fine 

    image_size = 224
    t_norm    = FinetuneTransform(image_size=image_size, normalize=normalize, crop='none') 
    # t_orig    = FinetuneTransform(image_size=image_size, crop='none')

    train_dataset = datasets.ImageFolder('DL22SP/train', t_norm)
    #train_dataset = UnlabeledDataset(root = 'DL22SP/train/unlabeled', transform=t_norm)

    # it seems like that the transform is just random horizontal flip

    #if True:
    #    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #else:
    train_sampler = None

    #load data
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=1,
                                           pin_memory=True,
                                           #sampler=train_sampler,
                                           drop_last=True,
                                           )#collate_fn=collate_fn)

    loader    = train_loader
    image_ids = get_image_ids(train_dataset)

    norm       = "coco"
    # todo: what should be the image size here?
    
    save_box_path = 'save_box/box_10.txt'

    # get model 
    projector      = load_projection(2048, 2048, 128, num_layers=2, last_bn=False)
    grad_model     = GradCAM(model.encoder_q, projector=projector, expand_res=1)

    grad_model = grad_model.to(device)
    grad_model.eval()

    pred_masks = compute_masks(grad_model, loader)

    red_masks = clean_mask(pred_masks, single=False, min_obj_scale=0.01) 
    print('after clean mask')
    print(red_masks.size())
    pred_boxes = extract_boxes(pred_masks, image_size=image_size, margin=0)

    print(pred_boxes)

    pred_boxes = {img_id: boxes for img_id, boxes in zip(image_ids, pred_boxes)}

    box_utils.save_boxes(pred_boxes, save_box_path)


if __name__ == '__main__':
    main()









