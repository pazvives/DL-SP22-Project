# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


import moco.builder
import torchvision.models as models
import transforms as T
import utils
from engine import train_one_epoch, evaluate


from dataset import UnlabeledDataset, LabeledDataset

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):

    checkpoint = torch.load('checkpoints/checkpoint_0100.pth.tar')
    model = moco.builder.MoCo(models.__dict__['resnet50'], 128, 65536, 0.999, 0.07, True)

    #TODO: use DataParallel
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
      if 'module' in k:
        k = k.replace('module.', '')
      # else:
      #   k = k.replace('features.module.', 'module.features.')
      new_state_dict[k]=v

    model.load_state_dict(new_state_dict)
    model_feature = model.encoder_q

    # backbone = []
    # for module in model_feature.modules():
    #     backbone.append(module)

    # features = nn.Sequential(*backbone)
    # features.out_channels = 128
    model_feature = nn.Sequential(*list(model.encoder_q.children()))[:-1]
    model_feature.out_channels = 2048

    # print(features)
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(model_feature,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

    # checkpoint = torch.load('/checkpoints/checkpoint_0075.pth.tar')
    # backbone = checkpoint['state_dict']
    #
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # # let's define what are the feature maps that we will
    # # use to perform the region of interest cropping, as well as
    # # the size of the crop after rescaling.
    # # if your backbone returns a Tensor, featmap_names is expected to
    # # be [0]. More generally, the backbone should return an
    # # OrderedDict[Tensor], and in featmap_names you can choose which
    # # feature maps to use.
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=7,
    #                                                 sampling_ratio=2)
    #
    # # put the pieces together inside a FasterRCNN model
    # model = FasterRCNN(backbone,
    #                    num_classes=2,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)
    #
    #
    # # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # # # get number of input features for the classifier
    # # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # # replace the pre-trained head with a new one
    # # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #
    # return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 100
    train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 20

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)

    print("That's it!")

if __name__ == "__main__":
    main()
