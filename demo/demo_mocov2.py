# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

# Generic 
import builtins
import argparse
import os
import shutil
import warnings
from collections import OrderedDict
from datetime import datetime

# Torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torch.distributed as dist
import torchvision.models as models
import torch.multiprocessing as mp
from detectron2.layers import FrozenBatchNorm2d
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Project Specific
import moco.builder
import transforms as T
import utils
from engine import train_one_epoch, evaluate
from dataset import UnlabeledDataset, LabeledDataset


# TODO: remove commented arguments once done
parser = argparse.ArgumentParser(description='PyTorch Object Detection Training')
# parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='FasterRCNN', 
                    help='model architecture name')
parser.add_argument('--bp', '--backbone-path',  metavar='DIR',  
                    help='path to backbone checkpoint', type=str, required=True)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# # options for moco v2
# parser.add_argument('--mlp', action='store_true',
#                     help='use mlp head')
# parser.add_argument('--aug-plus', action='store_true',
#                     help='use moco v2 data augmentation')
# parser.add_argument('--cos', action='store_true',
#                     help='use cosine lr schedule')


def main():

    args = parser.parse_args()

    print("** Batch_Size:{}".format(args.batch_size))
    print("** LR:{}".format(args.lr))
    print("** Momentum:{}".format(args.momentum))
    print("** Weight Decay:{}".format(args.weight_decay))
    print("** Backbone Used:{}".format(args.bp))

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    # TODO: i would like to try this in a second try, in the meanwhile just logging
    #print("ENV - WORLD_SIZE:{}".format(int(os.environ["WORLD_SIZE"])))
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node:{}".format(ngpus_per_node))
    if args.multiprocessing_distributed:
        # Update of total processes based on total # of gpus (ngpus processes per node) 
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu
    print("=>args.gpu {}".format(args.gpu))
    
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # rank needs to be global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend     = args.dist_backend, 
                                init_method = args.dist_url,
                                world_size  = args.world_size, 
                                rank        = args.rank)

    # create model
    print("=> creating model: {}'".format(args.arch))
    num_classes = 101
    model       = get_model(num_classes, args.bp)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per DistributedDataParallel, 
            # we need to divide the batch size based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # TODO[REQUIRED] define loss criterion and send to correct gpu device
    # Eg from MOCO: criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD( params, 
                                 args.lr, 
                                 momentum     = args.momentum, 
                                 weight_decay = args.weight_decay) 
                                 # ORIGINAL:  lr=0.001, momentum=0.9, momentum = ,weight_decay=0.0005

    # TODO[not now]: add here if we wanna resumefrom checkpoint (the whole model)

    cudnn.benchmark
    # Data loading
    # TODO[REQUIRED]: define transformations for training
    train_dataset = LabeledDataset(root       = '/labeled', 
                                   split      = "training", 
                                   transforms = get_transform(train = True))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader( train_dataset, 
                                                batch_size  = args.batch_size, 
                                                shuffle     = (train_sampler is None),
                                                num_workers = args.workers, 
                                                pin_memory  = True, 
                                                sampler     = train_sampler, 
                                                drop_last   = True, 
                                                collate_fn  = utils.collate_fn)

    # TODO[REQUIRED]: review validation, see inside loop below
    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)


    training_losses  = []
    validation_stats = []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Update lr
        adjust_learning_rate(optimizer, epoch, args)

        # Train one epoch
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Calling train_one_epoch with device: {}".format(device))

        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100) 
        
        # Saving total loss
        training_losses.append(metric_logger.meters['loss'])

        
        # Saving validation accuracy
        coco_evaluator = evaluate(model, valid_loader, device=torch.device('cpu')) 
        coco_eval_bbox = coco_evaluator.coco_eval['bbox'] #pycocotools.cocoeval.COCOeval object returned
        validation_stats.append(coco_eval_bbox.stats[0]) #(AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]


        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch':  args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch, 
                                                                          datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")))

        print("Epoch [{}] Complete!".format(epoch))


    print("Training Complete!")
    print("Training Losses: \n".format(training_losses))
    print("Validation Stats: \n".format(validation_stats))

# TODO: play with different lrs?
# If we use the stepswise only, we can leave something as below, where step_size
# tells us after how many epochs is the lr update and gamma tells us the proportion of update.
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
def adjust_learning_rate(optimizer, epoch, args):
    """ Decay the learning rate based on schedule """

    lr = args.lr
    #if args.cos:  # cosine lr schedule
    #    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    #else:  # stepwise lr schedule
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_transform(train):
    
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_model(num_classes, backbone_path): 
    """ Same FasterRCNN model but adding FPN. 
        Keeping both until we have a better understanding of what we want as final."""


    print("Loading backbone checkpoint at:{}".format(backbone_path))
    checkpoint = torch.load(backbone_path)
    model      = moco.builder.MoCo(models.__dict__['resnet50'], 128, 65536, 0.999, 0.07, True)

    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
      if 'module' in k:
        k = k.replace('module.', '')
      new_state_dict[k]=v

    model.load_state_dict(new_state_dict)
    model_feature = nn.Sequential(*list(model.encoder_q.children()))[:-1]
    model_feature.out_channels = 2048

    # FPN 
    pretrained_backbone       = model_feature
    trainable_backbone_layers = 5
    trainable_backbone_layers = _validate_trainable_layers(
                                                            pretrained_backbone, 
                                                            trainable_backbone_layers, 
                                                            5,
                                                            3)

    #Freeze Norm Layers - Not using it for now
    # for module in (pretrained_backbone.modules()):
    #     if isinstance(child, nn.BatchNorm2d):
    #         module.eval()
    #         for param in child.parameters():
    #             param.requires_grad = False
        

    frozen_pretrained_backbone = FrozenBatchNorm2d.convert_frozen_batchnorm(pretrained_backbone)
    backbone = _resnet_fpn_extractor(frozen_pretrained_backbone, trainable_backbone_layers)
    

    #TODO: delete anchor and roi pooler as they are default values
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

    return model


def get_model_backup(num_classes, backbone_path): 

    print("Loading backbone checkpoint at:{}".format(backbone_path))
    checkpoint = torch.load(backbone_path)
    model      = moco.builder.MoCo(models.__dict__['resnet50'], 128, 65536, 0.999, 0.07, True)

    #TODO: alternative, use DataParallel
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
      if 'module' in k:
        k = k.replace('module.', '')
      new_state_dict[k]=v

    model.load_state_dict(new_state_dict)
    model_feature = model.encoder_q
    model_feature = nn.Sequential(*list(model.encoder_q.children()))[:-1]
    model_feature.out_channels = 2048

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

    return model




if __name__ == "__main__":
    main()
