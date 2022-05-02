# Feel free to modify this file.
# It will only be used to verify the settings are correct
# modified from https://pytorch.org/docs/stable/data.html

# Generic
import builtins
import argparse
import warnings
from collections import OrderedDict
from datetime import datetime

# Torch
import torch
import torchvision

# Project Specific
import utils
from engine import evaluate
from dataset import LabeledDataset
from demo_mocov2_fpn import get_model_with_fpn, get_transform


parser = argparse.ArgumentParser(description='PyTorch Object Detection Training')

parser.add_argument('--checkpoint-path', default='', type=str, metavar='PATH',
                    help='path to a saved state dict (default: none)')

parser.add_argument('--model-path', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                    help='batch size for data loader')

parser.add_argument('--dataset-root', default='/labeled', type=str, metavar='PATH',
                    help='root directory for evaluation dataset')

parser.add_argument('--dataset-split', default='validation', type=str, metavar='PATH',
                    help='folder under root directory where dataset is')

parser.add_argument('--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers')

def main():
    
    args            = parser.parse_args()
    model_path      = args.model_path
    checkpoint_path = args.checkpoint_path
    batch_size      = args.batch_size
    data_root       = args.dataset_root
    data_split      = args.dataset_split
    workers         = args.workers
    print("** Checkpoint Path: {}".format(checkpoint_path))
    print("** Batch_Size:{}".format(batch_size))
    print("** Dataset Root: {}".format(data_root))
    print("** Dataset Split: {}".format(data_split))
    print("** Workers: {}".format(workers))

    model = None
    if model_path:
        model = get_saved_model(model_path)
    elif checkpoint_path:
        model = load_model_from_checkpoint(checkpoint_path)
    print("Model:\n{}".format(model))

    print("Loading Data")
    valid_dataset = LabeledDataset(root  = data_root,
                                   split = data_split,
                                   transforms = get_transform(train=False))

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size  = batch_size,
                                               shuffle     = False,
                                               num_workers = workers,
                                               collate_fn  = utils.collate_fn)

    print("Starting Evaluation")
    coco_evaluator = evaluate(model, valid_loader, device=torch.device('cpu'))
    print("Finish Evaluation")

def load_model_from_checkpoint(checkpoint_path):

    device        = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint    = torch.load(checkpoint_path, map_location=device)
    backbone_path = checkpoint['backbone_path']
    model         = get_model_with_fpn(backbone_path)
    if checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                k = k.replace('module.', '')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    return model

def get_saved_model(model_path):
    model = torch.load(model_path)
    return model

def save_model_from_checkpoint(model, name='model.pth'):
    torch.save(model, name)


if __name__ == "__main__":
    main()

