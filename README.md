# DL-SP22-Project
Deep Learning - Spring 2022 - Final Project 

All instructions below will assume:
- The use of GCP and Slurm for their execution (as it was provided for the demo of the project).
- The project is in a folder named "DL-SP22-Project".


## Model Evaluation

```
cd DL-SP22-Project/demo
sbatch eval_mocov2_fpn.slurm
```
Note: if needed, adjust the parameters --dataset-root and --dataset-split to the corresponding root and subfolder of the dataset you want to test.
At the moment the default values are the same as provided for validation in the demo:
```
--dataset-root /labeled \
--dataset-split validation \
```

## Evaluation Results

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.236
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.249
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.019
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.094
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.282
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.362
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.434
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.435
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.035
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.217
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.506
Epoch Stats [33] - Loss Avg: 0.115782, Loss Median: 0.104066, Validation AP: 0.236441, Is Best: True
```
