# DL-SP22-Final Project

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

Outputs: as a result of the run, the two files below will be created, where you can check evaluations results and errors respectively.
```
eval_<pid>.out
eval_<pid>.err
```
 

## Evaluation Results

Our current model had the following accuracy when evaluating it with the validation dataset provided in the demo project.

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.237
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.249
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.014
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.094
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.283
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.360
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.427
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.200
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.500
Epoch Stats [36] - Loss Avg: 0.112939, Loss Median: 0.091720, Validation AP: 0.236755, Is Best: True
```
