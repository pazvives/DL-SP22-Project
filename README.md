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
