# DL-SP22-Final Project

All instructions below will assume:

- The use of GCP and Slurm for their execution (as it was provided for the demo of the project).
- The project is in a folder  ```/scratch/$USER/DL-SP22-Project/```


# Model Evaluation

To evaluate our model run the following lines.

```
cd /scratch/$USER/DL-SP22-Project/demo
sbatch eval_mocov2_fpn.slurm
```

Notes on script arguments: 
     
       --checkpoint-path <e2e_checkpoint_path> 
       Use this option to indicate which e2e checkpoint you want to use for evaluation. 
       At the moment it is setup to the e2e checkpoint we provided 'best_e2e_checkpoint.pth.tar'
       Note that it should be a full network checkpoint (e2e), a backbone one would not be valid.
       
       --batch-size 2
       Batch size for evaluation
       
       --dataset-root 
       Root of dataset to be tested ('root' as required by 'LabeledDataset' class provided)
       At the moment this is setup to '/labeled'.
       
       --dataset-split 
       Split of dataset to be tested ('root' as required by 'LabeledDataset' class provided)
       At the moment this is setup to validation.
       

Outputs of evaluation: 

As a result of the script execution, the two files below will be created, where you can check evaluations results and errors respectively.

```
eval_<pid>.out
eval_<pid>.err
```
 
## Evaluation Results

Our best model had the following accuracy when evaluating it with the validation dataset provided in the demo project.

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



# Model Training

To replicate the results above, the two steps below should be followed. 

 1. Backbone Training with SSL (code under /DL-SP22-Project/moco/)
 2. Finetuning for object deteciton (code under /DL-SP22-Project/demo/)


   ## Backbone Training with SSL
  
   1. Copy unlabeled dataset to the Project folder
      ```
      cp -rp /scratch/DL22SP/unlabeled_224.sqsh /scratch/$USER/DL-SP22-Project/
      ```  
      
   2. Unzip data
      Our backbone training expects a folder of images (.PNG) and thus it is required to unzip the given sqsh dataset before starting the training 
      ```
      cd /scratch/$USER/DL-SP22-Project/
      unsquashfs -d train unlabeled_224.sqsh
      ```
      As a result of this action, there will be a directory /train/unlabeled/ under the Project folder, which contains all the images.
      
      
   3. Run SSL Backbone pretraining
   
      ```
      cd /DL-SP22-Project/moco
      sbatch moco_v2.slurm
      ```
      
      To reproduce our best model it is needed to run the slurm above with the parameters as already defined in file. 
      The result you the script with the parameters as defined in the slurm. 
      
      Notes on script arguments: 
     
            There is no need to modify any argument in the slurm file 
      Notes:
      
      - Checkpoints are going to be saved every ten epochs. In our case, we ran it for 100 epochs and took the last epoch as the starting point for the             finetuning.
      - This script assumes your dataset root is 'scratch/$USER/DL-SP22-Project' (as first step indicates). If that is not the case, you should replace that path inside the slurm script.

   ## Finetuning for object detection

   1. Run the finetuning script

      ```
      cd /DL-SP22-Project/demo
      sbatch demo_mocov2_fpn.slurm
      ``` 
     
     
   Notes on script arguments: 
     
       --bp <ssl_checkpoint_path> 
       Use this option if you want to start the finetuning from a checkpoint from the SSL training.
       Note that this type of checkpoints only have the backbone network.

       --resume <e2e_checkpoint_path>
       Use this option if you want to start the finetuning from a checkpoint from the Finetuning training.
       Note that this type of checkpoints have the entire network.

       --lr
       Use this option to run the script with different LR's.
     
     
  Notes on replicating our results:
     Our best model was the result of different runs:
     - [TODO PAZ: Complete with the details of our final run]
 
