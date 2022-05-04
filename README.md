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
     
       --checkpoint-path
       Use this option to indicate which e2e checkpoint you want to use for evaluation. 
       This path is expected to be under the demo folder.
       At the moment it is setup to the e2e checkpoint we provided 'best_e2e_checkpoint.pth.tar'
       Note that it should be a full network checkpoint (e2e), a backbone one would not be valid.
       
       --batch-size
       Batch size for evaluation.
       At the moment it is setup to 2.
       
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
 2. Finetuning for object detection (code under /DL-SP22-Project/demo/)



   ## Backbone Training with SSL
  
   1. Copy unlabeled dataset to the Project folder
      
      ```
      cp -rp /scratch/DL22SP/unlabeled_224.sqsh /scratch/$USER/DL-SP22-Project/
      ```  
      
   2. Unzip data
   
      Note our backbone training expects a folder of images (.PNG) and thus it is required to unzip the given sqsh dataset before starting the training 
      
      ```
      cd /scratch/$USER/DL-SP22-Project/
      unsquashfs -d train unlabeled_224.sqsh
      ```
      
      As a result of this action, there will be a directory /train/unlabeled/ under the Project folder, which contains all the images.
      
      
   3. Run SSL Backbone pretraining
      
      [For reproducibility read this step until the end before executing anything]
      
      ```
      cd /DL-SP22-Project/moco
      sbatch moco_v2.slurm
      ```
      
      Note: this slurm assumes you follow step (1) and thus your data is unzipped under the project folder. 
            If that is not the case, please replace the path ```/scratch/$USER/DL-SP22-Project/``` by the corresponding path.
      
      To reproduce our best model: run the slurm above with the parameters as already defined in file. 
      The resulting checkpoint after that run will be used as starting point for finetuning (```checkpoint_0100.pth.tar```).

      
      
      

   ## Finetuning for object detection
   
   1. Create a folder for backbone checkpoints under demo and copy your checkpoint from backbone training to the folder:
   
   
      ```
      mkdir /DL-SP22-Project/demo/checkpoints
      cp /DL-SP22-Project/moco/<ssl_checkpoint_name> /DL-SP22-Project/demo/checkpoints/<ssl_checkpoint_name>
      ```
     
      To reproduce our best model you should use ```checkpoint_0100.pth.tar``` in the operation above.


   2. Run the finetuning script
      
      [For reproducibility of our best model read this step until the end before executing anything]
      
      ```
      cd /DL-SP22-Project/demo
      sbatch demo_mocov2_fpn.slurm
      ``` 
     
     
      Notes on script arguments: 
      
       ```
       --bp 
       Use this option to provide a backbone checkpoint to start the finetuning with the results from the SSL pretraining.
       At the moment it is already setup to the desired checkpoint 'checkpoint_0100.pth.tar' setup in previous step.

       --resume
       Use this option if you want to resume the finetuning from a checkpoint from the Finetuning training (named as e2e checkpoints).
       Note that this type of checkpoints have the entire network versus the ssl/backbone checkpoints that only contain the backbone
       network weights.

       --lr
       Use this option to run the script with different LR's.
       At the moment this is setup to the value we used for traininig.
       
       --batch-size
       Use this option to run the script with different batch sizes.
       Note that this is the total batch-size (ie split among all GPUS). 
       Due to memory requirements, the maximum trainable batch size per GPU is 2. Thus, we suggest setting it up to 4 if you have 2 GPUS, and to 8 if you have 4 GPUs. 
       ``` 
       
      To reproduce our best model, you will need to execute this script three times, each one with different settings.
      To make that easier we created those scripts (see below). Each one has the corresponding set of epochs and LRs that took us to the best model (more to be explained on paper).
      It is important to know that each script builds on the result of the other (the --resume option of the r2 and r3 scripts is set
      to be the last checkpoint of the previous run). In the same way, r1 depends on the resulting checkpoint of the SSL training (that should already be in the expected folder after step 1 of this section).
      
      
      ```
      cd /DL-SP22-Project/demo
      sbatch demo_mocov2_fpn_r1.slurm
      sbatch demo_mocov2_fpn_r2.slurm
      sbatch demo_mocov2_fpn_r3.slurm
      ``` 
     
      After all this training, you should be able to evaluate your resulting checkpoint following our evaluation steps and arrive
      to the same results we provided. Congrats!
