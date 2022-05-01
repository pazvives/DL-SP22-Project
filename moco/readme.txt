*** Pre-training moco-v2 model by self-supervised learning ***



1. Baseline pre-training

	1) Data
	Obtained unlabeled images and record its path. In our case, the path         is /scratch/$USER/DL22SP (*Note: There should be a "train" folder under "DL22SP" and images are under "train")


	2) Running moco_v2.slurm
	Replaced the last line in moco_v2.slurm with your data path. Then submit the job by running sbatch moco_v2.slurm. To resume pre-training from a certain epoch, add an argument to the slurm file: --resume [the path to checkpoint]


	3) Results
	Checkpoints will be generated to the root every 10 epochs. 





2. CAM: Pre-training with object-aware cropping augmentation 

	1) Data
	Same as the baseline.


	2) Object-aware box reference files
	To use ready files: No action needed. We have generated box file for the unlabeled data provided in this class and it is put under save_box folder. 
	To re-generate files: Please go to folder generate_box.


	3) Running moco_v2_cam.slurm
	Replaced the last line in moco_v2_cam.slurm with your data path. Then submit the job by running sbatch moco_v2_cam.slurm. To resume pre-training with this augmentation, add an argument to the slurm file: --resume [the path to checkpoint]


	4) Results 
	Checkpoints will be generated to the root every 10 epochs. 
	


