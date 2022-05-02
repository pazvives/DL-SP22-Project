1. pip install -r requirements.txt

2. Put SSL unlabeled data under folder DL22SP (*Note: There should be a "train" folder under "DL22SP" and images are under "train")

3. Put a pre-training checkpoint under folder checkpoints (We would need some model learning before it does box inference!)

4. Run create_mask.py. A object-aware cropping box file will be generated under folder save_box