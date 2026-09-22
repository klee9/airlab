# Running
## Notice

For convenience, some checkpoints, such as the MAE-pretrained ViT-B model, are provided for manual download. Users must update the following paths accordingly. Relevant checkpoints can be acquired from the [website](https://drive.google.com/drive/folders/1zwqGvKKtjyuWdDaNSLVGJprJMPoSqAPk?usp=drive_link).
* :exclamation: **pretrain.sh, finetune.sh, scratch, eval.sh:**
Please update the following:
    * **save_checkpoint_path** to the parent directory where your experiment checkpoints are saved.  Recommend to create a ```checkpoints``` folder in the project root directory.
    * **finetune_from_pretrained_ckpt** to the location of your pre-trained checkpoint.
    * **resume_from_checkpoint** to the location of your fine-tuned checkpoint.
    * **vit_checkpoint_path** to the location of your ViT checkpoint (downloaded from the [website](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing)). Recommend to be stored in ```checkpoints/vit_mae/mae_pretrain_vit_base.pth```.
    * **libero_path** to the location of LIBERO dir.

## Seer
### Convert Data
```bash
python utils/convert_libero_per_step.py
```

### Pre-train
```bash
# Pre-train Seer on LIBERO-90 dataset
bash scripts/LIBERO_LONG/Seer/pretrain.sh
```

### Fine-tune
```bash
# Fine-tune Seer on LIBERO-10 dataset
bash scripts/LIBERO_LONG/Seer/finetune.sh
```

### Train from Scratch
```bash
# Train Seer on LIBERO-10 dataset from scratch
bash scripts/LIBERO_LONG/Seer/scratch.sh
```

### Eval
```bash
# Evaluate Seer on LIBERO-10 benchmark
bash scripts/LIBERO_LONG/Seer/eval.sh
```