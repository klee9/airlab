#!/bin/bash
### NEED TO CHANGE ###
calvin_dataset_path="calvin/dataset/task_ABC_D"
save_checkpoint_path="checkpoints/"
finetune_from_pretrained_ckpt="checkpoints/pretrain_calvin_abc_d/4.pth"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
### NEED TO CHANGE ###

node=4
node_num=8
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 16 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --save_checkpoint \
    --finetune_type "calvin" \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name finetune_calvin_abc_d_ep5 \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "finetune" \
    --action_pred_steps 3 \
    --sequence_length 10 \
    --future_steps 3 \
    --window_size 13 \
    --obs_pred \
    --loss_image \
    --loss_action \
    --report_to_wandb \
    --reset_action_token \
    --reset_obs_token \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \