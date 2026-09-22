save_checkpoint_path="/home/keon/vla_ft/Seer/checkpoints"
root_dir="/home/keon/vla_ft/Seer/test_data"
real_dataset_names="panda_pick_place_fixed"
finetune_from_pretrained_ckpt="/home/keon/vla_ft/Seer/seer.pth"

# Downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
vit_checkpoint_path="/home/keon/vla_ft/Seer/mae_pretrain_vit_base.pth"

### EXAMPLE ###
# - root_dir
#   - real_dataset_names
#       - 0000
#           - 000000
#           - ......
#           - xxxxxx
#       - ....
#       - 00xx 
### EXAMPLE ###

node=1
node_num=4
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Change accordingly
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=29511 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset "" \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 40 \
    --seed 42 \
    --batch_size 4 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --save_checkpoint \
    --finetune_type real \
    --root_dir ${root_dir} \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name sn_ft \
    --save_checkpoint_path ${save_checkpoint_path} \
    --except_lang \
    --transformer_layers 24 \
    --phase "finetune" \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --obs_pred \
    --loss_action \
    --loss_image \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint 15 \
    --warmup_epochs 5 \
    --real_dataset_names ${real_dataset_names} \
    --use_aug_data \
    --reset_action_token \
    --reset_obs_token \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
