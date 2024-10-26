#! /bin/bash
output_dir="logs_scgp"
run_name="SCGP"
lr_ini=1e-5
lr_min=1e-5
lr_base=5e-4
warmup=0
decay=49
flag_cgp="True"
epochs=50
use_wandb="False"
kernel_type="std"
keys_len=5
noise=0.5

anneal_kl="1.0"
flag_adaptive_anneal="False"
anneal_kl_ini=0.0
seed=0

cuda=0

python train_scgp.py \
    --output_dir ${output_dir} \
    --run_name ${run_name} \
    --seed ${seed} \
    --lr_ini  ${lr_ini} \
    --lr_min  ${lr_min} \
    --lr_base  ${lr_base} \
    --warmup  ${warmup} \
    --decay  ${decay} \
    --flag_cgp  ${flag_cgp} \
    --epochs  ${epochs} \
    --noise ${noise} \
    --use_wandb  ${use_wandb} \
    --anneal_kl  ${anneal_kl} \
    --kernel_type ${kernel_type} \
    --flag_adaptive_anneal  ${flag_adaptive_anneal} \
    --anneal_kl_ini  ${anneal_kl_ini} \
    --cuda  ${cuda} \

done
