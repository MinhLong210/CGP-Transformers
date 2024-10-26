#! /bin/bash
output_dir="checkpoints"
run_name="CGPT_COLA"
lr_ini=1e-5
lr_min=1e-5
lr_base=5e-4
warmup=0
decay=49
flag_cgp="True"
epochs=50
anneal_kl="1.0"
flag_adaptive_anneal="False"
anneal_kl_ini=0.0
seed=4

cuda=0

python train_cola.py \
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
    --anneal_kl  ${anneal_kl} \
    --flag_adaptive_anneal  ${flag_adaptive_anneal} \
    --anneal_kl_ini  ${anneal_kl_ini} \
    --cuda  ${cuda} \


