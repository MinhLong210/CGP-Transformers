output_dir="checkpoints" 
run_name="CGPT_CIFAR10"
batch_size=25
batch_size_test=200
lr_ini=1e-5
lr_min=1e-5
lr_base=5e-4
warmup=0
decay=395
depth=5
num_class=10
hdim=128
num_heads=4
sample_size=1
jitter=1e-7
drop_rate=0.1
keys_len=32
kernel_type="std"
flag_cgp="True"
epochs=400

flag_mle="True"
ckpt_dir="best_mle_epoch.ckpt"

anneal_kl="0.7"
flag_adaptive_anneal="False"
anneal_kl_ini=0.0

cuda=0
seed=0

declare -a seed_list=(2)

python train_cifar.py \
    --output_dir ${output_dir} \
    --run_name ${run_name} \
    --seed ${seed} \
    --batch_size ${batch_size} \
    --batch_size_test  ${batch_size_test} \
    --lr_ini  ${lr_ini} \
    --lr_min  ${lr_min} \
    --lr_base  ${lr_base} \
    --warmup  ${warmup} \
    --decay  ${decay} \
    --depth  ${depth} \
    --num_class  ${num_class} \
    --hdim  ${hdim} \
    --num_heads  ${num_heads} \
    --sample_size  ${sample_size} \
    --jitter  ${jitter} \
    --drop_rate  ${drop_rate} \
    --keys_len  ${keys_len} \
    --kernel_type  ${kernel_type} \
    --flag_cgp  ${flag_cgp} \
    --epochs  ${epochs} \
    --flag_mle ${flag_mle} \
    --anneal_kl  ${anneal_kl} \
    --flag_adaptive_anneal  ${flag_adaptive_anneal} \
    --anneal_kl_ini  ${anneal_kl_ini} \
    --cuda  ${cuda} \
    --ckpt_dir ${ckpt_dir} 