output_dir="output"
run_name="SCGPT"
batch_size=100
batch_size_test=100
lr_ini=1e-5
lr_min=1e-6
lr_base=1e-5
warmup=0
decay=495
depth=5
num_class=10
hdim=128
num_heads=4
sample_size=1
jitter=1e-7
noise=0.1
drop_rate=0.1
keys_len=8
kernel_type="std"
flag_cgp="True"
epochs=500
use_wandb="False"
seed=0
ckpt_dir="pretrained/08022024.MLE_ASYM_STD.seed_0/best_epoch.ckpt"

flag_mle="True"

anneal_kl="1.0"
flag_adaptive_anneal="True"
anneal_kl_ini=0.0
monte_carlo_samples=1

cuda=0

python3 train_scgpt.py \
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
    --use_wandb  ${use_wandb} \
    --flag_mle  ${flag_mle} \
    --anneal_kl  ${anneal_kl} \
    --flag_adaptive_anneal  ${flag_adaptive_anneal} \
    --anneal_kl_ini  ${anneal_kl_ini} \
    --cuda  ${cuda} \
    --noise ${noise} \
    --ckpt_dir ${ckpt_dir}
