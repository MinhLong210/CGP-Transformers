
python test_cifar10c_acc_nll.py --arch vit \
    --weight_path best_cgpt_epoch.ckpt \
    --data_root ../data --gpu_id 0 --batch_size 1000
python test_cifar10c_mce_ece.py --arch vit \
    --weight_path best_cgpt_epoch.ckpt \
    --data_root ../data --gpu_id 0 --batch_size 1000

