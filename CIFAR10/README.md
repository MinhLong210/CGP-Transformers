## Code for Image classification on CIFAR10 in our paper
 

## Running Instructions

- To pretrain by training VIT with asymmetric kernel attention, run:
```
bash scripts/mle.sh
```

- To train CGPT/SCGPT from pretrained checkpoint, change the ckpt_dir argument in file scripts/cgpt.sh to the path of the pretrained checkpoint and run:
```
bash scripts/cgpt.sh
bash scripts/scgpt.sh
```

- To evaluate OOD performance in CIFAR10C, change the weight_path argument in file scripts/ood.sh to the path of the best validation CGPT checkpoint and run:
```
bash scripts/ood.sh
```


