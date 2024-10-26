# CGPT_code_submission
Code submission for "Revisiting Kernel Attention via Correlated Gaussian Process Representation"

## Requirements
We follow the previously published implementation of the paper [Calibrating Transformers via Sparse Gaussian Processes](https://arxiv.org/abs/2303.02444). The setup for Anaconda environment is as follows:
1. Create new python environment with version python=3.8 and activate it
2. Run `pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`
3. Run `pip install -r requirements.txt` in this directory to install the remaining packages.
4. Download the CIFAR10 dataset [here](https://zenodo.org/records/2535967) to [data](data).
5. Download the COLA dataset [here](https://nyu-mll.github.io/CoLA/) to [data](data).

## Image classification on CIFAR10
The code for this task is in [CIFAR10](CIFAR10).
Please refer to this [README.md](CIAR10/README.md) file for more details.

## Linguistic acceptability on COLA
The code for this task is in [COLA](COLA).
Please refer to this [README.md](COLA/README.md) file for more details.
