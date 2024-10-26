import torch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def str2bool (val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))

def lr_scheduler(epoch, warmup_epochs, decay_epochs, initial_lr, base_lr, min_lr):
    if epoch <= warmup_epochs:
        pct = epoch / max(warmup_epochs,1)
        if warmup_epochs == 0 and epoch == 0:
            pct = 1
        return ((base_lr - initial_lr) * pct) + initial_lr
    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr
    return min_lr

def anneal_scheduler(cur_epoch, num_epochs, min_anneal, max_anneal):
    pct = cur_epoch / max(num_epochs,1)
    return ((max_anneal - min_anneal) * pct) + min_anneal


def kernel_ard(X1, X2, log_ls, log_sf):
    X1 = X1 * torch.exp(-log_ls).unsqueeze(1)
    X2 = X2 * torch.exp(-log_ls).unsqueeze(1)
    X1 = X1.permute(0,1,3,2).unsqueeze(4) 
    X2 = X2.unsqueeze(3) 
    return  torch.exp(log_sf).unsqueeze(1) * \
        torch.exp(-0.5* torch.sum((X1-X2.permute(0,1,4,3,2)).pow(2), 2)) 


def kernel_exp(X1, X2, log_ls, log_sf):
    X1 = X1 * torch.exp(-log_ls).unsqueeze(1) 
    X2 = X2 * torch.exp(-log_ls).unsqueeze(1)
    return torch.exp(log_sf).unsqueeze(1)* torch.exp(X1 @ X2.permute(0,1,3,2))

# standardized kernel
def kernel_std(X1, X2):
    X1 = X1.permute(0,1,3,2).unsqueeze(4) 
    X2 = X2.unsqueeze(3) 
    return torch.exp(-0.5* torch.sum((X1-X2.permute(0,1,4,3,2)).pow(2), 2))

def kernel_exp_std(X1, X2):
    return torch.exp(X1 @ X2.permute(0,1,3,2)) 
