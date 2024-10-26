import numpy.random as npr
import torch
import torch.nn as nn
from util import kernel_ard, kernel_exp, kernel_std
import torch.nn.functional as F
from  torch.distributions import multivariate_normal

def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)

class SCGP_LAYER(nn.Module):
    def __init__(self, device, num_heads, max_len, hdim, kernel_type, drop_rate, keys_len, sample_size, jitter, noise, flag_cgp):
        super(SCGP_LAYER, self).__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.hdim = hdim
        self.vdim = self.hdim // self.num_heads
        self.dq = self.vdim
        self.flag_cgp = flag_cgp
        self.keys_len = keys_len
        self.drop_rate = drop_rate
        
        if kernel_type == 'exponential':
            self.log_sf_exp = nn.Parameter(-4. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32)) # sf=scaling factor
            self.log_ls_exp = nn.Parameter(4. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) # ls=length scale
        elif kernel_type == 'ard' or kernel_type=='ard_asym':
            self.log_sf_ard = nn.Parameter(0. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32))   # sf= scaling factor
            self.log_ls_ard = nn.Parameter(0. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) # ls=length scale

        self.sample_size = sample_size
        self.jitter = jitter
        self.device = device
        self.sm = nn.Sigmoid()
        self.noise = noise
        self.kernel_type = kernel_type 
        
        # self.fc_qk = nn.Linear(self.hdim, self.hdim, bias=False)
        if self.kernel_type == 'scale_dot':
            self.fc_k = nn.Linear(self.hdim, self.hdim, bias=False)
            self.fc_q = nn.Linear(self.hdim, self.hdim, bias=False)
        self.fc_v = nn.Linear(self.hdim, self.hdim, bias=False) 
        

        # For CGP
        self.sigma_q = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.sigma_k = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.fc_q = nn.Linear(self.hdim, self.hdim, bias=False)
        self.fc_k = nn.Linear(self.hdim, self.hdim, bias=False)
        self.fc_x0_2 = nn.Linear(self.hdim, self.hdim,bias=False)
        
        self.fc_induce_m = nn.Linear(self.max_len, self.keys_len)
        init_linear_layer(self.fc_induce_m)
        
        self.W_O = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.Dropout(self.drop_rate))
        self.scale = 1 / (hdim ** 0.5)

    
    
    def get_q_k_GP(self, x):
        if self.flag_cgp:
            q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
            k = self.fc_k(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) # Asym
            
            x0 = self.fc_x0_2(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3)
            
            xm = self.fc_induce_m(x.permute(0,2,1)).view(x.shape[0], self.num_heads, self.vdim, self.keys_len).permute(0,1,3,2)
            xl = xm
                
        else:
            q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
            k = q.clone()
            x0 = None
        v = self.fc_v(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        return q, k, v, x0, xm, xl

    def get_q_k_SDP(self, x): 
        q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        k = q.clone()
        v = self.fc_v(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        return q, k, v
    
    def get_q_k_SDP_asym(self, x):
        q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        k = self.fc_k(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        v = self.fc_v(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        return q, k, v
        
    def forward(self, x):
        if self.flag_cgp:
            q, k, v, x0, xm, xl = self.get_q_k_GP(x)
            z_K = v
            xm = xm.to(self.device)
            xl = xl.to(self.device)
            jitter = self.jitter
            if self.kernel_type == 'std':
                K_kk = (self.sigma_k**2) * kernel_std(k, k)
                K_qq = (self.sigma_q**2) * kernel_std(q, q)
                K_0 = kernel_std(x0, x0)

                K_mm = kernel_std(xm, xm)
                K_qm = kernel_std(q, xm)
                K_mq = K_qm.permute(0,1,3,2)
                K_0m = kernel_std(x0, xm)
                K_m0 = K_0m.permute(0,1,3,2)
                K_mm_inv = torch.linalg.inv(K_mm + 1/self.noise**2 * K_m0 @ K_0m)

                K_ll = kernel_std(xl, xl)
                K_0l = kernel_std(x0, xl)
                K_l0 = K_0l.permute(0,1,3,2)
                K_kl = kernel_std(k, xl)
                K_lk = K_kl.permute(0,1,3,2)
                K_ll_inv = torch.linalg.inv(K_ll + 1/self.noise**2 * K_lk @ K_kl)
                
                K_qk = (1/self.noise**4) * K_qm @ K_mm_inv @ K_m0 @ K_0l @ K_ll_inv @ K_lk
        
                mean = K_qk @ v

                quad_z0_zk = self.noise**2 * torch.eye(K_0.shape[2]).to(self.device) + K_0l @\
                 (K_ll_inv + 1/self.noise**4 * K_ll_inv @ K_lk @ z_K @ z_K.permute(0,1,3,2) @ K_kl @ K_ll_inv) @ K_l0
                quad_zq_zk = self.noise**2 * torch.eye(K_qq.shape[2]).to(self.device) + K_qm @\
                    (K_mm_inv + 1/self.noise**4 * K_mm_inv @ K_m0 @ quad_z0_zk @ K_0m @ K_mm_inv) @ K_mq
                covar = quad_zq_zk - mean @ mean.permute(0,1,3,2)

                while True:
                    try:
                        chol_covar = torch.linalg.cholesky(covar + jitter * torch.eye(covar.shape[3]).to(self.device))  
                        break
                    except Exception:
                        jitter = jitter * 10
                chol_covar = chol_covar.unsqueeze(2) 
                samples = mean.permute(0,1,3,2).unsqueeze(2) + (chol_covar @\
                 torch.randn(mean.shape[0], mean.shape[1], self.sample_size, mean.shape[2], mean.shape[3]).to(self.device)).permute(0,1,2,4,3)
                samples = samples.permute(0,1,2,4,3) 
                samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
                samples = self.W_O(samples)
                
                K_mm_inv2 = torch.linalg.inv(K_mm + 1/self.noise**2 * torch.eye(K_mm.shape[2]).to(x.device))
                K_ll_inv2 = torch.linalg.inv(K_ll + 1/self.noise**2 * torch.eye(K_ll.shape[2]).to(x.device))
                first_term_q = K_0m @ K_mm_inv2 @ K_mq @ K_qm @ K_mm_inv2 @ K_m0 @ K_0
                first_term_q = 1/self.noise**4 * torch.einsum('bhii->bh', first_term_q).mean()
                second_term_q = torch.einsum('bhii->bh', K_qm @ K_mm_inv2 @ K_mq).mean()

                
                first_term_k = K_0l @ K_ll_inv2 @ K_lk @ K_kl @ K_ll_inv2 @ K_l0 @ K_0
                first_term_k = 1/self.noise**4 * torch.einsum('bhii->bh', first_term_k).mean()
                second_term_k = torch.einsum('bhii->bh', K_kl @ K_ll_inv2 @ K_lk).mean()
                log_joint_qk = first_term_q + second_term_q + first_term_k + second_term_k 

                return samples, log_joint_qk

            else:
                raise ValueError("kernel_type must be std")
        
        else:
            if self.kernel_type == 'ard': # Asym kernel
                q, k, v = self.get_q_k_SDP(x)
                K_qk = kernel_ard(q, k, self.log_ls_ard, self.log_sf_ard) 
                mean = K_qk @ v
                samples = mean.unsqueeze(2) 
                samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
                samples = self.W_O(samples)
                return samples, None
            
            elif self.kernel_type == 'std':
                q, k, v = self.get_q_k_SDP(x)
                K_qk = kernel_std(q, k) 
                mean = K_qk @ v 
                samples = mean.unsqueeze(2) 
                samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
                samples = self.W_O(samples)
                return samples, None

            elif self.kernel_type == "scale_dot":
                q, k, v = self.get_q_k_SDP(x)
                attn_score = (self.scale) * (torch.einsum('abid,abdj->abij', (q, k.permute(0,1,3,2))))
                attn_prob = F.softmax(attn_score, dim=1)
                out = attn_prob @ v
                samples = out.unsqueeze(2) 
                samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
                samples = self.W_O(samples)
                return samples, None