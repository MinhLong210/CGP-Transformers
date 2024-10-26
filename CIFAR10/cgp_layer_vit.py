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

class CGP_LAYER(nn.Module):
    def __init__(self, device, num_heads, max_len, hdim, kernel_type, sample_size, jitter, keys_len, drop_rate, flag_cgp):
        super(CGP_LAYER, self).__init__()
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
        elif kernel_type == 'ard' or kernel_type=='std':
            self.log_sf_ard = nn.Parameter(0. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32))   # sf= scaling factor
            self.log_ls_ard = nn.Parameter(0. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) # ls=length scale
        
        self.sample_size = sample_size
        self.jitter = jitter
        self.device = device
        self.kernel_type = kernel_type 
        
        # self.fc_qk = nn.Linear(self.hdim, self.hdim, bias=False)
        if self.kernel_type == 'scale_dot':
            self.fc_k = nn.Linear(self.hdim, self.hdim, bias=False)
            self.fc_q = nn.Linear(self.hdim, self.hdim, bias=False)
        self.fc_v = nn.Linear(self.hdim, self.hdim, bias=False) 
        
        self.v = nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.vdim), dtype=torch.float32))
        self.s_sqrt_ltri = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len, self.keys_len), dtype=torch.float32))
        self.log_s_sqrt_diag = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len), dtype=torch.float32))

        # For CGP
        self.sigma_q = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.sigma_k = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.fc_q = nn.Linear(self.hdim, self.hdim, bias=False)
        self.fc_k = nn.Linear(self.hdim, self.hdim, bias=False)
        self.fc_x0_2 = nn.Linear(self.hdim, self.hdim,bias=False)
        
        self.W_O = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.Dropout(self.drop_rate))
        self.scale = 1 / (hdim ** 0.5)
    
    
    def get_q_k_GP(self, x):
        if self.flag_cgp:
            q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
            k = self.fc_k(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) # Asym
            x0 = self.fc_x0_2(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3)
        else: # kernel attention case
            q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
            k = q.clone()
            x0 = None
        v = self.fc_v(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        return q, k, v, x0


    def get_q_k_SDP(self, x):
        q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        k = self.fc_k(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        v = self.fc_v(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        return q, k, v
        
    def forward(self, x):
        q, k, v, x0 = self.get_q_k_GP(x)
            
        if self.flag_cgp:
            jitter = self.jitter
            if self.kernel_type == 'std':
                # Asym
                K_kk = (self.sigma_k**2) * kernel_std(k, k)
                K_qq = (self.sigma_q**2) * kernel_std(q, q)

                K_0 = kernel_std(x0, x0)
                K_qk = kernel_std(q, x0) @ torch.linalg.inv(K_0 + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ kernel_std(x0, k)
                
                f_K = (K_kk + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ v
                
                while True:
                    try:
                        chol_K_0 = torch.linalg.cholesky(K_0 + jitter* torch.eye(K_0.shape[2]).to(self.device)) 
                        break
                    except Exception:
                        jitter = jitter * 10
                # import pdb; pdb.set_trace()
                z0_samples = torch.zeros_like(x0) + (chol_K_0 @ torch.randn_like(x0).to(self.device))   

                while True:
                    try:
                        chol_K_kk = torch.linalg.cholesky(K_kk + jitter* torch.eye(K_kk.shape[2]).to(self.device)) 
                        break
                    except Exception:
                        jitter = jitter * 10
                
                # Full GP mean and covar
                mean = K_qk @ v
                E_z0z0 = K_0.unsqueeze(2)
                v0 = torch.triangular_solve(kernel_std(k, x0).permute(0,1,3,2), chol_K_kk, upper=False).solution
                E_z0z0 = E_z0z0 - v0.unsqueeze(2).permute(0,1,2,4,3) @ v0.unsqueeze(2) 
                # import pdb; pdb.set_trace()
                E_z0z0 = E_z0z0 + (kernel_std(x0, k) @ torch.linalg.inv(K_kk + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ f_K @ f_K.permute(0,1,3,2) @ \
                    torch.linalg.inv(K_kk + jitter * torch.eye(K_kk.shape[2]).to(self.device)) @ kernel_std(k, x0)).unsqueeze(2)
                

                covar = K_qq.unsqueeze(2) 
                v1 = torch.triangular_solve(kernel_std(q, x0).permute(0,1,3,2), chol_K_0, upper=False).solution
                covar -= v1.unsqueeze(2).permute(0,1,2,4,3) @ v1.unsqueeze(2) 
                covar += (kernel_std(q, x0) @ torch.linalg.inv(K_0 + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ E_z0z0.squeeze() @ \
                    torch.linalg.inv(K_0 + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ kernel_std(x0, q)).unsqueeze(2)
                covar -= mean.unsqueeze(2) @ mean.unsqueeze(2).permute(0,1,2,4,3)

                # Cholesky of covar
                while True:
                    try:
                        chol_covar = torch.linalg.cholesky(covar + jitter * torch.eye(covar.shape[3]).to(self.device))  
                        break
                    except Exception:
                        jitter = jitter * 10
                chol_covar = chol_covar.unsqueeze(2) 
                samples = mean.permute(0,1,3,2).unsqueeze(2) + (chol_covar @ \
                torch.randn(mean.shape[0], mean.shape[1], self.sample_size, mean.shape[3], mean.shape[2], 1).to(self.device)).squeeze(-1)   
                
                # mean only, no covar
                #samples = mean.permute(0,1,3,2).unsqueeze(2)

                samples = samples.permute(0,1,2,4,3) 
                samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1)
                samples = self.W_O(samples)

                ############################### log joint q ###############################
                mean_P_zq_z0 = kernel_std(q, x0) @ torch.linalg.inv(K_0 + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ z0_samples
                covar_P_zq_z0 = K_qq.unsqueeze(2) 
                vq = torch.triangular_solve(kernel_std(q, x0).permute(0,1,3,2), chol_K_0, upper=False).solution
                covar_P_zq_z0 = covar_P_zq_z0 - vq.unsqueeze(2).permute(0,1,2,4,3) @ vq.unsqueeze(2)
                
                while True:
                    try:
                        chol_covar_P_zq_z0 = torch.linalg.cholesky(covar_P_zq_z0 + jitter * torch.eye(covar_P_zq_z0.shape[3]).to(self.device))  
                        break
                    except Exception:
                        jitter = jitter * 10
                chol_covar_P_zq_z0 = chol_covar_P_zq_z0.unsqueeze(2)

                Lq = torch.triangular_solve((mean-mean_P_zq_z0), chol_covar_P_zq_z0.squeeze(), upper=False).solution 
                q_term = Lq.permute(0,1,3,2) @ Lq 

                log_joint_q = torch.mean(torch.sum(q_term, (-1,-2,-3))) + 2 * torch.abs(torch.mean(torch.sum(torch.log(torch.diagonal(chol_covar_P_zq_z0, dim1=-2, dim2=-1)), dim=-1)))

                ############################### log joint k ###############################
                mean_P_zk_z0 = kernel_std(k, x0) @ torch.linalg.inv(K_0 + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ z0_samples
                covar_P_zk_z0 = K_kk.unsqueeze(2) 
                vk = torch.triangular_solve(kernel_std(k, x0).permute(0,1,3,2), chol_K_0, upper=False).solution
                covar_P_zk_z0 = covar_P_zk_z0 - vk.unsqueeze(2).permute(0,1,2,4,3) @ vk.unsqueeze(2)
                
                while True:
                    try:
                        chol_covar_P_zk_z0 = torch.linalg.cholesky(covar_P_zk_z0 + jitter * torch.eye(covar_P_zk_z0.shape[3]).to(self.device))  
                        break
                    except Exception:
                        jitter = jitter * 10
                chol_covar_P_zk_z0 = chol_covar_P_zk_z0.unsqueeze(2)

                Lk = torch.triangular_solve((f_K-mean_P_zk_z0), chol_covar_P_zk_z0.squeeze(), upper=False).solution 
                k_term = Lk.permute(0,1,3,2) @ Lk 

                log_joint_k = torch.mean(torch.sum(k_term, (-1,-2,-3))) + 2 * torch.abs(torch.mean(torch.sum(torch.log(torch.diagonal(chol_covar_P_zk_z0, dim1=-2, dim2=-1)), dim=-1)))

                log_joint_qk = log_joint_q + log_joint_k
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
            
            elif self.kernel_type == 'std': # Asym kernel
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


