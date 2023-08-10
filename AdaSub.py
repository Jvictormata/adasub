import torch
from torch.optim import Optimizer
import numpy as np


class SubHes(Optimizer):
    def __init__(self, parameters, lr=1e-3, n_directions = 2, device = "cpu"):
        defaults = {"lr": lr, "device":device}
        super().__init__(parameters, defaults)
        self.state['step'] = 0
        self.n_directions = n_directions
        self.ro = 0.02
        self.device = device
        for p in self.get_params():
            p.update_value = 0
      
    
    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)
 
        
    @torch.no_grad()
    def update_subspace(self,old_subSpace,new_gradient):
        if self.state['step'] < self.n_directions:
            new_subSpace = torch.cat([old_subSpace,new_gradient],1)
        else:
            new_subSpace = torch.cat([old_subSpace[:,1:],new_gradient],1)
        return new_subSpace
        
       
    
    def Hessian_v(self,grad,vec,p):                
        Hv = torch.autograd.grad(
        grad,
        p,
        grad_outputs=vec,
        only_inputs=True,
        retain_graph=True)
        
        Hv_flaten = []
        for i in range(len(Hv)):
            Hv_flaten.append(Hv[i].reshape(-1))
        return torch.cat(Hv_flaten,0).view(-1,1)
    
    
    
    def Hessian_M(self,grad,matrix,p):
        H_M = []
        for i in range(matrix.shape[1]):
            H_M.append(self.Hessian_v(grad,matrix[:,i].view(-1,1),p))
        return torch.cat(H_M,1)  
        
        
    @torch.no_grad()    
    def correction_Hessian(self,H):
        eig, U = torch.linalg.eigh(H)
        alfa = 0
        if eig.min() < self.ro:
            alfa = self.ro - eig.min()            
        return U, eig, alfa           
        
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
                      
                

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.state['step'] == 0:
                    d_p = p.grad
                    with torch.no_grad():
                        p.add_(d_p, alpha=-group['lr'])
                    p.subSpace = p.grad.data.view(-1,1)
                else:
                    flat_grad = p.grad.view(-1,1)
                    p.subSpace = self.update_subspace(p.subSpace,flat_grad.data)
                    Q, _ = torch.linalg.qr(p.subSpace.data)
                    HQ = self.Hessian_M(flat_grad,Q,p)
                    U, eig, alfa = self.correction_Hessian(Q.T@HQ)
                    y = U@torch.diag(1/(eig+alfa))@U.T@(Q.T@flat_grad)
                    d_p = Q@y
                    p.update_value = d_p.view_as(p.data)
                    
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                with torch.no_grad():
                    p.add_(p.update_value, alpha=-group['lr'])
                            
        
        self.state['step'] += 1        
        return 




