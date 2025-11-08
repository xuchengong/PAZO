""" code adopted from https://github.com/dayu11/Gradient-Embedding-Perturbation/blob/master/models/basis_matching.py"""

import torch
import torch.nn as nn
import numpy as np
import math

#package for computing individual gradients
# from backpack import backpack, extend
# from backpack.extensions import BatchGrad

def flatten_tensor(tensor_list):

    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param

@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col

def clip_column(tsr, clip=1.0, inplace=True):
    if(inplace):
        inplace_clipping(tsr, torch.tensor(clip).cuda())
    else:
        norms = torch.norm(tsr, dim=1)

        scale = torch.clamp(clip/norms, max=1.0)
        return tsr * scale.view(-1, 1) 

@torch.jit.script
def inplace_clipping(matrix, clip):
    n, m = matrix.shape
    for i in range(n):
        # Normalize the i'th row
        col = matrix[i:i+1, :]
        col_norm = torch.sqrt(torch.sum(col ** 2))     
        if(col_norm > clip):
            col /= (col_norm/clip)

def check_approx_error(L, target):
    encode = torch.matmul(target, L) # n x k
    decode = torch.matmul(encode, L.T)
    error = torch.sum(torch.square(target - decode))
    target = torch.sum(torch.square(target))
    if(target.item()==0):
        return -1
    return error.item()/target.item()

def get_bases(pub_grad, num_bases, power_iter=1, logging=False):
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]
  
    num_bases = min(num_bases, num_p)
    L = torch.normal(0, 1.0, size=(pub_grad.shape[1], num_bases), device=pub_grad.device)
    for i in range(power_iter):
        R = torch.matmul(pub_grad, L) # n x k
        L = torch.matmul(pub_grad.T, R) # p x k
        orthogonalize(L)
    error_rate = check_approx_error(L, pub_grad)
    return L, num_bases, error_rate



class GEP(nn.Module):

    def __init__(self, num_bases, clip0=1, clip1=1, power_iter=1):
        super(GEP, self).__init__()

        self.num_bases = num_bases
        self.clip0 = clip0
        self.clip1 = clip1
        self.power_iter = power_iter
        self.approx_error = {}

    def get_approx_grad(self, embedding):
        bases_list, num_bases_list, num_param_list = self.selected_bases_list, self.num_bases_list, self.num_param_list
        grad_list = []
        offset = 0
        if(len(embedding.shape)>1):
            bs = embedding.shape[0]
        else:
            bs = 1
        embedding = embedding.view(bs, -1)

        for i, bases in enumerate(bases_list):
            num_bases = num_bases_list[i]

            grad = torch.matmul(embedding[:, offset:offset+num_bases].view(bs, -1), bases.T)
            if(bs>1):
                grad_list.append(grad.view(bs, -1))
            else:
                grad_list.append(grad.view(-1))
            offset += num_bases           
        if(bs>1):
            return torch.cat(grad_list, dim=1)
        else:
            return torch.cat(grad_list)

    def get_anchor_space_func(self, pub_grads, logging=False):

        with torch.no_grad():
            
            num_param_list = self.num_param_list

            selected_bases_list = []
            num_bases_list = []
            pub_errs = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            num_bases_list = self.num_bases * (sqrt_num_param_list/np.sum(sqrt_num_param_list))
            num_bases_list = num_bases_list.astype(np.int32)
            
            offset = 0

            for i, num_param in enumerate(num_param_list):
                pub_grad = pub_grads[:, offset:offset+num_param]
                offset += num_param
                
                num_bases = num_bases_list[i]
                
                selected_bases, num_bases, pub_error = get_bases(pub_grad, num_bases, self.power_iter, logging)
                pub_errs.append(pub_error)

                num_bases_list[i] = num_bases
                selected_bases_list.append(selected_bases)

            self.selected_bases_list = selected_bases_list
            self.num_bases_list = num_bases_list
            self.approx_errors = pub_errs
        del pub_grads

    def get_anchor_space_persample(self, net, loss_func, data, device, logging=False):
        # compute public grads
        x_public, y_public = [d.to(device) for d in data]
        predicted = net(x_public)
        l = loss_func(predicted, y_public)  # (microbatch_size, )
        net.zero_grad()
        pub_grads = []
        for j in l:
            net.zero_grad()
            j.backward(retain_graph=True)
            grad = torch.cat([p.grad.clone().view(-1) for _, p in net.named_parameters()])
            pub_grads.append(grad)
        pub_grads = torch.stack(pub_grads, dim=0) # (B, D)
          
        with torch.no_grad():            
            num_param_list = self.num_param_list

            selected_bases_list = []
            num_bases_list = []
            pub_errs = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            num_bases_list = self.num_bases * (sqrt_num_param_list/np.sum(sqrt_num_param_list))
            num_bases_list = num_bases_list.astype(np.int32)
            
            offset = 0

            for i, num_param in enumerate(num_param_list):
                pub_grad = pub_grads[:, offset:offset+num_param]
                offset += num_param
                
                num_bases = num_bases_list[i]
                
                selected_bases, num_bases, pub_error = get_bases(pub_grad, num_bases, self.power_iter, logging)
                pub_errs.append(pub_error)

                num_bases_list[i] = num_bases
                selected_bases_list.append(selected_bases)

            self.selected_bases_list = selected_bases_list
            self.num_bases_list = num_bases_list
            self.approx_errors = pub_errs
        del pub_grads
    
    def get_anchor_space(self, net, loss_func, data, device, logging=False):
        # compute public grads
        x_public, y_public = [d.to(device) for d in data]
        predicted = net(x_public)
        l = loss_func(predicted, y_public)  # (microbatch_size, )
        net.zero_grad()
        pub_grads = torch.autograd.grad(l, net.parameters(), torch.eye(len(l)).to(device), 
                                    retain_graph=False, is_grads_batched=True) # num_layer of (microbatch_size, num_params, ...)
        pub_grads = [layer.view(layer.size(0), -1) for layer in pub_grads]
        pub_grads = torch.cat(pub_grads, dim=1) # (microbatch_size, D)
          
        with torch.no_grad():            
            num_param_list = self.num_param_list

            selected_bases_list = []
            num_bases_list = []
            pub_errs = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            num_bases_list = self.num_bases * (sqrt_num_param_list/np.sum(sqrt_num_param_list))
            num_bases_list = num_bases_list.astype(np.int32)
            
            offset = 0

            for i, num_param in enumerate(num_param_list):
                pub_grad = pub_grads[:, offset:offset+num_param]
                offset += num_param
                
                num_bases = num_bases_list[i]
                
                selected_bases, num_bases, pub_error = get_bases(pub_grad, num_bases, self.power_iter, logging)
                pub_errs.append(pub_error)

                num_bases_list[i] = num_bases
                selected_bases_list.append(selected_bases)

            self.selected_bases_list = selected_bases_list
            self.num_bases_list = num_bases_list
            self.approx_errors = pub_errs
        del pub_grads

    def get_anchor_space_roberta(self, net, loss_func, data, device, logging=False):
        # compute public grads
        [x, mask, y, pos] = [d.to(device) for d in data]
                
        predicted = net(input_ids=x, attention_mask=mask, mask_pos=pos)
        l = loss_func(predicted, y)  # (microbatch_size, )
        net.zero_grad()
        pub_grads = torch.autograd.grad(l, net.parameters(), torch.eye(len(l)).to(device), 
                                    retain_graph=False, is_grads_batched=True) # num_layer of (microbatch_size, num_params, ...)
        pub_grads = [layer.view(layer.size(0), -1) for layer in pub_grads]
        pub_grads = torch.cat(pub_grads, dim=1) # (microbatch_size, D)
          
        with torch.no_grad():            
            num_param_list = self.num_param_list

            selected_bases_list = []
            num_bases_list = []
            pub_errs = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            num_bases_list = self.num_bases * (sqrt_num_param_list/np.sum(sqrt_num_param_list))
            num_bases_list = num_bases_list.astype(np.int32)
            
            offset = 0

            for i, num_param in enumerate(num_param_list):
                pub_grad = pub_grads[:, offset:offset+num_param]
                offset += num_param
                
                num_bases = num_bases_list[i]
                
                selected_bases, num_bases, pub_error = get_bases(pub_grad, num_bases, self.power_iter, logging)
                pub_errs.append(pub_error)

                num_bases_list[i] = num_bases
                selected_bases_list.append(selected_bases)

            self.selected_bases_list = selected_bases_list
            self.num_bases_list = num_bases_list
            self.approx_errors = pub_errs
        del pub_grads


    def forward(self, target_grad, logging=False):
        with torch.no_grad():
            num_param_list = self.num_param_list
            embedding_list = []

            offset = 0

            for i, num_param in enumerate(num_param_list): 
                grad = target_grad[:, offset:offset+num_param]
                selected_bases = self.selected_bases_list[i]
                embedding = torch.matmul(grad, selected_bases)
                if(logging):
                    cur_approx = torch.matmul(torch.mean(embedding, dim=0).view(1, -1), selected_bases.T).view(-1)
                    cur_target = torch.mean(grad, dim=0)
                    cur_error = torch.sum(torch.square(cur_approx-cur_target))/torch.sum(torch.square(cur_target))
                    print('group %d, param: %d, num of bases: %d, group wise approx error: %.2f%%'%(i, num_param, self.num_bases_list[i], 100*cur_error.item()))
                    if(i in self.approx_error):
                        self.approx_error[i].append(cur_error.item())
                    else:
                        self.approx_error[i] = []
                        self.approx_error[i].append(cur_error.item())

                embedding_list.append(embedding)
                offset += num_param

            concatnated_embedding = torch.cat(embedding_list, dim=1)
            clipped_embedding = clip_column(concatnated_embedding, clip=self.clip0, inplace=False) 
            
            sum_clipped_embedding = torch.sum(clipped_embedding, dim=0) 

            no_reduction_approx = self.get_approx_grad(concatnated_embedding)
            residual_gradients = target_grad - no_reduction_approx
            
            if(logging):
                print('norm of embedding', torch.norm(concatnated_embedding, dim=1).mean().item(),
                  'norm of residual', torch.norm(residual_gradients, dim=1).mean().item())
            
            clip_column(residual_gradients, clip=self.clip1) #inplace clipping to save memory
            clipped_residual_gradients = residual_gradients

            sum_clipped_residual_gradients = torch.sum(clipped_residual_gradients, dim=0)
            return sum_clipped_embedding.view(-1), sum_clipped_residual_gradients.view(-1)