
import os
import sys
import json
import math
import copy
import random
from time import perf_counter
from datetime import datetime
import numpy as np
from typing import Tuple

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torchvision.models import *

codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)
from trainers import BaseTrainer, read_options
import warnings
warnings.filterwarnings("ignore")
from torch.utils._pytree import tree_map
from torch.func import functional_call, stack_module_state, vmap
    
class Trainer(BaseTrainer):
    def __init__(self, params):
        super(Trainer, self).__init__(params)
        
        for key in ['test_acc', 'train_acc', 'test_loss']:
            setattr(self, key, [])
        self.epoch_start = 0
        
        if self.continue_from is not None:
            self.log_name = os.path.join(os.path.dirname(__file__), '../logs', self.continue_from)
            results = json.load(open(self.log_name+"/results.json", 'r'))
            
            for key in ['test_acc', 'train_acc', 'test_loss']:
                setattr(self, key, results[key])
            
            for file in os.listdir(self.log_name):
                if file.endswith(".pt"):
                    self.epoch_start = int(file.split('.pt')[0][5:])
                    self.model.load_state_dict(torch.load(self.log_name+"/"+file))
            print('continue from', self.log_name, 'epoch', self.epoch_start)
    
        self.query_loss = vmap(self.compute_loss, in_dims=(0, None, None, None), randomness='different') # over b samples
        
    def compute_loss(self, params, buffers, x, y):
        pred = functional_call(self.model, (params, buffers), (x, y))
        return self.loss_flat(pred, y)
    
    
    def set_grad_to_vec(self, vec):
        """
        Helper function that sets the model's gradient to a given vector.
        """
        self.model.zero_grad()
        for param in self.model.parameters():
            size = param.data.numel()
            param.grad = vec[:size].view_as(param.data).clone()
            vec = vec[size:]
        return
    
    def get_dict_from_vec(self, vec):
        """
        Helper function that converts vec to dict.
        """
        weights = dict()
        for p_name, p in self.model.named_parameters():
            size = p.numel()
            weights[p_name] = vec[:size].view_as(p).clone()
            vec = vec[size:]
        assert vec.size(0) == 0
        return weights
    
    def unflatten_params_batched(self, param_matrix):
        """
        Args:
            param_matrix: [num_models, D] tensor of flattened params
            reference_model: nn.Module with the same structure
        Returns:
            param_tree: dict of parameter tensors with shape [num_models, *original_shape]
        """
        param_tree = {}
        num_models = param_matrix.shape[0]

        for p_name, p in self.model.named_parameters():
            shape = p.shape
            size = p.numel()
            param_tree[p_name] = param_matrix[:, :size].clone().reshape(num_models, *shape)
            param_matrix = param_matrix[:, size:]
        assert param_matrix.numel() == 0
        return param_tree
        
    def train(self):
        total_step = len(self.train_loader)
        test_interval = total_step // self.num_test_per_epoch
        
        tmp_g = torch.cat([p.data.clone().view(-1) for _, p in self.model.named_parameters()])
        self.D = len(tmp_g)
        print(f"D={self.D}")
        del tmp_g

        avg_iter_time = 0
        top_test_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.epoch_start, self.epochs):
            
            if epoch % self.eval_every_epoch == 0:
                train_accu = self.get_train_accuracy()
                print('epoch {} train accuracy {:.5f}'.format(epoch, train_accu), flush=True)
                self.train_acc.append(train_accu)
                self.model.train()

            for i, (x, y) in enumerate(self.train_loader):
                t1 = perf_counter()
            
                if i % test_interval == 0 and i > 0:
                    test_accu, L_test = self.get_test_accuracy_and_loss()
                    print('epoch', epoch, 'iter', i, 'test accuracy', test_accu, 'test loss', L_test)
                    self.test_loss.append(L_test)
                    self.test_acc.append(test_accu)
                    self.model.train()
                    
                    if test_accu > top_test_acc:
                        top_test_acc = test_accu
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= 100:
                            print('Early stopping at epoch', epoch)
                            return

                (x, y) = (x.to(self.device), y.to(self.device))
                B = x.size(0)
                
                curr_weight = torch.cat([p.data.clone().view(-1) for _, p in self.model.named_parameters()]).unsqueeze(0)
                overall_grad = torch.zeros(self.D, device=self.device)
                _, buffers = stack_module_state([self.model])
                buffers = tree_map(lambda p: p[0], buffers)
                    
                G = self.get_pub_candidates()
                
                coeffs = torch.randn(self.num_directions, self.num_candidate, device=self.device)  # [N_dir, N_cand]
                perturbations = coeffs @ G.T  # [N_dir, D]
                perturbations = perturbations * self.perturbation_scale
                
                self.model.zero_grad()
                
                microbatch_size = int(math.ceil(self.num_directions / self.num_microbatches))
                
                with torch.no_grad():
                    for j in range(self.num_microbatches):
                        lower = j * microbatch_size
                        upper = min((j + 1) * microbatch_size, B)
                        per = perturbations[lower:upper]
                        
                        params_plus  = curr_weight + per
                        params_minus = curr_weight - per
                        
                        params_plus  = self.unflatten_params_batched(params_plus)
                        params_minus = self.unflatten_params_batched(params_minus)
                    
                        l1s = self.query_loss(params_plus, buffers, x, y)
                        l2s = self.query_loss(params_minus, buffers, x, y)

                        loss_diffs = 0.5 * (l1s - l2s) / self.perturbation_scale  # [mbs, B]
                        loss_diffs = self.loss_clip(loss_diffs, self.clipping_bound).sum(dim=1)  # [mbs,]
                        noisy_l = (loss_diffs + np.sqrt(self.num_candidate+1) * self.clipping_bound * self.sigma * torch.randn_like(loss_diffs)) / B
                        overall_grad += per.T @ noisy_l   # [D]
                        
                if epoch%20==0 and i==0:
                    print('overall_grad', overall_grad/self.num_directions)
                self.set_grad_to_vec(overall_grad/self.num_directions)
                self.optimizer.step()
                
                if epoch==0 and i<20:
                    avg_iter_time += perf_counter() - t1
                    if i==19:
                        avg_iter_time /= 20
                        print('avg iter time', avg_iter_time, 's')
            
            json.dump({key: eval(f'self.{key}') for key in ['test_acc', 'train_acc', 'test_loss']}, 
                        open(self.log_name+"/results.json", 'w'), indent=4)
            torch.save(self.model.state_dict(), self.log_name+"/epoch"+str(epoch+1)+".pt")
            for file in [self.log_name+"/epoch"+str(epoch)+".pt"]:
                if os.path.exists(file):
                    os.remove(file)
    
    
    def get_pub_candidates(self):
        G = torch.zeros(self.D, self.num_candidate, device=self.device)
        for j in range(self.num_candidate):
            try:
                x_pub, y_pub = next(self.public_iterator)
            except:
                self.public_iterator = iter(self.public_loader)
                x_pub, y_pub = next(self.public_iterator)
            (x_pub, y_pub) = (x_pub.to(self.device), y_pub.to(self.device))
            pred = self.model(x_pub)
            l = self.loss(pred, y_pub)
            self.model.zero_grad()
            l.backward()
            g_public = torch.cat([p.grad.clone().view(-1) for _, p in self.model.named_parameters()])
            G[:, j] = g_public.clone() / torch.norm(g_public)
        return G
    

def main():
    options = read_options()
    t = Trainer(options)
    t.train()


if __name__ == "__main__":
    main()
    
    
