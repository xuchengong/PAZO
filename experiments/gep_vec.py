
import os
import sys
import json
import math
import copy
import random
import importlib
import numpy as np
from datetime import datetime
from time import perf_counter
from typing import Tuple

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as data
from torchvision.models import *

codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)
from trainers import BaseTrainer, read_options, GEP
from privacy_analysis.rdp_accountant import compute_rdp, get_privacy_spent
from functorch import make_functional, vmap, grad_and_value
import warnings
warnings.filterwarnings("ignore")

def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=True):
    while True:
        orders = ([1.1, 1.2, 1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75] \
              + list(np.arange(5, 64, 0.5)) + [128, 256, 512])
        steps = T
        if(rgp):
            rdp = compute_rdp(q, cur_sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break    
    return cur_sigma, previous_eps


## interval: init search inerval
## rgp: use residual gradient perturbation or not
def get_sigma(q, T, eps, delta, init_sigma=18, interval=1., rgp=True):
    cur_sigma = init_sigma
    
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps

    
class Trainer(BaseTrainer):
    def __init__(self, params):
        super(Trainer, self).__init__(params)
        
        for key in ['test_acc', 'train_acc', 'test_loss', 'test_loss']:
            setattr(self, key, [])
        self.epoch_start = 0
        
        if self.continue_from is not None:
            self.log_name = os.path.join(os.path.dirname(__file__), '../logs', self.continue_from)
            results = json.load(open(self.log_name+"/results.json", 'r'))
            for key in ['test_acc', 'train_acc']:
                setattr(self, key, results[key])
                
            for file in os.listdir(self.log_name):
                if file.endswith(".pt"):
                    self.epoch_start = int(file.split('.pt')[0][5:])
                    self.model.load_state_dict(torch.load(self.log_name+"/"+file))
            print('continue from', self.log_name, 'epoch', self.epoch_start)
        
        delta = 1.0 / self.training_sample_size
        print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(self.target_eps, delta))
        sampling_prob = self.private_bs / self.training_sample_size
        steps = int(self.epochs/sampling_prob)
        sigma, eps = get_sigma(sampling_prob, steps, self.target_eps, delta, rgp=1)
        self.noise_multiplier0 = self.noise_multiplier1 = sigma
        print('noise scale for gradient embedding: ', self.noise_multiplier0, 
              'noise scale for residual gradient: ', self.noise_multiplier1, 
              'privacy guarantee: ', eps)

        self.gep = GEP(self.num_bases, self.clip0, self.clip1, self.power_iter).cuda()

        func_model, weights = make_functional(self.model)
        
        def compute_loss(weights, x, y):
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            predicted = func_model(weights, x)
            loss = self.loss(predicted, y)
            return loss

        self.compute_grad_and_loss = grad_and_value(compute_loss)
        self.per_sample_grad_and_loss = vmap(self.compute_grad_and_loss, in_dims=(None, 0, 0), 
                                             randomness='different')
        
    def group_params(self, groups):
        assert groups >= 1
        p_per_group = self.D//groups
        num_param_list = [p_per_group] * (groups-1)
        num_param_list = num_param_list + [self.D-sum(num_param_list)]
        return num_param_list
    
    def train(self):
        avg_iter_time = 0
        total_step = len(self.train_loader)
        test_interval = total_step // self.num_test_per_epoch
        tmp_g = torch.cat([p.data.clone().view(-1) for _, p in self.model.named_parameters()])
        self.D = len(tmp_g)
        print('D=', self.D)
        del tmp_g
        
        print('\n==> Dividing parameters in to %d groups'%self.num_groups)
        self.gep.num_param_list = self.group_params(self.num_groups)
        
        for epoch in range(self.epoch_start, self.epochs):
            
            if epoch % self.eval_every_epoch == 0 and epoch > 0:
                L_test = self.get_test_loss()
                train_accu = self.get_train_accuracy()
                print('epoch {} test loss {:.5f} train accuracy {:.5f}'.format(epoch, L_test, train_accu), flush=True)
                self.test_loss.append(L_test)
                self.train_acc.append(train_accu)
                self.model.train()

            # itime = perf_counter()
            for i, (xs, s) in enumerate(self.train_loader):
                
                mtime = perf_counter()
                embedding_sum, residual_sum = None, None
                _, weights = make_functional(self.model)
                
                if i % test_interval == 0 and i > 0:
                    test_accu = self.get_test_accuracy()
                    print('epoch', epoch, 'iter', i, 'test accuracy', test_accu)
                    self.test_acc.append(test_accu)
                    self.model.train()

                xs = xs.to(self.device)
                s = s.to(self.device)
                
                B = xs.size(0)
                microbatch_size = int(math.ceil(B / self.num_microbatches))
                # print('batch size', B, 'microbatch size', microbatch_size)

                # compute anchor subspace
                self.optimizer.zero_grad()
                logging = i % 20 == 0
                try:
                    x_public, y_public = next(self.public_iterator)
                except:
                    self.public_iterator = iter(self.public_loader)
                    x_public, y_public = next(self.public_iterator)
                x_public = x_public.to(self.device)
                y_public = y_public.to(self.device)
                grads, loss = self.per_sample_grad_and_loss(weights, x_public, y_public)
                with torch.no_grad():
                    grad_tensor = []
                    for grad in grads:
                        grad_tensor.append(grad.view(grad.size(0), -1).detach())
                    del grads
                    grad_tensor = torch.cat(grad_tensor, 1)
                self.gep.get_anchor_space_func(grad_tensor, logging=False)
                
                for j in range(self.num_microbatches):
                    lower = j * microbatch_size
                    upper = min((j + 1) * microbatch_size, B)
                    
                    grads, loss = self.per_sample_grad_and_loss(weights, xs[lower:upper], s[lower:upper])
                    
                    with torch.no_grad():
                        grad_tensor = []
                        for grad in grads:
                            grad_tensor.append(grad.view(grad.size(0), -1).detach())
                        del grads
                        grad_tensor = torch.cat(grad_tensor, 1)
                    
                    # compute gradient embeddings and residual gradients
                    embedding, residual = self.gep(grad_tensor, logging = False)
                    if embedding_sum is None:
                        embedding_sum, residual_sum = embedding, residual
                    else:
                        embedding_sum += embedding
                        residual_sum += residual
                        
                embedding_sum += torch.normal(0, self.noise_multiplier0 * self.clip0,
                                            size=embedding_sum.shape, device=self.device)
                residual_sum += torch.normal(0, self.noise_multiplier1 * self.clip1, 
                                            size=residual_sum.shape, device=self.device)
                
                saved_var = (self.gep.get_approx_grad(embedding_sum) + residual_sum) / B
                
                for p_name, p in self.model.named_parameters():
                    num_params = p.numel()
                    p.grad = saved_var[:num_params].view_as(p).clone()
                    saved_var = saved_var[num_params:]
                assert saved_var.size(0) == 0

                self.optimizer.step()  # apply p.grad
                
                # print('iter', i, 'iter time', perf_counter() - itime, 'model time', perf_counter() - mtime)
                if epoch==0 and i<20:
                    avg_iter_time += perf_counter() - mtime
                    if i==19:
                        avg_iter_time /= 20
                        print('avg iter time', avg_iter_time, 's')
                # itime = perf_counter()
            
            json.dump({key: eval(f'self.{key}') for key in ['test_acc', 'train_acc', 'test_loss']}, 
                        open(self.log_name+"/results.json", 'w'), indent=4)
            torch.save(self.model.state_dict(), self.log_name+"/epoch"+str(epoch+1)+".pt")
            for file in [self.log_name+"/epoch"+str(epoch)+".pt"]:
                if os.path.exists(file):
                    os.remove(file)
    
    
def main():
    options = read_options()
    t = Trainer(options)
    t.train()


if __name__ == "__main__":
    main()
    


