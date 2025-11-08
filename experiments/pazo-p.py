
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
        
    def compute_loss(self, params, inputs, targets):
        pred = torch.func.functional_call(self.model, params, (inputs,))
        return self.loss_flat(pred, targets)
    
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
    
    def train(self):
        total_step = len(self.train_loader)
        test_interval = total_step // self.num_test_per_epoch
        
        tmp_g = torch.cat([p.data.clone().view(-1) for _, p in self.model.named_parameters()])
        self.D = len(tmp_g)
        print(f"D={self.D}")
        del tmp_g

        avg_iter_time = 0
        
        for epoch in range(self.epoch_start, self.epochs):
            
            if epoch % self.eval_every_epoch == 0:
                L_test = self.get_test_loss()
                train_accu = self.get_train_accuracy()
                print('epoch {} test loss {:.5f} train accuracy {:.5f}'.format(epoch, L_test, train_accu), flush=True)
                self.train_acc.append(train_accu)
                self.test_loss.append(L_test)
                self.model.train()

            for i, (xs, ys) in enumerate(self.train_loader):
                t1 = perf_counter()
            
                if i % test_interval == 0 and i > 0:
                    test_accu = self.get_test_accuracy()
                    print('epoch', epoch, 'iter', i, 'test accuracy', test_accu)
                    self.test_acc.append(test_accu)
                    self.model.train()

                xs = xs.to(self.device)
                ys = ys.to(self.device)
                B = xs.size(0)
                
                curr_weight = torch.cat([p.data.clone().view(-1) for _, p in self.model.named_parameters()])
                overall_grad = torch.zeros_like(curr_weight, device=self.device)
                
                G = self.get_pub_candidates()
                print('G', G.T @ G)
                self.model.zero_grad()
                
                for _ in range(self.num_directions):
                    coeffs = torch.randn(self.num_candidate, device=self.device) # coeffs has var I_n
                    perturb = G @ coeffs
                    weight = self.get_dict_from_vec(curr_weight + perturb * self.perturbation_scale)
                    l1 = self.compute_loss(weight, xs, ys)
                    
                    weight = self.get_dict_from_vec(curr_weight - perturb * self.perturbation_scale)
                    l2 = self.compute_loss(weight, xs, ys)

                    l = 0.5 * (l1 - l2) / self.perturbation_scale
                    if epoch%20==0 and i==0:
                        print('l', l)
                    sum_clipped_loss = torch.sum(self.loss_clip(l, self.clipping_bound))
                    noisy_l = (sum_clipped_loss + self.clipping_bound * np.sqrt(self.num_directions) * self.sigma * torch.randn_like(sum_clipped_loss)) / B
                        
                    overall_grad += noisy_l * perturb.clone()
                
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
                x_public, y_public = next(self.public_iterator)
            except:
                self.public_iterator = iter(self.public_loader)
                x_public, y_public = next(self.public_iterator)
            x_public = x_public.to(self.device)
            y_public = y_public.to(self.device)
            predicted = self.model(x_public)
            l = self.loss(predicted, y_public)
            self.model.zero_grad()
            l.backward()
            g_public = torch.cat([p.grad.clone().view(-1) for _, p in self.model.named_parameters()])
            # print('pub candidate norm', torch.norm(g_public).item())
            G[:, j] = g_public.clone() / torch.norm(g_public)
        return G
    

def main():
    options = read_options()
    t = Trainer(options)
    t.train()


if __name__ == "__main__":
    main()
    
    
