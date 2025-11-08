
import os
import sys
import json
import math
import copy
import random
from time import perf_counter
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torchvision.models import *


codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)
from trainers import BaseTrainer, read_options
from torch.utils._pytree import tree_map
from torch.func import functional_call, stack_module_state, vmap
    
class Trainer(BaseTrainer):
    def __init__(self, params):
        super(Trainer, self).__init__(params)
        
        for key in ['test_acc', 'train_acc', 'test_loss','noisy_win', 'true_win', 'sign_preserved']:
            setattr(self, key, [])
        self.epoch_start = 0
        
        if self.continue_from is not None:
            self.log_name = os.path.join(os.path.dirname(__file__), '../logs', self.continue_from)
            results = json.load(open(self.log_name+"/results.json", 'r'))
            
            for key in ['test_acc', 'train_acc', 'test_loss','noisy_win', 'true_win', 'sign_preserved']:
                setattr(self, key, results[key])
            
            for file in os.listdir(self.log_name):
                if file.endswith(".pt"):
                    self.epoch_start = int(file.split('.pt')[0][5:])
                    self.model.load_state_dict(torch.load(self.log_name+"/"+file))
            print('continue from', self.log_name, 'epoch', self.epoch_start)
    
        def compute_loss(params, buffers, x, y):
            with torch.no_grad():
                pred = functional_call(self.model, (params, buffers), (x,))
            return self.loss_flat(pred, y)

        self.query_loss_fn = vmap(compute_loss, in_dims=(0, None, None, None), randomness='different') # [q, b]

    def train(self):
        total_step = len(self.train_loader)
        test_interval = total_step // self.num_test_per_epoch
        
        tmp_g = torch.cat([p.data.clone().view(-1) for _, p in self.model.named_parameters()])
        self.D = len(tmp_g)
        del tmp_g

        avg_iter_time = 0
        top_test_acc = 0.0
        patience_counter = 0
        noise = dict()
        num_iter = len(self.train_loader)
        
        for epoch in range(self.epoch_start, self.epochs):
            
            if epoch % self.eval_every_epoch == 0:
                train_accu = self.get_train_accuracy()
                print('epoch {} train accuracy {:.5f}'.format(epoch, train_accu), flush=True)
                self.train_acc.append(train_accu)
                self.model.train()

            count_noisy_win, count_true_win, count_sign_preserved = 0, 0, 0
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
                
                params, buffers = stack_module_state([self.model])
                buffers = tree_map(lambda p: p[0], buffers)
                                
                pub_gs = self.get_pub_candidates()
                
                params_minus = tree_map(lambda p, d: p - self.lr * d, params, pub_gs)
                losses_minus = self.query_loss_fn(params_minus, buffers, x, y)
                true_losses = losses_minus.clone().sum(dim=1)
                losses_minus = self.loss_clip(losses_minus, self.clipping_bound).sum(dim=1)
                losses_minus = (losses_minus + np.sqrt(self.num_candidate+1) * self.clipping_bound * self.sigma * torch.randn_like(losses_minus)) / B
                
                # get both the smallest loss and the index                
                
                best_idx = torch.argmin(losses_minus).item()
                l_min = losses_minus[best_idx]
                best_g = tree_map(lambda p: p[best_idx], pub_gs)
                
                if torch.argmin(true_losses) == best_g:
                    count_true_win += 1
                
                self.model.zero_grad()
                # use the best pub_g + noise as the sampled random gradient
                for p_name, p in self.model.named_parameters():
                    noise[p_name] = torch.randn_like(p.data) * self.epsilon_scale
                    p.data = p.data - self.lr * (best_g[p_name] + noise[p_name])

                pred = self.model(x)
                l = self.loss_flat(pred, y)

                for p_name, p in self.model.named_parameters():
                    p.data = p.data + self.lr * (best_g[p_name] + noise[p_name])
                
                sum_clipped_loss = torch.sum(self.loss_clip(l, self.clipping_bound))
                noisy_l = (sum_clipped_loss + np.sqrt(self.num_candidate+1) * self.clipping_bound * self.sigma * torch.randn_like(sum_clipped_loss)) / x.size(0)

                if noisy_l < l_min:
                    count_noisy_win += 1
                    for p_name, p in self.model.named_parameters():
                        p.grad = best_g[p_name] + noise[p_name]
                else:
                    for p_name, p in self.model.named_parameters():
                        p.grad = best_g[p_name]

                self.optimizer.step()  # apply p.grad
                
                if epoch==0 and i<20:
                    avg_iter_time += perf_counter() - t1
                    if i==19:
                        avg_iter_time /= 20
                        print('avg iter time', avg_iter_time, 's')
                        # return
            
            self.noisy_win.append(count_noisy_win / num_iter)
            self.true_win.append(count_true_win / num_iter)
            self.sign_preserved.append(count_sign_preserved / num_iter / self.num_candidate)
            
            json.dump({key: eval(f'self.{key}') for key in ['test_acc', 'train_acc', 'test_loss','noisy_win', 'true_win', 'sign_preserved']}, 
                        open(self.log_name+"/results.json", 'w'), indent=4)
            torch.save(self.model.state_dict(), self.log_name+"/epoch"+str(epoch+1)+".pt")
            for file in [self.log_name+"/epoch"+str(epoch)+".pt"]:
                if os.path.exists(file):
                    os.remove(file)
    
    
    def get_pub_candidates(self):
        pub_g = dict()
        for i in range(self.num_candidate):
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
            
            if i==0:
                for p_name, p in self.model.named_parameters():
                    pub_g[p_name] = [p.grad]
            else:
                for p_name, p in self.model.named_parameters():
                    pub_g[p_name].append(p.grad)
            if i==self.num_candidate-1:
                for p_name, p in self.model.named_parameters():
                    pub_g[p_name] = torch.stack(pub_g[p_name], dim=0)
        return pub_g
    

def main():
    options = read_options()
    t = Trainer(options)
    t.train()


if __name__ == "__main__":
    main()
    
    
