
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
        
        
    def train(self):
        total_step = len(self.train_loader)
        test_interval = total_step // self.num_test_per_epoch
        
        tmp_g = torch.cat([p.data.clone().view(-1) for _, p in self.model.named_parameters()])
        self.D = len(tmp_g)
        del tmp_g

        avg_iter_time = 0
        noise = dict()
        num_iter = len(self.train_loader)
        
        for epoch in range(self.epoch_start, self.epochs):
            
            if epoch % self.eval_every_epoch == 0:
                L_test = self.get_test_loss()
                train_accu = self.get_train_accuracy()
                print('epoch {} test loss {:.5f} train accuracy {:.5f}'.format(epoch, L_test, train_accu), flush=True)
                self.train_acc.append(train_accu)
                self.test_loss.append(L_test)
                self.model.train()

            count_noisy_win, count_true_win, count_sign_preserved = 0, 0, 0
            for i, (xs, ys) in enumerate(self.train_loader):
                t1 = perf_counter()
                
                if i % test_interval == 0 and i > 0:
                    test_accu = self.get_test_accuracy()
                    print('epoch', epoch, 'iter', i, 'test accuracy', test_accu)
                    self.test_acc.append(test_accu)
                    self.model.train()

                if self.pre_trained == 'linear':
                    xs = xs.reshape(-1, 32*32*3)

                xs = xs.to(self.device)
                ys = ys.to(self.device)
                l_min = 1e8
                best_g = 0
                
                pub_gs = self.get_pub_candidates()
                self.model.zero_grad()
                
                # find the best pub_g
                true_losses = []
                for j, g in enumerate(pub_gs):
                    for p_name, p in self.model.named_parameters():
                        p.data = p.data - self.lr * g[p_name]

                    pred = self.model(xs)
                    l = self.loss_flat(pred, ys)
                    if epoch%20==0 and i==0:
                        print('l', l)
                    true_loss = torch.mean(l).item()
                    true_losses.append(true_loss)
                    sum_clipped_loss = torch.sum(self.loss_clip(l, self.clipping_bound))
                    noisy_l = (sum_clipped_loss + np.sqrt(self.num_candidate+1) * self.clipping_bound * self.sigma * torch.randn_like(sum_clipped_loss)) / xs.size(0)
                    count_sign_preserved += (noisy_l.item() * true_loss > 0)
                    
                    if noisy_l.item() < l_min:
                        l_min = noisy_l
                        best_g = j
                    
                    for p_name, p in self.model.named_parameters():
                        p.data = p.data + self.lr * g[p_name]                    
                
                true_losses = torch.tensor(true_losses)
                if torch.argmin(true_losses) == best_g:
                    count_true_win += 1
                         
                # use the best pub_g + noise as the sampled random gradient
                for p_name, p in self.model.named_parameters():
                    noise[p_name] = torch.randn_like(p.data) * self.epsilon_scale
                    p.data = p.data - self.lr * (pub_gs[best_g][p_name] + noise[p_name])

                pred = self.model(xs)
                l = self.loss_flat(pred, ys)

                for p_name, p in self.model.named_parameters():
                    p.data = p.data + self.lr * (pub_gs[best_g][p_name] + noise[p_name])
                
                sum_clipped_loss = torch.sum(self.loss_clip(l, self.clipping_bound))
                noisy_l = (sum_clipped_loss + np.sqrt(self.num_candidate+1) * self.clipping_bound * self.sigma * torch.randn_like(sum_clipped_loss)) / xs.size(0)

                if noisy_l < l_min:
                    count_noisy_win += 1
                    for p_name, p in self.model.named_parameters():
                        p.grad = pub_gs[best_g][p_name] + noise[p_name]

                else:
                    for p_name, p in self.model.named_parameters():
                        p.grad = pub_gs[best_g][p_name]

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
        pub_g = []
        tmp_g = dict()
        for _ in range(self.num_candidate):
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
            
            for p_name, p in self.model.named_parameters():
                tmp_g[p_name] = p.grad
            pub_g.append(tmp_g)
        
        return pub_g
    

def main():
    options = read_options()
    t = Trainer(options)
    t.train()


if __name__ == "__main__":
    main()
    
    
