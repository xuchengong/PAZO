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
import torchvision.transforms as transforms
from torchvision.models import *

codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)
from trainers import BaseTrainer, read_options
from functorch import make_functional, make_functional_with_buffers, grad, vmap, grad_and_value
import warnings
warnings.filterwarnings("ignore")
    
class Trainer(BaseTrainer):
    def __init__(self, params):
        super(Trainer, self).__init__(params)
        
        for key, val in params.items():
            setattr(self, key, val)
        
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

    
    def use_g_public(self):
        try:
            (x, mask, y, pos) = next(self.public_iterator)
        except:
            self.public_iterator = iter(self.public_loader)
            (x, mask, y, pos) = next(self.public_iterator)
        
        (x, mask, y, pos) = (x.to(self.device), mask.to(self.device), y.to(self.device), pos.to(self.device))
        predicted = self.model(input_ids=x, attention_mask=mask, labels=y, mask_pos=pos)
        l = self.loss(predicted, y)
        self.model.zero_grad()
        l.backward()
        
        g_public = torch.cat([p.grad.clone().view(-1) for _, p in self.model.named_parameters()])
        return g_public
    
    
    def test_alpha(self):
        alphas = []
        num_iter = len(self.train_loader)
        total_iter = num_iter * self.epochs
        
        for epoch in range(self.epoch_start, self.epochs):
            for i in range(387):
                t = epoch * num_iter + i
                self.alpha = np.cos(np.pi*t/(2*total_iter))
                alphas.append(self.alpha)
        json.dump({'alpha': alphas}, open(self.log_name+"/alpha.json", 'w'))
        
    def get_train_accuracy(self):
        self.model.eval()
        with torch.no_grad():
            total_sample = 0
            num_correct = 0
            for i, (x, mask, y, pos) in enumerate(self.train_loader):
                (x, mask, y, pos) = (x.to(self.device), mask.to(self.device), y.to(self.device), pos.to(self.device))
                pred = self.model(input_ids=x, attention_mask=mask, labels=y, mask_pos=pos)
                predicted = torch.argmax(pred, 1)
                num_correct += torch.sum((predicted == y).float()).item()
                total_sample += len(y)
                if total_sample >= 2000:
                    break
            return num_correct / total_sample
    
    def get_test_accuracy_and_loss(self):
        self.model.eval()
        with torch.no_grad():
            total_sample = 0
            num_correct = 0
            loss = 0
            for i, (x, mask, y, pos) in enumerate(self.test_loader):
                (x, mask, y, pos) = (x.to(self.device), mask.to(self.device), y.to(self.device), pos.to(self.device))
                pred = self.model(input_ids=x, attention_mask=mask, mask_pos=pos)
                l = self.loss_sum(pred, y)
                loss += l.item()
                predicted = torch.argmax(pred, 1)
                num_correct += torch.sum((predicted == y).float()).item()
                total_sample += len(y)
            return num_correct / total_sample, loss / total_sample
        
    def train(self):
        total_step = len(self.train_loader)
        test_interval = total_step // self.num_test_per_epoch
        tmp_g = torch.cat([p.data.clone().view(-1) for _, p in self.model.named_parameters()])
        self.D = len(tmp_g)
        num_iter = len(self.train_loader)
        total_iter = num_iter * self.epochs
        avg_iter_time = 0
        top_test_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.epoch_start, self.epochs):

            if epoch % self.eval_every_epoch == 0 and epoch > 0:
                train_accu = self.get_train_accuracy()
                print('epoch {} train accuracy {:.5f}'.format(epoch, train_accu), flush=True)
                self.train_acc.append(train_accu)
                self.model.train()

            # itime = perf_counter()
            for i, (x, mask, y, pos) in enumerate(self.train_loader):
                mtime = perf_counter()
                
                t = epoch * num_iter + i
                self.alpha = np.cos(np.pi*t/(2*total_iter))
                
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

                (x, mask, y, pos) = (x.to(self.device), mask.to(self.device), y.to(self.device), pos.to(self.device))
                saved_var = torch.zeros(self.D).to(self.device)

                B = x.size(0)
                microbatch_size = int(math.ceil(B / self.num_microbatches))
                
                for j in range(self.num_microbatches):
                    lower = j * microbatch_size
                    upper = min((j + 1) * microbatch_size, B)
                    predicted = self.model(input_ids=x[lower:upper], attention_mask=mask[lower:upper], mask_pos=pos[lower:upper])                    
                    l = self.loss_flat(predicted, y[lower:upper])  # (microbatch_size, )
                    self.model.zero_grad()
                    # compute per-sample grads using vmap
                    grads = torch.autograd.grad(l, self.model.parameters(), torch.eye(len(l)).to(self.device), retain_graph=False, is_grads_batched=True) # num_layer of (microbatch_size, num_params, ...)
                    grads = [layer.view(layer.size(0), -1) for layer in grads]
                    grads = torch.cat(grads, dim=1) # (microbatch_size, D)
                    grads = grads.t() # (D, microbatch_size)
                    
                    if self.clipping_bound > 1e-6:
                        norm_ = torch.norm(grads, dim=0, keepdim=True).to(self.device) # (1, microbatch_size)
                        if epoch%20==0 and i==0:
                            print('grad norm max', norm_.max().item(), 'min', norm_.min().item())
                        multiplier = torch.ones_like(norm_)
                        exceeded = (norm_ > self.clipping_bound)
                        multiplier[exceeded] = self.clipping_bound / (norm_[exceeded] + 1e-6)
                    else:
                        multiplier = torch.ones_like(l).unsqueeze(0)
                    
                    saved_var += (multiplier * grads).sum(dim=1)
                
                saved_var += self.clipping_bound * self.sigma * torch.randn_like(saved_var)
                saved_var /= B
                g_public = self.use_g_public()
                saved_var = self.alpha * saved_var  + (1 - self.alpha) * g_public
                
                for p_name, p in self.model.named_parameters():
                    num_params = p.numel()
                    p.grad = saved_var[:num_params].view_as(p).clone()
                    saved_var = saved_var[num_params:]
                assert saved_var.size(0) == 0

                self.optimizer.step()  # apply p.grad
                
                if epoch==0 and i<20:
                    avg_iter_time += perf_counter() - mtime
                    if i==19:
                        avg_iter_time /= 20
                        print('avg iter time', avg_iter_time, 's')

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
    # t.test_alpha()


if __name__ == "__main__":
    main()

