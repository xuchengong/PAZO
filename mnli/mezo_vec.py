
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
        
        
        def compute_loss(params, buffers, x, mask, y, pos):
            (x, mask, y, pos) = (x.unsqueeze(0), mask.unsqueeze(0), y.unsqueeze(0), pos.unsqueeze(0))
            pred = functional_call(self.model, (params, buffers), kwargs={"input_ids": x, "attention_mask": mask, "mask_pos": pos})
            return self.loss(pred, y)  # returns scalar
    
        self.batched_loss_fn = vmap(compute_loss, in_dims=(None, None, 0, 0, 0, 0), randomness='different') # over b samples


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

            for i, (x, mask, y, pos) in enumerate(self.train_loader):
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

                (x, mask, y, pos) = (x.to(self.device), mask.to(self.device), y.to(self.device), pos.to(self.device))
                B = x.size(0)
                
                params, buffers = stack_module_state([self.model])
                buffers = tree_map(lambda p: p[0], buffers)
                
                # Sample random directions matching param tree
                directions = tree_map(lambda p: torch.randn(self.num_directions, *p[0].shape, device=p.device), params)

                # Apply +delta and -delta to each direction
                params_plus  = tree_map(lambda p, d: p + self.perturbation_scale * d, params, directions)
                params_minus = tree_map(lambda p, d: p - self.perturbation_scale * d, params, directions)
                
                query_loss_fn = vmap(self.batched_loss_fn, in_dims=(0, None, None, None, None, None), randomness='different') # [q, b]
                losses_plus  = query_loss_fn(params_plus,  buffers, x, mask, y, pos).mean(dim=1)
                losses_minus = query_loss_fn(params_minus, buffers, x, mask, y, pos).mean(dim=1)
                
                # Compute directional derivatives and scale by u_i
                scalar_diff = (losses_plus - losses_minus) / (2 * self.perturbation_scale)

                # Multiply each scalar diff back with direction u_i
                overall_grad = tree_map(lambda u, s: (s * u).mean(dim=0), directions, scalar_diff)
                
                update_norm = 0
                for p_name, p in self.model.named_parameters():
                    p.grad = overall_grad[p_name]
                    update_norm += torch.sum(p.grad ** 2).item()
                if epoch % 20 == 0 and i == 0:
                    print('update norm', math.sqrt(update_norm))
                
                self.optimizer.step()  # apply p.grad
                
                if epoch==0 and i<20:
                    avg_iter_time += perf_counter() - t1
                    if i==19:
                        avg_iter_time /= 20
                        print('avg iter time', avg_iter_time, 's')
                        # return
            
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
    
    
