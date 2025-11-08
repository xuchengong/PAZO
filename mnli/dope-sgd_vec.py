
import os
import sys
import json
import math
import copy
import random
import numpy as np
from time import perf_counter

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.models import *

codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)
from trainers import BaseTrainer, read_options

import warnings
warnings.filterwarnings("ignore")

from functorch import make_functional_with_buffers, vmap, grad_and_value

    
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
        
        func_model, weights, buffers = make_functional_with_buffers(self.model)
        
        def compute_loss(weights, buffers, x, mask, y):
            (x, mask, y) = (x.unsqueeze(0), mask.unsqueeze(0), y.unsqueeze(0))
            outputs = func_model(weights, buffers, input_ids=x, attention_mask=mask)
            loss = self.loss(outputs.logits, y)
            return loss

        self.compute_grad_and_loss = grad_and_value(compute_loss)
        self.per_sample_grad_and_loss = vmap(self.compute_grad_and_loss, in_dims=(None, None, 0, 0, 0),
                                             randomness='different')
    
    
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
    
    
    def get_train_accuracy(self):
        self.model.eval()
        with torch.no_grad():
            total_sample = 0
            num_correct = 0
            for i, (x, mask, y) in enumerate(self.train_loader):
                (x, mask, y) = (x.to(self.device), mask.to(self.device), y.to(self.device))
                outputs = self.model(input_ids=x, attention_mask=mask, labels=y)
                predicted = torch.argmax(outputs.logits, 1)
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
            for i, (x, mask, y) in enumerate(self.test_loader):
                (x, mask, y) = (x.to(self.device), mask.to(self.device), y.to(self.device))
                outputs = self.model(input_ids=x, attention_mask=mask)
                l = self.loss_sum(outputs.logits, y)
                loss += l.item()
                predicted = torch.argmax(outputs.logits, 1)
                num_correct += torch.sum((predicted == y).float()).item()
                total_sample += len(y)
            return num_correct / total_sample, loss / total_sample
        
    def train(self):
        total_step = len(self.train_loader)
        test_interval = total_step // self.num_test_per_epoch
        tmp_g = torch.cat([p.data.clone().view(-1) for _, p in self.model.named_parameters()])
        self.D = len(tmp_g)
        print('D:', self.D)
        # self.num_candidate = math.ceil(self.public_size / self.public_bs)
        del tmp_g
        avg_iter_time = 0
        avg_model_time = 0
        top_test_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.epoch_start, self.epochs):
            
            if epoch % self.eval_every_epoch == 0:
                train_accu = self.get_train_accuracy()
                print('epoch {} train accuracy {:.5f}'.format(epoch, train_accu), flush=True)
                self.train_acc.append(train_accu)
                self.model.train()
                    
            itime = perf_counter()
            for i, (x, mask, y) in enumerate(self.train_loader):
                
                mtime = perf_counter()
                grad_vec = None
                _, weights, buffers = make_functional_with_buffers(self.model)
                
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
                        
                (x, mask, y) = (x.to(self.device), mask.to(self.device), y.to(self.device))
                
                B = x.size(0)
                microbatch_size = int(math.ceil(B / self.num_microbatches))

                # get g_public
                g_public = self.get_g_avg_public().unsqueeze(0) # (D, ) -> (1, D)
                
                for j in range(self.num_microbatches):
                    lower = j * microbatch_size
                    upper = min((j + 1) * microbatch_size, B)
                    self.model.zero_grad()
                    
                    grads, loss = self.per_sample_grad_and_loss(weights, buffers, x[lower:upper], mask[lower:upper], y[lower:upper])
                    
                    with torch.no_grad():
                        grad_tensor = []
                        for grad in grads:
                            grad_tensor.append(grad.reshape(grad.size(0), -1).detach())
                        del grads
                        grad_tensor = torch.cat(grad_tensor, 1) # (microbatch_size, D)
                        grad_tensor -= g_public
                        grad_norm = grad_tensor.norm(2, 1)
                        multiplier = grad_norm.new(grad_norm.size()).fill_(1)
                        multiplier[grad_norm.gt(self.clipping_bound)] = self.clipping_bound / grad_norm[grad_norm.gt(self.clipping_bound)]
                        grad_tensor *= multiplier.unsqueeze(1)
                        if grad_vec is None:
                            grad_vec = grad_tensor.sum(0)
                        else:
                            grad_vec += grad_tensor.sum(0)
                
                grad_vec += g_public.squeeze() * B # (D,)
                grad_vec += self.clipping_bound * self.sigma * torch.randn_like(grad_vec)
                grad_vec /= B
                
                self.set_grad_to_vec(grad_vec)
                self.optimizer.step()
                
                if epoch==0 and i<20:
                    avg_iter_time += perf_counter() - itime
                    avg_model_time += perf_counter() - mtime
                    if i==19:
                        avg_iter_time /= 20
                        avg_model_time /= 20
                        print('avg iter time', avg_iter_time, 's', 'avg model time', avg_model_time, 's')
                itime = perf_counter()

            
            json.dump({key: eval(f'self.{key}') for key in ['test_acc', 'train_acc', 'test_loss']}, 
                        open(self.log_name+"/results.json", 'w'), indent=4)
            torch.save(self.model.state_dict(), self.log_name+"/epoch"+str(epoch+1)+".pt")
            for file in [self.log_name+"/epoch"+str(epoch)+".pt"]:
                if os.path.exists(file):
                    os.remove(file)
    
    
    def get_g_avg_public(self):
        
        try:
            x_pub, mask_pub, y_pub = next(self.public_iterator)
        except:
            self.public_iterator = iter(self.public_loader)
            x_pub, mask_pub, y_pub = next(self.public_iterator)
        
        (x_pub, mask_pub, y_pub) = (x_pub.to(self.device), mask_pub.to(self.device), y_pub.to(self.device))
        outputs = self.model(input_ids=x_pub, attention_mask=mask_pub)
        l = self.loss(outputs.logits, y_pub)
        self.model.zero_grad()
        l.backward()
        
        g_public = torch.cat([p.grad.clone().view(-1) for _, p in self.model.named_parameters()])
        return g_public
    
    
def main():
    options = read_options()
    t = Trainer(options)
    t.train()


if __name__ == "__main__":
    main()

