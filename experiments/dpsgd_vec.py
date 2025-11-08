
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

import warnings
warnings.filterwarnings("ignore")

codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)
from trainers import BaseTrainer, read_options

from functorch import make_functional, vmap, grad_and_value

    
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

        func_model, weights = make_functional(self.model)
        
        def compute_loss(weights, x, y):
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            predicted = func_model(weights, x)
            loss = self.loss(predicted, y)
            return loss

        self.compute_grad_and_loss = grad_and_value(compute_loss)
        self.per_sample_grad_and_loss = vmap(self.compute_grad_and_loss, in_dims=(None, 0, 0))


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
    
    def train(self):
        total_step = len(self.train_loader)
        test_interval = total_step // self.num_test_per_epoch
        
        avg_iter_time = 0
        avg_model_time = 0
        
        for epoch in range(self.epoch_start, self.epochs):

            if epoch % self.eval_every_epoch == 0 and epoch > 0:
                train_accu = self.get_train_accuracy()
                print('epoch {} train accuracy {:.5f}'.format(epoch, train_accu), flush=True)
                self.train_acc.append(train_accu)
                self.model.train()

            itime = perf_counter()
            for i, (xs, ys) in enumerate(self.train_loader):
                mtime = perf_counter()
                grad_vec = None
                _, weights = make_functional(self.model)
                
                if i % test_interval == 0:
                    test_accu, L_test = self.get_test_accuracy_and_loss()
                    print('epoch', epoch, 'iter', i, 'test accuracy', test_accu, 'test loss', L_test)
                    self.test_loss.append(L_test)
                    self.test_acc.append(test_accu)
                    self.model.train()

                xs = xs.to(self.device)
                ys = ys.to(self.device)

                B = xs.size(0)
                microbatch_size = int(math.ceil(B / self.num_microbatches))
                
                for j in range(self.num_microbatches):
                    lower = j * microbatch_size
                    upper = min((j + 1) * microbatch_size, B)
                    self.model.zero_grad()
                    
                    grads, loss = self.per_sample_grad_and_loss(weights, xs[lower:upper], ys[lower:upper])
                    
                    with torch.no_grad():
                        grad_tensor = []
                        for grad in grads:
                            grad_tensor.append(grad.view(grad.size(0), -1).detach())
                        del grads
                        grad_tensor = torch.cat(grad_tensor, 1)
                        grad_norm = grad_tensor.norm(2, 1)
                        multiplier = grad_norm.new(grad_norm.size()).fill_(1)
                        multiplier[grad_norm.gt(self.clipping_bound)] = self.clipping_bound / grad_norm[grad_norm.gt(self.clipping_bound)]
                        grad_tensor *= multiplier.unsqueeze(1)
                        if grad_vec is None:
                            grad_vec = grad_tensor.sum(0)
                        else:
                            grad_vec += grad_tensor.sum(0)
                            

                with torch.no_grad():
                    grad_vec = (grad_vec + self.sigma * self.clipping_bound * torch.randn_like(grad_vec).to(self.device)) / B
                    self.set_grad_to_vec(grad_vec)

                self.optimizer.step()
                
                if epoch==0 and i<20:
                    avg_iter_time += perf_counter() - itime
                    avg_model_time += perf_counter() - mtime
                    if i==19:
                        avg_iter_time /= 20
                        avg_model_time /= 20
                        print('avg iter time', avg_iter_time, 's', 'avg model time', avg_model_time, 's')
                        # return
                itime = perf_counter()
                
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

