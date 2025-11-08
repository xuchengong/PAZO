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
import torch.utils.data as data
from torchvision.models import *

codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)
from trainers import BaseTrainer, read_options

    
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
    

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, 
        #                                  momentum=0.9, weight_decay=self.wd)
    
    
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        lr = self.lr * (0.1 ** (epoch // 10))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
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
        tmp_g = torch.cat([p.data.clone().view(-1) for _, p in self.model.named_parameters()])
        self.D = len(tmp_g)
        # self.num_candidate = math.ceil(self.public_size / self.public_bs)
        del tmp_g
        print(self.D, 'parameters in the model', flush=True)
        total_step = len(self.train_loader)
        test_interval = total_step // self.num_test_per_epoch
        print('test_interval', test_interval, flush=True)
        train_accu = self.get_train_accuracy()
        print('init train accuracy {:.5f}'.format(train_accu), flush=True)
        self.model.train()
        
        avg_iter_time = 0
        avg_model_time = 0
        top_test_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.epoch_start, self.epochs):
            # self.adjust_learning_rate(epoch)
            
            self.model.train()

            running_train_loss = 0.0
            running_train_correct = 0
            num_train_sample = 0
            
            itime = perf_counter()
            for i, (x, mask, y, pos) in enumerate(self.train_loader):
                mtime = perf_counter()
                
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
                B = y.size(0)
                num_train_sample += B
                
                self.optimizer.zero_grad()
                predicted = self.model(input_ids=x, attention_mask=mask, mask_pos=pos)
                l = self.loss(predicted, y)
                l.backward()
                self.optimizer.step()
                
                running_train_loss += l.item() * B
                running_train_correct += (predicted.argmax(1) == y).sum().item()
                
                if epoch==0 and i<20:
                    avg_iter_time += perf_counter() - itime
                    avg_model_time += perf_counter() - mtime
                    
                    if i==19:
                        avg_iter_time /= 20
                        avg_model_time /= 20
                        print('avg iter time', avg_iter_time, 's', 'avg model time', avg_model_time, 's')
                        # return
                itime = perf_counter()
            
            train_acc = running_train_correct / num_train_sample
            train_loss = running_train_loss / num_train_sample
            
            self.train_acc.append(train_acc)
            print('epoch {}, train loss {:.5f}, train acc {:.5f}, test acc {:.5f}'.format(epoch, train_loss, train_acc, test_accu), flush=True)
            
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

