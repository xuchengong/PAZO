import numpy as np
import copy
import math
import sys
import random
from typing import Tuple

import torch
import torchvision
import torchvision.datasets as datasets
import os
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.models import *
import torch.optim.lr_scheduler as lr_scheduler

codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)

from .model.nfresnet import nf_resnet18
from .model.lstm import LSTM

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2023, 0.1994, 0.2010])
CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408])
CIFAR100_STD = np.array([0.2675, 0.2565, 0.2761])
TINYIMGNET_MEAN = np.array([0.4802, 0.4481, 0.3975])
TINYIMGNET_STD = np.array([[0.2302, 0.2265, 0.2262]])
PLACES365_MEAN = np.array([0.485, 0.456, 0.406])
PLACES365_STD = np.array([0.229, 0.224, 0.225])

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Cutout(torch.nn.Module):
    def __init__(self, crop_size: int, fill: Tuple[float, float, float] = (0, 0, 0)):
        super().__init__()
        self.crop_size = crop_size
        self.fill = torch.tile(torch.tensor(fill), (self.crop_size, self.crop_size, 1)).permute(2, 0, 1)

    def forward(self, img):
        coord = (np.random.randint(img.shape[1] - self.crop_size + 1),
                 np.random.randint(img.shape[2] - self.crop_size + 1),
        )
        img = img.clone()
        img[:, coord[0]:coord[0] + self.crop_size, coord[1]:coord[1] + self.crop_size] = self.fill
        return img

    
class BaseTrainer(object):
    def __init__(self, params):
        for key, val in params.items():
            setattr(self, key, val)

        if self.dataset == 'cifar10':
            num_classes = 10
            assert self.public_size==2000
        elif self.dataset == 'imdb10k':
            num_classes = 2
        elif self.dataset == 'tiny-imagenet':
            num_classes = 200
        elif self.dataset == 'places365':
            num_classes = 365
        elif self.dataset == 'mnli_snli_512':
            num_classes = 3
        else:
            raise ValueError('unknown dataset {}'.format(self.dataset))
        
        if self.pretrain=='nfresnet18':
            self.model = nf_resnet18()
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
            self.model.load_state_dict(torch.load('nfresnet18_public2000.pth'))
            print('loaded public2000-warmup nfresnet18', flush=True)
        elif self.pretrain=='vit':
            import timm
            self.model = timm.create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=False)
            self.model.head = nn.Linear(self.model.head.in_features, num_classes, bias=True)
            self.model.load_state_dict(torch.load('ViT10M_public.pt'))
            print('loaded places365 pretrained public-warmup ViT', flush=True)
        elif self.pretrain=='lstm':
            self.model = LSTM(vocab_size=10000+2)
            print('loaded LSTM from scratch', flush=True)
        elif self.pretrain=='roberta':
            from transformers import AutoConfig
            from .model.roberta_prompt import RobertaModelForPromptFinetuning
            config = AutoConfig.from_pretrained("roberta-base")
            self.model = RobertaModelForPromptFinetuning(config)
            print('loaded pretrained RobertaModelForPromptFinetuning', flush=True)
        else:
            raise ValueError('unknown pretrain value {}'.format(self.pretrain))

        self.set_dataset()
        
        self.loss = nn.CrossEntropyLoss()
        self.loss_sum = nn.CrossEntropyLoss(reduction='sum')
        self.loss_flat = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        assert torch.cuda.is_available()
        self.device = torch.device("cuda")
        self.model.to(self.device)
        
        self.log_name = os.path.join(os.path.dirname(__file__), f'../logs/{self.time}')

    def set_dataset(self):
        if self.dataset == 'cifar10':
            nonaugmented_transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                ])
            augmented_transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                Cutout(8, CIFAR10_MEAN),
                            ])
        
            print('processing CIFAR10 dataset...', flush=True)
            data_path = os.path.join(os.path.dirname(__file__), '../data')
            train_set = torchvision.datasets.CIFAR10(root=data_path,
                                                        train=True,
                                                        transform=nonaugmented_transform,
                                                        download=False)
            self.train_dataset = data.Subset(train_set, range(self.public_size, len(train_set)))
            public_set = torchvision.datasets.CIFAR10(root=data_path,
                                                    train=True,
                                                    transform=augmented_transform if self.augmented==1 else nonaugmented_transform,
                                                    download=False)
            self.public_dataset = data.Subset(public_set, range(0, self.public_size))
            self.test_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                                                train=False,
                                                                transform=nonaugmented_transform,
                                                                download=False
                                                                )
            del train_set, public_set
            
            self.private_size = len(self.train_dataset)
            self.training_sample_size = self.private_size
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.private_bs, 
                                                            num_workers=2, shuffle=True, drop_last=True)
            self.public_loader = torch.utils.data.DataLoader(dataset=self.public_dataset, 
                                                            batch_size=self.public_bs, 
                                                            num_workers=2, shuffle=True, drop_last=True)
            self.public_iterator = iter(self.public_loader)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=64, 
                                                        num_workers=2, shuffle=False)

            print('data processing done')
            
        elif self.dataset == 'imdb10k':
            print('loading imdb10k...')
            public_data = dict(np.load(os.path.join(os.path.dirname(__file__), '../data/imdb10k/public.npz')))
            train_data = dict(np.load(os.path.join(os.path.dirname(__file__), '../data/imdb10k/train.npz')))
            test_data = dict(np.load(os.path.join(os.path.dirname(__file__), '../data/imdb10k/test.npz')))
            x_public, y_public = public_data['x'],  public_data['y']
            x_train, y_train = train_data['x'], train_data['y']
            x_test, y_test = test_data['x'], test_data['y']
            print('x public: ', x_public.shape) # (1000, 290)
            print('x train: ', x_train.shape)  # (25000, 290)
            print('x test: ', x_test.shape) # (2000, 290)
            self.training_sample_size = x_train.shape[0]
            self.public_size = 1000

            self.public_dataset = data.TensorDataset(torch.LongTensor(x_public), torch.LongTensor(y_public))
            self.train_dataset = data.TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
            self.test_dataset = data.TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
            
            self.public_loader = torch.utils.data.DataLoader(dataset=self.public_dataset,
                                                            batch_size=self.public_bs,
                                                            num_workers=0,
                                                            shuffle=True, drop_last=True)
            self.public_iterator = iter(self.public_loader)
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                            batch_size=self.private_bs,
                                                            num_workers=0,
                                                            shuffle=True, drop_last=True)

            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                        batch_size=256,
                                                        num_workers=0,
                                                        shuffle=False)
        
        elif self.dataset == 'mnli_snli_512':
            print('loading mnli_snli_512...')
            public_data = torch.load(os.path.join(os.path.dirname(__file__), '../data/mnli_snli_512/public.pt'))
            train_data = torch.load(os.path.join(os.path.dirname(__file__), '../data/mnli_snli_512/train.pt'))
            test_data = torch.load(os.path.join(os.path.dirname(__file__), '../data/mnli_snli_512/test.pt'))
            x_public, mask_public, y_public, pos_public = [torch.LongTensor(public_data[key]) for key in ['input_ids', 'attention_mask', 'label', 'mask_pos']]
            x_train, mask_train, y_train, pos_train = [torch.LongTensor(train_data[key]) for key in ['input_ids', 'attention_mask', 'label', 'mask_pos']]
            x_test, mask_test, y_test, pos_test = [torch.LongTensor(test_data[key]) for key in ['input_ids', 'attention_mask', 'label', 'mask_pos']]
            print('x public: ', x_public.shape) # (300, 256)
            print('x train: ', x_train.shape)  # (1536, 256)
            print('x test: ', x_test.shape) # (1000, 256)
            self.training_sample_size = x_train.shape[0]
            self.public_size = x_public.shape[0]

            self.public_dataset = data.TensorDataset(x_public, mask_public, y_public, pos_public)
            self.train_dataset = data.TensorDataset(x_train, mask_train, y_train, pos_train)
            self.test_dataset = data.TensorDataset(x_test, mask_test, y_test, pos_test)
            
            self.public_loader = torch.utils.data.DataLoader(dataset=self.public_dataset,
                                                            batch_size=self.public_bs,
                                                            num_workers=0,
                                                            shuffle=True, drop_last=True)
            self.public_iterator = iter(self.public_loader)
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                            batch_size=self.private_bs,
                                                            num_workers=0,
                                                            shuffle=True, drop_last=True)

            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                        batch_size=256,
                                                        num_workers=0,
                                                        shuffle=False)
            
        elif self.dataset == 'tiny-imagenet':
            nonaugmented_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
            augmented_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                    Cutout(8, TINYIMGNET_MEAN),
                                ])
        
            print('processing Tiny Imagenet dataset...', flush=True)
            data_path = os.path.join(os.path.dirname(__file__), '../../tiny-224')
            self.train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), nonaugmented_transform)
            self.public_dataset = datasets.ImageFolder(os.path.join(data_path, 'pub'), augmented_transform)
            self.test_dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), nonaugmented_transform)
            self.private_size = len(self.train_dataset)
            self.public_size = len(self.public_dataset)
            print('private_size', self.private_size, 'public_size', self.public_size)
            self.training_sample_size = self.private_size
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.private_bs, 
                                                            pin_memory=True, num_workers=4, shuffle=True, drop_last=True)
            self.public_loader = torch.utils.data.DataLoader(dataset=self.public_dataset, 
                                                            pin_memory=True, batch_size=self.public_bs, 
                                                            num_workers=4, shuffle=True, drop_last=True)
            self.public_iterator = iter(self.public_loader)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=64, 
                                                        pin_memory=True, num_workers=4, shuffle=False)

            print('data processing done')
        elif self.dataset == 'places365':
            nonaugmented_transform = transforms.Compose([
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=PLACES365_MEAN, std=PLACES365_STD),
                                ])
            augmented_transform = transforms.Compose([
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=PLACES365_MEAN, std=PLACES365_STD),
                                    Cutout(8, PLACES365_MEAN),
                                ])
        
            print('processing Places365 dataset...', flush=True)
            data_path = os.path.join(os.path.dirname(__file__), '../../../data/places365')
            self.train_dataset = torchvision.datasets.Places365(data_path, 
                                                                split='train-standard', 
                                                                small=True, 
                                                                download=False, 
                                                                transform=augmented_transform)
            self.test_dataset = torchvision.datasets.Places365(data_path, 
                                                                split='val', 
                                                                small=True, 
                                                                download=False, 
                                                                transform=nonaugmented_transform)
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.private_bs, 
                                                            pin_memory=True, num_workers=8, shuffle=True, drop_last=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=128, 
                                                        pin_memory=True, num_workers=8, shuffle=False)

            print('data processing done')
        else:
            raise ValueError('unknown dataset {}'.format(self.dataset))

    def get_test_loss(self):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                predicted = self.model(images)
                l = self.loss(predicted, labels)
                loss += l.item()

        return loss/i
    
    def get_test_accuracy(self):
        self.model.eval()
        with torch.no_grad():
            total_sample = 0
            num_correct = 0
            # loss = 0
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                predicted = self.model(images)
                # l = self.loss_sum(predicted, labels)
                # loss += l.item()
                predicted = torch.argmax(predicted, 1)
                num_correct += torch.sum((predicted == labels).float()).item()
                total_sample += len(labels)
            return num_correct / total_sample
    
    def get_test_accuracy_and_loss(self):
        self.model.eval()
        with torch.no_grad():
            total_sample = 0
            num_correct = 0
            loss = 0
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                predicted = self.model(images)
                l = self.loss_sum(predicted, labels)
                loss += l.item()
                predicted = torch.argmax(predicted, 1)
                num_correct += torch.sum((predicted == labels).float()).item()
                total_sample += len(labels)
            return num_correct / total_sample, loss / total_sample

    def get_train_accuracy(self):
        self.model.eval()
        with torch.no_grad():
            total_sample = 0
            num_correct = 0
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                predicted = self.model(images)
                predicted = torch.argmax(predicted, 1)
                num_correct += torch.sum((predicted == labels).float()).item()
                total_sample += len(labels)
                if total_sample >= 2000:
                    break
            return num_correct / total_sample
    
    def get_pub_accuracy(self):
        self.model.eval()
        with torch.no_grad():
            total_sample = 0
            num_correct = 0
            for i, (images, labels) in enumerate(self.public_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                predicted = self.model(images)
                predicted = torch.argmax(predicted, 1)
                num_correct += torch.sum((predicted == labels).float()).item()
                total_sample += len(labels)
            return num_correct / total_sample

    def get_train_loss(self):
        with torch.no_grad():
            loss = 0
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                predicted = self.model(images)
                l = self.loss(predicted, labels)
                loss += l.item()
            return loss / i

    def loss_clip(self, loss_v, threshold):
        # element-wise clipping of losses in loss_v
        loss_v = torch.clamp(loss_v, min=-1*threshold, max=threshold)
        return loss_v