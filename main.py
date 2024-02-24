# Copyright 2024 Jonghoon Im
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Main script to launch BAA training on CIFAR-10/100.

Supports WideResNet, ResNet,VGG models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python main.py`

"""
from __future__ import print_function

import argparse
import os
import shutil
import time

from models.cifar.allconv import AllConvNet
import numpy as np
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models

from PIL import Image

from randaugment import RandAugment, ImageNetPolicy
from cutout import Cutout
import BAA

import torchvision.transforms.autoaugment

import sys
import ssl
from torch.utils.data.sampler import SubsetRandomSampler

ssl._create_default_https_context = ssl._create_unverified_context

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Version check
# print(sys.version) #3.8.8
# print(torch.__version__)  #1.8.1->1.13.1+cu116
# print(torchvision.__version__) #0.9.1->0.14.1+cu116
# print(torch.cuda.is_available())
# print(torch.version.cuda) #10.2? 11.0?->11.6
# print(torch.backends.cudnn.version()) #?->8302
# print(np.__version__)
# CUDA 11.6
#pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
#conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100', 'piston data','svhn'],
    help='Choose dataset CIFAR-10, CIFAR-100, piston data.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='wrn40_2',
    choices=['wrn40_2','wrn28_10', 'allconv', 'densenet', 'resnext','resnet50','resnet101','resnet152','wide_resnet101_2','densenet201','resnext101_32x8d','vgg19','vgg19_bn'],
    help='Choose architecture.')
parser.add_argument(
    '--data_augmentation',
    '-da',
    type=str,
    default='transform_BAA',  #transform_BAA
    choices=['transform_Baseline','transform_Cutout', 'transform_RandAugment', 'transform_RandAugment_BAA', 'transform_AugMix','transform_AutoAug',
             'transform_AutoAug_SVHN','transform_Fast_AutoAug','transform_Fast_AutoAug_BAA','transform_BAA','transform_BAA_SVHN','transform_DADA',
             'transform_DADA_BAA'],
    help='Choose data_augmentation.')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,   #VGG19->0.01
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# WRN Architecture options
# parser.add_argument(
#     '--layers', default=40, type=int, help='total number of layers')
# parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
parser.add_argument(
    '--droprate', default=0.0, type=float, help='Dropout probability')

# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=0,
    help='Number of pre-fetching threads.')

args = parser.parse_args()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

#이미지 저장용 변수
global img_num
img_num=0

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

#train
def train(net, train_loader, optimizer, scheduler):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  total_loss = 0.
  total_correct = 0

  for i, (images, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    images = images.cuda()
    targets = targets.cuda()
    logits = net(images)
    loss = F.cross_entropy(logits, targets)

    loss.backward()
    optimizer.step()
    #scheduler.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    pred = logits.data.max(1)[1]
    #total_loss += float(loss_ema)
    total_correct += pred.eq(targets.data).sum().item()

    if i % args.print_freq == 0:
      print('Train Loss {:.3f}'.format(loss_ema))

  #print('Train Error {:.3f}'.format(100-100*total_correct / len(train_loader.dataset)))
  return loss_ema,total_correct/len(train_loader.dataset)

#validation
def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0

  with torch.no_grad():
      for images, targets in test_loader:
          images, targets = images.cuda(), targets.cuda()
          logits = net(images)
          loss = F.cross_entropy(logits, targets)
          pred = logits.data.max(1)[1]
          total_loss += float(loss.data)
          total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)

#test
def test2(net, test_loader):
  """Evaluate network on given dataset."""
  # switch to evaluate mode
  net.eval()
  total_loss = 0.
  total_correct = 0
  temp_i = 0
  global log_path
  with open(log_path, 'a') as f:
      f.write('Number, Predicted, Actual\n')
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

      for i in range(len(pred)):
          with open(log_path, 'a') as f:
              f.write('%05d,%05d,%05d\n' % (temp_i+1,pred[i],targets[i]))
          temp_i = temp_i+1
          #if temp_i==len(test_loader):
          #    break

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)

def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  # switch to evaluate mode
  net.eval()
  corruption_accs = []
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))
    with open(log_path, 'a') as f:
        f.write('{}\n\tTest Loss {:.3f} | Test Error {:.3f}\n'.format(
        corruption, test_loss, 100 - 100. * test_acc))
  return np.mean(corruption_accs)

def main():
  #seed
  torch.manual_seed(1)
  np.random.seed(1)

  print(args.data_augmentation)

  if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':

      val_transform = transforms.Compose(
          [transforms.ToTensor(),
           transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
           ])
      test_transform = transforms.Compose(
          [transforms.ToTensor(),
           transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
           ])

      if args.data_augmentation == 'transform_Baseline':
          train_transform = transforms.Compose(
          [
           transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
           ])

      elif args.data_augmentation == 'transform_Cutout':
          train_transform = transforms.Compose(
          [
           transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
           Cutout(1, 16)
           ])

      elif args.data_augmentation == 'transform_RandAugment':
          train_transform = transforms.Compose(
          [
              transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
              transforms.RandomHorizontalFlip(),
              RandAugment(),  ##op=random,magnitude=random
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          ])

      #In Pytorch Library
      # transform_RandAugment2 = transforms.Compose(
      #     [
      #     transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
      #     transforms.RandomHorizontalFlip(),
      #     #torchvision.transforms.RandAugment(), #op=2,magnitude=9
      #     BAA.RandAugment(),
      #     #Cutout(1,16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
      #     transforms.ToTensor(),
      #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      #     ])
      #transform_RandAugment2.transforms.insert(0, RandAugment(3, 7))

      elif args.data_augmentation == 'transform_RandAugment_BAA':
          train_transform = transforms.Compose(
          [
          transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
          transforms.RandomHorizontalFlip(),
          #torchvision.transforms.RandAugment(), #op=2,magnitude=9
          BAA.RandAugment_BAA(),
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          Cutout(1, 16)
          ])

      elif args.data_augmentation == 'transform_AugMix':
          train_transform = transforms.Compose(
          [
          transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
          transforms.RandomHorizontalFlip(),
          torchvision.transforms.AugMix(),
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          ])

      ######CIFAR10&100######
      elif args.data_augmentation == 'transform_AutoAug':
          train_transform = transforms.Compose(
          [
              transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
              transforms.RandomHorizontalFlip(),
              torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
              Cutout(1, 16)
          ])
      ######SVHN######
      elif args.data_augmentation == 'transform_AutoAug_SVHN':
          train_transform = transforms.Compose(
          [
              torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.SVHN),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
              Cutout(1, 20)
          ])

      elif args.data_augmentation == 'transform_Fast_AutoAug':
          train_transform = transforms.Compose(
          [
              transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
              transforms.RandomHorizontalFlip(),
              BAA.Fast_AutoAugment(policy=BAA.AutoAugmentPolicy.CIFAR10),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              Cutout(1, 16)
          ])

      elif args.data_augmentation == 'transform_Fast_AutoAug_BAA':
          train_transform = transforms.Compose(
          [
              transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
              transforms.RandomHorizontalFlip(),
              BAA.Fast_AutoAugment_BAA(policy=BAA.AutoAugmentPolicy.CIFAR10),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              Cutout(1, 16)
          ])

      ###CIFAR10&100
      elif args.data_augmentation == 'transform_BAA':
          train_transform = transforms.Compose(
          [
              transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
              transforms.RandomHorizontalFlip(),
              BAA.BAA(policy=BAA.AutoAugmentPolicy.CIFAR10),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
              Cutout(1, 16)
          ])
      #########SVHN############
      elif args.data_augmentation == 'transform_BAA_SVHN':
          train_transform = transforms.Compose(
          [
              BAA.BAA(policy=BAA.AutoAugmentPolicy.SVHN),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
              Cutout(1, 20)
          ])

      elif args.data_augmentation == 'transform_DADA':
          train_transform = transforms.Compose(
          [
              transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
              transforms.RandomHorizontalFlip(),
              #BAA.DADA_BAA(policy=BAA.DADAPolicy.CIFAR10), #When using the CIFAR-10 dataset,
              BAA.DADA(policy=BAA.DADAPolicy.CIFAR100),  #When using the CIFAR-100 dataset,
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
              Cutout(1, 16)
          ])

      #Applying synthesis weight randomly to images synthesized with policies found in DADA.
      elif args.data_augmentation == 'transform_DADA_BAA':
          train_transform = transforms.Compose(
          [
              transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
              transforms.RandomHorizontalFlip(),
              #BAA.DADA_BAA(policy=BAA.DADAPolicy.CIFAR10),  #When using the CIFAR-10 dataset,
              BAA.DADA_BAA(policy=BAA.DADAPolicy.CIFAR100),  #When using the CIFAR-100 dataset,
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              Cutout(1, 16)
          ])

  if args.dataset == 'cifar10':
      train_data = torchvision.datasets.CIFAR10(
          root='C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR10',
          train=True, download=True, transform=train_transform)

      val_data = torchvision.datasets.CIFAR10(
          root='C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR10',
          train=True, download=True, transform=val_transform)

      test_data = torchvision.datasets.CIFAR10(
          root='C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR10',
          train=False, download=True, transform=test_transform)

      train_loader = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True
      )

      val_loader = torch.utils.data.DataLoader(
          test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True
      )

      test_loader = torch.utils.data.DataLoader(
          test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
      )
      base_c_path = 'C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR-10-C/CIFAR-10-C/'
      num_classes = 10
      '''
      classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
      '''

  elif args.dataset == 'cifar100':
      train_data = torchvision.datasets.CIFAR100(
          root='C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR100',
          train=True, download=True, transform=train_transform)

      val_data = torchvision.datasets.CIFAR100(
          root='C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR100',
          train=True, download=True, transform=val_transform)

      test_data = torchvision.datasets.CIFAR100(
          root='C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR100',
          train=False, download=True, transform=test_transform)

      train_loader = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
      )

      val_loader = torch.utils.data.DataLoader(
          test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
      )

      test_loader = torch.utils.data.DataLoader(
          test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
      )
      #test_data = datasets.CIFAR100('C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR100', train=False, transform=test_transform, download=True)
      base_c_path = 'C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR-100-C/CIFAR-100-C/'
      num_classes = 100
      #train_data = datasets.CIFAR100('C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR100', train=True, transform=train_transform, download=True)

  elif args.dataset == 'svhn':
      train_data = torchvision.datasets.SVHN(
          root='C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/SVHN',
          split='train', download=True, transform=train_transform)

      # extra_data = torchvision.datasets.SVHN(
      #     root='C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/SVHN',
      #     split='extra', download=True, transform=transform_AutoAug_SVHN)

      test_data = torchvision.datasets.SVHN(
          root='C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/SVHN',
          split='test', download=True, transform=test_transform)

      # If only train_data is used,
      # print(len(train_data))
      # print(len(test_data))
      # train_loader = torch.utils.data.DataLoader(
      #     total_train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
      # )

      #If all data(train_data+extra_data) are used,
      total_train_data = torch.utils.data.ConcatDataset([train_data, extra_data])
      print(len(total_train_data))
      train_loader = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
      )

      val_loader = torch.utils.data.DataLoader(
          test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
      )

      test_loader = torch.utils.data.DataLoader(
          test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
      )
      #test_data = datasets.CIFAR100('C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR100', train=False, transform=test_transform, download=True)
      base_c_path = ''
      num_classes = 10
      #train_data = datasets.CIFAR100('C:/Users/User/anaconda3/envs/pytorch/Lib/site-packages/torchvision/datasets/CIFAR100', train=True, transform=train_transform, download=True)

  # Create model
  if args.model == 'densenet':
    net = densenet(num_classes=num_classes)
    net=net.to(device)
  elif args.model == 'wrn40_2':
    net = WideResNet(40, num_classes, 2, args.droprate)
    net=net.to(device)
  elif args.model == 'wrn28_10':
    net = WideResNet(28, num_classes, 10, args.droprate)
    net=net.to(device)
  elif args.model == 'allconv':
    net = AllConvNet(num_classes)
    net=net.to(device)
  elif args.model == 'resnext':
    net = resnext29(num_classes=num_classes)
    net=net.to(device)
  elif args.model == 'resnet50':
    net=models.resnet50(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num_classes)
    net = net.to(device)
  elif args.model == 'resnet101':
    net=models.resnet101(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num_classes)
    net = net.to(device)
  elif args.model == 'resnet152':
    net=models.resnet152(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num_classes)
    net = net.to(device)
  elif args.model == 'wide_resnet101_2':
    net=models.wide_resnet101_2(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num_classes)
    net = net.to(device)
  elif args.model == 'densenet201':
    net=models.densenet201(pretrained=False)
    num_ftrs = net.classifier.in_features #net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num_classes)
    net = net.to(device)
  elif args.model == 'resnext101_32x8d':
    net=models.resnext101_32x8d(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num_classes)
    net = net.to(device)
  elif args.model == 'vgg19':
    net = models.vgg19(pretrained=False, num_classes=num_classes)
    net.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    net = net.to(device)
  elif args.model == 'vgg19_bn':
    net=models.vgg19_bn(pretrained=False, num_classes=num_classes)
    net.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    net = net.to(device)

  print(net)

  from torchsummary import summary
  summary(net, input_size=(3, 32, 32))

  optimizer = torch.optim.SGD(
      net.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.decay,
      nesterov=True)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0

  if args.resume:
    if os.path.isfile(args.cc):
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch'] + 1
      best_acc = checkpoint['best_acc']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print('Model restored from epoch:', start_epoch)

  if args.evaluate:
    # Evaluate clean accuracy first because test_c mutates underlying data
    test_loss, test_acc = test(net, val_loader)
    print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
        test_loss, 100 - 100. * test_acc))

    test_c_acc = test_c(net, test_data, base_c_path)
    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
    return

  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
          step,
          args.epochs,
          1,  # lr_lambda computes multiplicative factor
          1e-6 / args.learning_rate))

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  global  log_path
  log_path = os.path.join(args.save,
                          args.dataset + '_' + args.model + '_training_log.csv')
  with open(log_path, 'w') as f:
    f.write('epoch,time(s),learning late,train_loss,validation_loss,validation_error(%)\n')

  best_acc = 0
  print('Beginning training from epoch:', start_epoch + 1)
  for epoch in range(start_epoch, args.epochs):
    begin_time = time.time()
    lr = scheduler.get_last_lr()[0]
    #lr = optimizer.param_groups[0]['lr']
    print("learning late : {0}".format(lr))
    train_loss_ema,train_acc = train(net, train_loader, optimizer, scheduler)
    scheduler.step() #원래는 train 함수 안에 있었음
    test_loss, test_acc = test(net, val_loader)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    checkpoint = {
        'epoch': epoch,
        'dataset': args.dataset,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(args.save, 'checkpoint.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

    with open(log_path, 'a') as f:
      f.write('%05d,%05d,%0.20f,%0.6f,%0.2f,%0.8f,%0.2f\n' % (
          (epoch + 1),
          time.time() - begin_time,
          lr,
          train_loss_ema,
          100-100*train_acc,
          test_loss,
          100 - 100. * test_acc,
      ))

    print(
        'Epoch {0:5d} | Time {1:5d} | Leaning late {2:20f} |Train Loss {3:.6f} |Train Error {4:.2f}| Validation Loss {5:.8f} |Validation Error {6:.2f}'
        .format((epoch + 1), int(time.time() - begin_time),lr, train_loss_ema, 100 - 100. * train_acc, test_loss*len(test_loader.dataset), 100 - 100. * test_acc))

  #Final results(for saving)
  test_loss2, test_acc2 = test2(net, test_loader)
  #test_loss, test_acc = test(net, val_loader)
  print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
      test_loss2, 100 - 100. * test_acc2))

  with open(log_path, 'a') as f:
      f.write('\nTest Loss, Test Error\n')
      f.write('%0.5f,%0.2f\n\n' % (test_loss2,100 - 100. * test_acc2))

  # 파라미터 저장용
  with open(log_path, 'a') as f:
      f.write('Dataset : %s\n' % (args.dataset))
      f.write('Epochs : %s\n' % (args.epochs))
      f.write('Learning rate : %s\n' % (args.learning_rate))
      f.write('Batch size : %s\n' % (args.batch_size))
      f.write('Weight decay : %s\n' % (args.decay))
  #Save model
  torch.save(net.state_dict(), "model.pth")
  print("Saved PyTorch Model State to model.pth")

  '''
  #Load model
  model = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
  model.to("cuda:0")
  model.load_state_dict(torch.load("model.pth"))
  model.eval()
  '''

  #If CIFAR-10-C or CIFAR-100-C datasets are used
  test_c_acc = test_c(net, test_data, base_c_path)
  print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

  with open(log_path, 'a') as f:
    f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
            (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))

if __name__ == '__main__':
  main()
