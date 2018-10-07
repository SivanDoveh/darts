import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dset

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_dataset(args):#new

  assert args.dataset in set(['cifar10', 'fashion_mnist', 'SVHN']), 'unknown dataset %s' % dataset
  if args.dataset == 'cifar10':
      mean = (0.4914, 0.4822, 0.4465)
      std = (0.24705882352941178, 0.24352941176470588, 0.2615686274509804)
      image_size = 32
      cutout_length = 16

  elif args.dataset == 'fashion_mnist':
      mean = (0.28604060411453247,)
      std = (0.3530242443084717,)
      image_size = 28
      cutout_length = 16

  elif args.dataset == 'SVHN':
      mean = (0.43768218, 0.44376934, 0.47280428)
      std = (0.1980301, 0.2010157, 0.19703591)
      image_size = 32
      cutout_length = 20

  train_transform = transforms.Compose([
    transforms.RandomCrop(image_size, padding=4),# cifar and svhn are 32 , fashion-mnist is 28
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])
  return train_transform, valid_transform

def _data_transforms_cifar10(args):# not used
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
      
#new func
def dataset_fields(args,train=True):
    if args.dataset == 'cifar10':
        in_channels = 3
        num_classes = 10
        dataset_in_torch = dset.CIFAR10
        stride_for_aux = 3
    elif args.dataset == 'fashion_mnist':
        in_channels = 1
        num_classes = 10
        dataset_in_torch = dset.FashionMNIST
        stride_for_aux = 2
    elif args.dataset == 'SVHN':
        in_channels = 3
        num_classes = 10
        dataset_in_torch = dset.SVHN
        stride_for_aux = 3        
    if train == False:
       return in_channels,num_classes,dataset_in_torch,stride_for_aux     
    return in_channels,num_classes,dataset_in_torch

def dataset_split_and_transform(dataset_in_torch,args,train=True):
    train_transform, valid_transform =_data_transforms_dataset(args)    
    if args.dataset == 'SVHN':  # different fields for svhn
        train_data = dataset_in_torch(root=args.data, split='train', download=True, transform=train_transform)
        if train==False:
            valid_data = dataset_in_torch(root=args.data, split='test', download=True, transform=valid_transform)
            return train_data,valid_data
    else:
        train_data = dataset_in_torch(root=args.data, train=True, download=True, transform=train_transform)
        if train==False:
            valid_data = dataset_in_torch(root=args.data, train=False, download=True, transform=valid_transform)
            return train_data,valid_data
            
    return train_data

