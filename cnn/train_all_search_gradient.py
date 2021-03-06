import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network

from architect_gp import Architect
#from architect_all_alpha import Architect

import torchvision.transforms as transforms


class Args(object):
    dataset = 'cifar10'
    data = '../data' # '/data/datasets/automl-pytorch/'
    batch_size = 64
    learning_rate = 0.025
    learning_rate_min = 0.001
    momentum = 0.9
    weight_decay = 3e-4
    report_freq = 50
    gpu = ''
    epochs = ''
    init_channels = 16
    #layers = 8
    model_path = 'saved_models'
    cutout = False
    drop_path_prob = 0.3
    save = ''
    seed = 0
    grad_clip = 5
    train_portion = 0.5
    unrolled = ''
    arch_learning_rate = 3e-4
    arch_weight_decay = 1e-3
    epochs_pre_prune = ''
    steps_accum = ''
    exponent = ''

    def __init__(self, gpu, unrolled, epochs_pre_prune, steps_accum, save, exponent, epochs, layers, se, beta):
        self.gpu = gpu
        self.save = save
        self.unrolled = unrolled
        self.epochs_pre_prune = epochs_pre_prune
        self.steps_accum = steps_accum
        self.exponent = exponent
        self.epochs = epochs
        self.layers = layers
        self.SE = se
        self.Beta = beta

def search_phase(logging,args):

    def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(train_queue):  # input,target is for w step
            model.train()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))  # input_search,target_search is for alpha step
            input_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(target_search, requires_grad=False).cuda(async=True)

            if args.Beta == 'abs':
                beta = 1
            if args.Beta == 'gradient':
                beta = 0
            if args.Beta == 'gradient_to_abs':
                beta = (epoch-args.epochs_pre_prune)/(args.epochs-args.epochs_pre_prune)

            prune_args = {'step': step, 'epoch': epoch, 'epochs_pre_prune': args.epochs_pre_prune, 'epochs': args.epochs,
                          'steps_accum': args.steps_accum, 'logging': logging, 'exponent': args.exponent, 'beta': beta}
            # architect.step(input, target, input_search, target_search, lr, optimizer, step, epoch, args.epochs_pre_prune, args.epochs, args.steps_accum, logging,
            #                unrolled=args.unrolled,)  # update alpha
            architect.step(input, target, input_search, target_search, lr, optimizer, prune_args, unrolled=args.unrolled, )
            # update alpha

            # during the arch.step the optimization for alpha happen
            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)

            if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


    def infer(valid_queue, model, criterion):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()

        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda(async=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    in_channels, num_classes, dataset_in_torch = utils.dataset_fields(args)  # new
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, in_channels, num_classes, args.layers, criterion, args.SE)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(  # SGD for weights
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_data = utils.dataset_split_and_transform(dataset_in_torch, args)  # new
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)

        logging.info('train_acc %f', train_acc)

        if epoch == args.epochs-1:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        logging.info(F.softmax(model.alphas_normal, dim=-1))
        logging.info(F.softmax(model.alphas_reduce, dim=-1))

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    return genotype

