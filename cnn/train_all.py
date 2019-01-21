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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network


class Args(object):
    dataset = 'cifar10'
    data = '../data'
    batch_size = 60
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4
    report_freq = 50
    gpu = ''
    epochs = 600
    init_channels = 36
    layers = 20
    model_path = 'saved_models'
    auxiliary = ''
    auxiliary_weight = 0.4
    cutout = ''
    drop_path_prob = 0.2
    save = ''
    seed = 0
    grad_clip = 5

    def __init__(self, gpu, auxiliary, cutout, save):
        # self.dataset = 'cifar10'
        # self.data = '../data'
        # self.batch_size = 60
        # self.learning_rate = 0.025
        # self.momentum = 0.9
        # self.weight_decay = 3e-4
        # self.report_freq = 50
        self.gpu = gpu
        # self.epochs = 600
        # self.init_channels = 36
        # self.layers = 20
        # self.model_path = 'saved_models'
        self.auxiliary = auxiliary
        # self.auxiliary_weight = 0.4
        self.cutout = cutout
        # self.drop_path_prob = 0.2
        self.save = save
        # self.seed = 0
        # self.grad_clip = 5


def train_phase(genotype_s, logging, args):


    def train(train_queue, model, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.train()

        for step, (input, target) in enumerate(train_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda(async=True)

            optimizer.zero_grad()
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += args.auxiliary_weight * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
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

            logits, _ = model(input)
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
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    in_channels, num_classes, dataset_in_torch ,stride_for_aux = utils.dataset_fields(args , train=False)  # new
    genotype = genotype_s#eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels,in_channels,stride_for_aux, num_classes, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    train_data,valid_data = utils.dataset_split_and_transform(dataset_in_torch, args, train=False)  # new
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    for epoch in range(args.epochs):
      scheduler.step()
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

      train_acc, train_obj = train(train_queue, model, criterion, optimizer)
      logging.info('train_acc %f', train_acc)

      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)

      utils.save(model, os.path.join(args.save, 'weights.pt'))






