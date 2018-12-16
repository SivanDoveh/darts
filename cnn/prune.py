import torch
import numpy as np


class Prune(object):

    def __init__(self, epochs_pre_prune):
        self.zeros_indices_alphas_normal = list(range(0, 112, 8))  # in order to not zero the 'zero' op
        self.zeros_indices_alphas_reduce = list(range(0, 112, 8))
        self.num_to_zero = 90//(49-epochs_pre_prune)
        self.sparsity = 0

    def prune_alphas_step(self, model):
        self.prune_alphas(model.alphas_normal, False)
        self.prune_alphas(model.alphas_reduce, True)

    def prune_alphas(self, alphas, reduce):
        alphas.data.resize_(1, 112)
        val, ind = alphas.data.sort()
        if reduce:
            list_zeroed = self.zeros_indices_alphas_reduce
        else:
            list_zeroed = self.zeros_indices_alphas_normal

        ind = [x for x in ind[0] if x not in list_zeroed]

        for i in ind[0:self.num_to_zero]:
            list_zeroed.append(i)
            alphas.data[0, i] = -1000

        alphas.data.resize_(14, 8)

    def num_to_zero_sparse(self, epoch, args):
        sparsity_prev = self.sparsity
        self.sparsity = args.s_f - args.s_f(
            1 - (epoch - args.epochs_pre_prune) / (args.epochs - 1 - args.epochs_pre_prune)) ** 3
        self.num_to_zero = np.floor((self.sparsity - sparsity_prev) * 98)