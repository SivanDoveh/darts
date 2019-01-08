import torch
import numpy as np


class Prune(object):

    def __init__(self, num_to_zero):
        self.zeros_indices_alphas_normal = list(range(0, 112, 8))  # in order to not zero the 'zero' op
        self.zeros_indices_alphas_reduce = list(range(0, 112, 8))
        self.num_to_zero = num_to_zero

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


