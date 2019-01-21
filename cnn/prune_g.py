import torch
import numpy as np


class Prune(object):

    def __init__(self, num_to_zero):
        zeros_indices_alphas_normal = list(range(0, 112, 8))  # in order to not zero the 'zero' op
        zeros_indices_alphas_reduce = list(range(0, 112, 8))
        self.zeros_indices = [zeros_indices_alphas_normal, zeros_indices_alphas_reduce, ]
        self.num_to_zero = num_to_zero
        self.zeroed_until_now = [0, 0]


    def prune_alphas_step(self, arch_parameters, dalpha, dalpha_normalized_params):
        for k in range(0, 2):

            dalpha_normalized = dalpha_normalized_params[k]
            alphas = arch_parameters[k]
            dalphas = dalpha[k]
            self.prune_alphas(self, alphas, dalphas, dalpha_normalized, k)

    def prune_alphas(self, alphas, dalphas, dalpha_normalized, k):

        dalpha_normalized.data.resize_(1, 112)
        alphas.data.resize_(1, 112)
        dalphas.data.resize_(1, 112)

        val, ind = dalpha_normalized.data.sort()
        list_zeroed = self.zeros_indices[k]

        ind = [x for x in ind[0] if x not in list_zeroed[k]]

        for i in ind[0:self.num_to_zero]:
            list_zeroed.append(i)
            alphas.data[0, i] = -1000
            dalphas.data[0, i] = 0

        alphas.data.resize_(14, 8)
        dalphas.data.resize_(14, 8)
        #dalpha_normalized.data.resize_(14, 8)


