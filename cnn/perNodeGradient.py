import torch
import numpy as np


class Prune(object):

    class NodePrune(object):
        def __init__(self, node_num):
            # in order to not zero the 'zero' op
            zeros_indices_alphas_normal = list(range(0, 8*(2+node_num), 8))
            zeros_indices_alphas_reduce = list(range(0, 8*(2+node_num), 8))
            self.zeros_indices = [zeros_indices_alphas_normal, zeros_indices_alphas_reduce, ]
            self.sparsity = [0, 0]
            self.num_to_zero = [0, 0]
            self.alphas_size = 14 + 7 * node_num
            self.s_f_per_node = 1 - 2 / self.alphas_size
            self.zeroed_until_now = [0, 0]

        def prune_node(self, alphas, k, epoch, epochs_pre_prune, epochs):
            val, ind = alphas.data.sort()
            list_zeroed = self.zeros_indices[k]
            ind = [x for x in ind if x not in list_zeroed]
            self.num_to_zero_sparse(epoch, epochs_pre_prune, epochs, k)
            for i in ind[0:int(self.num_to_zero[k])]:
                list_zeroed.append(i)

        def num_to_zero_sparse(self, epoch, epochs_pre_prune, epochs, k):
            sparsity_prev = self.zeroed_until_now[k] / self.alphas_size
            self.sparsity[k] = self.s_f_per_node - self.s_f_per_node*(
                1 - (epoch + 1 - epochs_pre_prune) / (epochs - epochs_pre_prune)) ** 3
            self.num_to_zero[k] = np.floor((self.sparsity[k] - sparsity_prev) * self.alphas_size)

            if epoch == epochs - 1:
                self.num_to_zero[k] = self.alphas_size - 2 - self.zeroed_until_now[k]

            self.zeroed_until_now[k] = self.zeroed_until_now[k] + self.num_to_zero[k]

    def __init__(self):
        self.nodes = []
        for i in range(0, 4):
            self.nodes.append(self.NodePrune(i))

    @staticmethod
    def split_to_alpha_groups(alphas, alphas_split=[]):
        start = 0
        for i in range(0, 4):
            alphas_split.append(alphas[0, start:start + 8 * (i + 2)])
            start = start + 8 * (i + 2)
        return alphas_split

    def all_zero_indices_from_all_nodes(self, k, list_zeroed=[]):
        start = 0
        for i in range(0, 4):
            list_of_node = self.nodes[i].zeros_indices[k]
            list_of_node = [x+start for x in list_of_node]
            list_zeroed.extend(list_of_node)
            start = start + (i + 2) * 8

        return list_zeroed

    def prune_alphas_step(self, arch_parameters, dalpha, dalpha_normalized_params, epoch, epochs_pre_prune, epochs):
        for k in range(0, 2):

            dalpha_normalized = dalpha_normalized_params[k]
            alphas = arch_parameters[k]
            dalphas = dalpha[k]

            zeroed = 0

            dalpha_normalized.data.resize_(1, 112)
            alphas.data.resize_(1, 112)
            dalphas.data.resize_(1, 112)

            alphas_split = self.split_to_alpha_groups(dalpha_normalized)

            for i in range(0, 4):
                self.nodes[i].prune_node(alphas_split[i], k, epoch, epochs_pre_prune, epochs)
                zeroed = zeroed + self.nodes[i].zeroed_until_now[k]

            ind_all_alphas = self.all_zero_indices_from_all_nodes(k)
            ind_all_alphas = [x for x in ind_all_alphas if x not in list(range(0, 112, 8))]

            for i in ind_all_alphas:
                alphas.data[0, i] = -1000
                dalphas.data[0, i] = 0

                #dalpha_normalized_params.data.resize_(14, 8)
            alphas.data.resize_(14, 8)
            dalphas.data.resize_(14, 8)

            print(str(zeroed))