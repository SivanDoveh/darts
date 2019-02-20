import torch
import numpy as np
import torch.nn.functional as F

class Prune(object):

    class NodePrune(object):
        def __init__(self, node_num):
            self.node_num = node_num
            # in order to not zero the 'zero' op
            zeros_indices_alphas_normal = list(range(0, 8*(2+node_num), 8))
            zeros_indices_alphas_reduce = list(range(0, 8*(2+node_num), 8))
            self.zeros_indices = [zeros_indices_alphas_normal, zeros_indices_alphas_reduce, ]
            self.sparsity = [0, 0]
            self.num_to_zero = [0, 0]
            self.alphas_size_per_node = 14 + 7 * node_num
            self.s_f_per_node = 1 - 2 / self.alphas_size_per_node
            self.zeroed_until_now = [0, 0]  # without zero ops
            self.zeroed_alpha_i_j = [[0]*(2+node_num), [0]*(2+node_num)]

        def last_zero(self, k):
            counter_last_non_zero = self.zeroed_alpha_i_j[k].count(6)
            counter_all_zeroed = self.zeroed_alpha_i_j[k].count(7)
            list_to_no_zero = []
            if counter_all_zeroed == self.node_num and counter_last_non_zero == self.node_num+1:
                ind = self.zeroed_alpha_i_j[k].index(6)
                start = ind*8 + 8*sum(range(2, self.node_num + 2))
                list_to_no_zero = list(range(start, start+8))

            return list_to_no_zero

        def prune_node(self, alphas, k, prune_args):
            # if prune_args['epochs_pre_prune'] < 50:
            #     val, ind = alphas.data.sort()
            #
            # else:
            #     val, ind = alphas.data.sort(descending=True)
            #     print('reverse')
            val, ind = alphas.data.sort()
            list_zeroed = self.zeros_indices[k]

            self.num_to_zero_sparse(prune_args, k)
            if self.num_to_zero[k] != 0:
                list_to_no_zero = self.last_zero(k)
                ind = [x for x in ind if x not in list_zeroed or list_to_no_zero]

                for i in ind[0:int(self.num_to_zero[k])]:
                        list_zeroed.append(i)
                        self.zeroed_alpha_i_j[k][i//8] += 1

        def num_to_zero_sparse(self, prune_args, k):
            sparsity_prev = self.zeroed_until_now[k] / self.alphas_size_per_node
            self.sparsity[k] = self.s_f_per_node - self.s_f_per_node*(
                1 - (prune_args['epoch'] + 1 - prune_args['epochs_pre_prune']) / (prune_args['epochs'] - prune_args['epochs_pre_prune'])) ** prune_args['exponent']
            self.num_to_zero[k] = np.floor((self.sparsity[k] - sparsity_prev) * self.alphas_size_per_node)

            if prune_args['epoch'] == prune_args['epochs'] - 1:
                self.num_to_zero[k] = self.alphas_size_per_node - 2 - self.zeroed_until_now[k]

            self.zeroed_until_now[k] = self.zeroed_until_now[k] + self.num_to_zero[k]

    def __init__(self, epochs_pre_prune):
        self.counter = [epochs_pre_prune*112, epochs_pre_prune*112]
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

    def prune_alphas_step(self, arch_parameters, dalpha, dalpha_normalized_params, prune_args):
        print('beta ' + str(prune_args['beta']))
        for k in range(0, 2):

            ranking_values = dalpha_normalized_params[k]*(1-prune_args['beta'])+torch.abs(F.softmax(arch_parameters[k], dim=-1))*prune_args['beta']
            alphas = arch_parameters[k]
            dalphas = dalpha[k]

            zeroed = 0

            ranking_values.data.resize_(1, 112)
            alphas.data.resize_(1, 112)
            dalphas.data.resize_(1, 112)

            ranking_split = self.split_to_alpha_groups(ranking_values)

            for i in range(0, 4):
                self.nodes[i].prune_node(ranking_split[i], k, prune_args)
                zeroed = zeroed + self.nodes[i].zeroed_until_now[k]

            ind_all_alphas = self.all_zero_indices_from_all_nodes(k)
            ind_all_alphas = [x for x in ind_all_alphas if x not in list(range(0, 112, 8))]

            for i in ind_all_alphas:
                alphas.data[0, i] = -1000
                dalphas.data[0, i] = 0

                #dalpha_normalized_params.data.resize_(14, 8)
            alphas.data.resize_(14, 8)
            dalphas.data.resize_(14, 8)

            self.counter[k] = self.counter[k] + 112-zeroed
            if k == 1:
                prune_args['logging'].info('zeroed %f', zeroed)
                prune_args['logging'].info('counter %f', ((self.counter[k]+(112-zeroed)*(prune_args['epochs'] - 1 - prune_args['epoch'])) / (112 * 50)))
