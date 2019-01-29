import numpy as np

#sivan
class Prune(object):
    class NodePrune(object):

        def __init__(self, node_num):
            self.sparsity = [0, 0]
            self.num_to_zero = [0, 0]
            self.alphas_size_per_node = 14 + 7 * node_num
            self.s_f_per_node = 1 - 2 / self.alphas_size_per_node
            self.zeroed_until_now = [0, 0]  # without zero ops

        def prune_node(self, k, prune_args):
            self.num_to_zero_sparse(prune_args, k)

        def num_to_zero_sparse(self, prune_args, k):
            sparsity_prev = self.zeroed_until_now[k] / self.alphas_size_per_node
            self.sparsity[k] = self.s_f_per_node - self.s_f_per_node * (
                    1 - (prune_args['epoch'] + 1 - prune_args['epochs_pre_prune']) / (
                        prune_args['epochs'] - prune_args['epochs_pre_prune'])) ** 1.4
            self.num_to_zero[k] = np.floor((self.sparsity[k] - sparsity_prev) * self.alphas_size_per_node)

            if prune_args['epoch'] == prune_args['epochs'] - 1:
                self.num_to_zero[k] = self.alphas_size_per_node - 2 - self.zeroed_until_now[k]

            self.zeroed_until_now[k] = self.zeroed_until_now[k] + self.num_to_zero[k]

    def __init__(self, epochs_pre_prune):
        self.counter = [epochs_pre_prune*112, epochs_pre_prune*112]
        self.nodes = []
        for i in range(0, 4):
            self.nodes.append(self.NodePrune(i))

    def prune_alphas_step(self, prune_args):
        for k in range(0, 1):

            zeroed = 0
            for i in range(0, 4):
                self.nodes[i].prune_node(k, prune_args)
                zeroed = zeroed + self.nodes[i].zeroed_until_now[k]

            self.counter[k] = self.counter[k] + 112 - zeroed
            print('zeroed ', zeroed)
            print('counter ', (
                    (self.counter[k] + (112 - zeroed) * (prune_args['epochs'] - 1 - prune_args['epoch'])) / (112 * 50)))


pre_prune = 30
prune = Prune(pre_prune)
epochs = 67

for epoch in range(0, epochs):
    if epoch >= pre_prune:
        print('epoch ', epoch)
        prune_args = {'epoch': epoch, 'epochs_pre_prune': pre_prune, 'epochs': epochs}
        prune.prune_alphas_step(prune_args)