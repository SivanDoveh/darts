import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.SGD(self.model.arch_parameters(), lr=args.arch_learning_rate, weight_decay=args.arch_weight_decay)
    #self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        #lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data#W
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta #dl_train/dw +wd*w - they use autograd in order to not accumulate grads-they only wanr to calculate them
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))#w -eta*(moment+dl/dw) -> unrolled model = network on w'
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):#eta is lr of weights
    self.optimizer.zero_grad()
    self.compute_rank(self)
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)#network_optimizer = SGD for w
    else:
        self._backward_step(input_valid, target_valid)
    #prune is activated every time
    self.optimizer.step() #step of adam for alpha
    self.prune_alpha
  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def compute_rank(self):
    activation_index = len(self.activations) - self.grad_index - 1
    activation = self.activations[activation_index]
    values = \
      torch.sum((activation * grad), dim=0). \
        sum(dim=2).sum(dim=3)[0, :, 0, 0].data

    # Normalize the rank by the filter dimensions
    values = \
      values / (activation.size(0) * activation.size(2) * activation.size(3))

    if activation_index not in self.filter_ranks:
      self.filter_ranks[activation_index] = \
        torch.FloatTensor(activation.size(1)).zero_().cuda()

    self.filter_ranks[activation_index] += values
    self.grad_index += 1

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):#calculates update rule for alpha
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)# L_val  on model with w'

    unrolled_loss.backward()#
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]#dL_val/dalpha
    vector = [v.grad.data for v in unrolled_model.parameters()]#dL_val/dw'
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train) #this is eq.7. now we need to do l_val on w'/daplha - implicit_grads

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)# g is dL_val/dalpha, this line computes dL_val/dalpha - eta*implicit.. this is eq.6 for updating alpha

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)#here they put the grad so the optmizer will know what grads to take
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)#p is w ,R is epsilon, v is dl/dw' . now p is w+
    loss = self.model._loss(input, target) #loss_train on w+
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())#d loss_train on w+/ dalpha

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)#add 1 then sub 2
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())#d loss_train on w-/ dalpha

    for p, v in zip(self.model.parameters(), vector):#retuen p to be original model.parameters()
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

