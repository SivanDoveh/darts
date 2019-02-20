import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)#inside op there are all the operations 
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):#x is feature map (node) and weights are the weight for every operation and the mixed_op is alpha_i,j 
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):#if reduction=true- then cell will be reduction

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, se):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.se = se

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps#number of nodes inside
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):#nodes in cell- the get info from j
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)#list of operations per i,j. its instance of  aclass that can be applied after to feature map
        self._ops.append(op)

    ####SElayer
    if se:
        self.seLayer = Seq_Ex_Block(C*multiplier)



  def forward(self, s0, s1, weights):#cell gets as input the previos outputs. weights is arch_weights:
    s0 = self.preprocess0(s0)#the 1x1 conv on the last inputs.s0 is feature map
    s1 = self.preprocess1(s1)
    
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
        s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))#here we get the mixed_op value on feature_maps 0 and 1 
        offset += len(states)
        states.append(s)
    cat = torch.cat(states[-self._multiplier:], dim=1)

    if self.se:
        cat = self.seLayer(cat)
    return cat

class Seq_Ex_Block(nn.Module):
  def __init__(self, in_ch, r=16):
    super(Seq_Ex_Block, self).__init__()
    self.se = nn.Sequential(
      GlobalAvgPool(),
      nn.Linear(in_ch, in_ch // r),
      nn.ReLU(inplace=True),
      nn.Linear(in_ch // r, in_ch),
      nn.Sigmoid()
      )

  def forward(self, x):
    se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
    # print(f'x:{x.sum()}, x_se:{x.mul(se_weight).sum()}')
    return x.mul(se_weight)

class GlobalAvgPool(nn.Module):
  def __init__(self):
    super(GlobalAvgPool, self).__init__()

  def forward(self, x):
    return x.view(*(x.shape[:-2]), -1).mean(-1)

# class SELayer(nn.Module):
#   def __init__(self, channel, reduction=16):
#     super(SELayer, self).__init__()
#     self.avg_pool = nn.AdaptiveAvgPool2d(1)
#     self.fc = nn.Sequential(
#       nn.Linear(channel, channel // reduction, bias=False),
#       nn.ReLU(inplace=True),
#       nn.Linear(channel // reduction, channel, bias=False),
#       nn.Sigmoid()
#     )
#
#   def forward(self, x):
#     b, c, _, _ = x.size()
#     y = self.avg_pool(x).view(b, c)
#     y = self.fc(y).view(b, c, 1, 1)
#     return x * y.expand_as(x)


class Network(nn.Module):

  def __init__(self, C, in_channels, num_classes, layers, criterion,se, steps=4, multiplier=4, stem_multiplier=3):#in channels
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.in_channels = in_channels
    self.se = se

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, se)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self.in_channels ,self._num_classes, self._layers, self._criterion, self.se).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    #h = self.alphas_normal.register_hook(self.prune)#
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [ #in init we create alpha in class
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):#function to get back alpha- reduction and normal
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
