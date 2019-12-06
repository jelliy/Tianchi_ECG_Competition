# -*- coding: utf-8 -*-
'''
@time: 2019/9/12 15:16

@ author: javis
'''
import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import pickle

def get_result(y, p):
    print(confusion_matrix(y, p))
    print(accuracy_score(y, p))
    print(f1_score(y, p))

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

#计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)

def calc_f1_all(y_true, y_pre):
    y_true = y_true.cpu().detach().numpy()
    y_pre = y_pre.cpu().detach().numpy() > 0.5
    for i in range(34):
        print(str(i)+"--------")
        get_result(y_true[:,i],y_pre[:,i])
#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.sigmod = nn.Sigmoid()
        self.cerition = nn.BCELoss(reduction='none')

    def clip_by_tensor(self, t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        t = t.float()
        t_min = torch.tensor(t_min, dtype=torch.float).to(device)

        t_max = torch.tensor(t_max, dtype=torch.float).to(device)

        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def forward(self, y_pred, y_true):
        epsilon = 1.e-7

        y_pred = self.sigmod(y_pred)

        #y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        #y_pred = self.clip_by_tensor(y_pred, epsilon, 1. - epsilon)
        alpha_weight_factor = (y_true * self.alpha + (1 - y_true) * (1 - self.alpha))

        p_t = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))

        #cross_entropy_loss = -torch.log(p_t)
        cross_entropy_loss = self.cerition(y_pred, y_true)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * cross_entropy_loss)

        return focal_cross_entropy_loss.mean()

"""
def forward(self, y_true, y_pred):
    cross_entropy_loss = torch.nn.BCELoss(y_true, y_pred)
    p_t = ((y_true * y_pred) +
           ((1 - y_true) * (1 - y_pred)))
    modulating_factor = 1.0
    if self._gamma:
        modulating_factor = torch.pow(1.0 - p_t, self._gamma)
    alpha_weight_factor = 1.0
    if self._alpha is not None:
        alpha_weight_factor = (y_true * self._alpha +
                               (1 - y_true) * (1 - self._alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                cross_entropy_loss)
    return focal_cross_entropy_loss.mean()
"""

