import torch
import torch.nn as nn
from torch.nn import functional as F

# DDILoss
class DDILoss(torch.nn.Module):

    def __init__(self, args):
        super(DDILoss, self).__init__()
        self.args = args

    def forward(self, y_prob, ddi_adj):
        # Calculate loss
        y_prob = y_prob.t() * y_prob
        loss = self.args.weight_ddi_loss * y_prob.mul(ddi_adj).sum()
        return loss

