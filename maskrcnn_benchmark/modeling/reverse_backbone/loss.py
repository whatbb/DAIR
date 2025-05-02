from torch import nn
import torch
from maskrcnn_benchmark.layers import GradientScalarLayer
import torch.nn.functional as F


class RESLossComputation(object):
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.mse_loss = nn.MSELoss()
        # self.DaInv = DAInvHead(3)
        # self.inv_grl = GradientScalarLayer(-1*0.1)

    def __call__(self, imgs,ins_imgs):
        # inv = self.inv_grl(imgs)
        # inv = self.DaInv(imgs)
        # mask = torch.tensor([1,0], dtype=torch.float64)
        # inv_loss = F.binary_cross_entropy_with_logits(
        #     torch.squeeze(inv), mask.type(torch.cuda.FloatTensor)
        # )
        inv_loss = self.mse_loss(imgs, ins_imgs)
        return inv_loss

def make_inv_heads_loss_evaluator(cfg):
    loss_evaluator = RESLossComputation(cfg)
    return loss_evaluator