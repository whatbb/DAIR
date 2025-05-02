# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..da_heads.da_heads import build_da_heads
from ..reverse_backbone.self_Inversion_backbone import build_inv_heads
from ..reverse_backbone.self_Inversion_backbone import ReverseModule
from maskrcnn_benchmark.layers import GradientScalarLayer
# from sklearn.manifold import TSNE



class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.da_heads = build_da_heads(cfg)
        self.inv_heads = build_inv_heads(cfg)
        self.few_shot = cfg.MODEL.FEW_SHOT
        self.ratios = cfg.MODEL.FEW_SHOT.RATIOS

    

    def forward(self, images, targets=None, iteration=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        oldfeatures = self.backbone(images.tensors)

        da_features = oldfeatures 
        features = [oldfeatures[-1]]
        
            
        proposals, proposal_losses = self.rpn(images, features, targets)
        da_losses = {}
        
        if self.roi_heads:
            x, result, detector_losses, da_ins_feas, da_ins_labels = self.roi_heads(features, proposals, targets)
            
            if self.da_heads and self.training:
                if  not self.few_shot:
                    da_losses = self.da_heads(da_features, da_ins_feas, da_ins_labels,targets)
                else:
                    if iteration%self.ratios == 0:
                        da_losses = self.da_heads(da_features, da_ins_feas, da_ins_labels,targets)
        
        
        # else:
        #     # RPN-only models don't have roi_heads
        #     x = features
        #     result = proposals
        #     detector_losses = {}
        
        if self.inv_heads:
            inv_loss = self.inv_heads(images.tensors,da_features)
   

        # print(detector_losses[""])
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(da_losses)
            losses.update(inv_loss)
            return losses

        return result
