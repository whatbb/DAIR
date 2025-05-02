# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                #proposals 中由2000多个下采样到256个
                proposals = self.loss_evaluator.subsample(proposals, targets)
                
        

        

        # [BoxList(num_boxes=256, image_width=1200, image_height=600, mode=xyxy), BoxList(num_boxes=256, image_width=1200, image_height=600, mode=xyxy)]
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        

        # x [512,2048,7,7]
        # final classifier that converts the features into predictions
        # class_logits [512, 9]
        # box_regression [512, 36]
        class_logits, box_regression = self.predictor(x)
        

        if not self.training:
            
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}, x, None
        
        # testpython tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes.yaml" 

        
        
        #tag_box_labels = test_result((class_logits[255:-1], box_regression[255:-1]), proposals[1])
        
        
        loss_classifier, loss_box_reg, da_ins_labels = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        if self.training:
            with torch.no_grad():
                da_proposals = self.loss_evaluator.subsample_for_da(proposals, targets)

        da_ins_feas = self.feature_extractor(features, da_proposals)

        

        class_logits, box_regression  = self.predictor(da_ins_feas)

        
        _, _, da_ins_labels = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            da_ins_feas,
            da_ins_labels,
        )
def getbox(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

def test_result(x, box):
    class_logits, box_regression = x
    class_prob = F.softmax(class_logits, -1)
    image_shapes = box.size
    boxes_per_image = len(box)
    concat_boxes = box.bbox

    num_classes = class_prob.shape[1]
    box_coder = BoxCoder(weights=(10., 10., 5., 5.))
    proposals = box_coder.decode(
            box_regression.view(boxes_per_image, -1), concat_boxes
        )

    #proposals = proposals.split(boxes_per_image, dim=0)
    #class_prob = class_prob.split(boxes_per_image, dim=0)

    results = []
    #for prob, boxes_per_img, image_shape in zip(
        #class_prob, proposals, image_shapes
    #):
    boxlist = prepare_boxlist(proposals, class_prob, image_shapes)
    boxlist = boxlist.clip_to_image(remove_empty=False)
    boxlist = filter_results(boxlist, num_classes)
    
    return boxlist


def prepare_boxlist(boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

def filter_results(boxlist, num_classes):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).
    """
    # unwrap the boxlist to avoid additional overhead.
    # if we had multi-class NMS, we could perform this directly on the boxlist
    boxes = boxlist.bbox.reshape(-1, num_classes * 4)
    scores = boxlist.get_field("scores").reshape(-1, num_classes)            

    device = scores.device
    result = []
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    inds_all = scores > 0.7
    for j in range(1, num_classes):
        inds = inds_all[:, j].nonzero().squeeze(1)
        # scores_j boxes_j 取分数大于0.05的框
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
        boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
        boxlist_for_class.add_field("scores", scores_j)
        boxlist_for_class = boxlist_nms(
            boxlist_for_class, 0.5
        )
        num_labels = len(boxlist_for_class)
        boxlist_for_class.add_field(
            "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
        )
        result.append(boxlist_for_class)

    result = cat_boxlist(result)
    number_of_detections = len(result)

    # Limit to max_per_image detections **over all classes**
    if number_of_detections > 100 > 0:
        cls_scores = result.get_field("scores")
        image_thresh, _ = torch.kthvalue(
            cls_scores.cpu(), number_of_detections - 100 + 1
        )
        keep = cls_scores >= image_thresh.item()
        keep = torch.nonzero(keep).squeeze(1)
        result = result[keep]
    return result


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
