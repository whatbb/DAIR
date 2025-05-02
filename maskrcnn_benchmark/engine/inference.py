# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.structures.image_list import to_image_list
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision import transforms as T




def compute_on_dataset(model,data_loader,device):
    model.eval()

    results_dict = {}
    cpu_device = torch.device("cpu")

    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch

        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        
        # tag_visualization(images,output)
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict



def tag_visualization(images,output):
    box_color = (251,210,106) 
    toPIL = transforms.ToPILImage()
    images = to_image_list(images).tensors
    images = toPIL(images[0])
    # cv_img = cv2.cvtColor(np.asarray(images), cv2.COLOR_RGB2BGR)
    cv_img = np.asarray(images)

    channel_mean = torch.tensor([102.9801, 115.9465, 122.7717])
    channel_std = torch.tensor([1., 1., 1.])

    MEAN = [-mean/std for mean, std in zip(channel_mean, channel_std)]
    STD = [1/std for std in channel_std]
    denormalizer = T.Normalize(mean=MEAN, std=STD)
    mean = channel_mean.type_as(images[0])
    tag_imgs = denormalizer(images[0])
    mean = [102.9801, 115.9465, 122.7717]
    tag_img = tag_imgs[0]
    for k in range(3):
        tag_img[k] = tag_img[k] + mean[k]
    tag_img = tag_img/255
    PIL_tag_imgs= T.ToPILImage()(tag_img)
    image_rgb = cv2.cvtColor(np.asarray(PIL_tag_imgs), cv2.COLOR_BGR2RGB)
    bboxs = output[0].bbox
    for j in range(len(bboxs)):
        cv2.rectangle(image_rgb, (int(bboxs[j][0]), int(bboxs[j][1])), (int(bboxs[j][2]), int(bboxs[j][3])), color=box_color, thickness=2)
        cv2.rectangle(image_rgb, (int(bboxs[j][0])+1, int(bboxs[j][1])+1), (int(bboxs[j][2])+1, int(bboxs[j][3])+1), color=box_color, thickness=2)
        cv2.rectangle(image_rgb, (int(bboxs[j][0])-1, int(bboxs[j][1])-1), (int(bboxs[j][2])-1, int(bboxs[j][3])-1), color=box_color, thickness=2)
        cv2.rectangle(image_rgb, (int(bboxs[j][0])-2, int(bboxs[j][1])-2), (int(bboxs[j][2])-2, int(bboxs[j][3])-2), color=box_color, thickness=2)
        cv2.rectangle(image_rgb, (int(bboxs[j][0])+2, int(bboxs[j][1])+2), (int(bboxs[j][2])+2, int(bboxs[j][3])+2), color=box_color, thickness=2)
    image_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    image_pil = Image.fromarray(image_rgb)
    image_pil.save('../models/images/label_ourmethod/{ii}.jpg'.format(ii=i))
    cv2.imwrite('../models/images/cu_ourmethod_1/{ii}.jpg'.format(ii=i), image_rgb)


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        model_2,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, model_2,data_loader,device)
    
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
