# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
from .picodet import PicoDet


@DETECTORS.register_module()
class PPYOLOE(PicoDet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch

    def _preprocess(self, img, gt_bboxes):
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = F.interpolate(
                img,
                size=self._input_size,
                # mode='bilinear',
                mode=self.get_mode(),
                # align_corners=False
                )
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
        return img, gt_bboxes

    def init_weights(self):
        return super().init_weights()

