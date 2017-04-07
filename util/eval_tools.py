from __future__ import absolute_import, division, print_function

import numpy as np
import pyximport; pyximport.install()
from util.nms import cpu_nms as nms

# all boxes are [xmin, ymin, xmax, ymax] format, 0-indexed, including xmax and ymax
def compute_bbox_iou(bboxes, target):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))

    if isinstance(target, list):
        target = np.array(target)
    target = target.reshape((-1, 4))

    A_bboxes = (bboxes[..., 2]-bboxes[..., 0]+1) * (bboxes[..., 3]-bboxes[..., 1]+1)
    A_target = (target[..., 2]-target[..., 0]+1) * (target[..., 3]-target[..., 1]+1)
    assert(np.all(A_bboxes >= 0))
    assert(np.all(A_target >= 0))
    I_x1 = np.maximum(bboxes[..., 0], target[..., 0])
    I_y1 = np.maximum(bboxes[..., 1], target[..., 1])
    I_x2 = np.minimum(bboxes[..., 2], target[..., 2])
    I_y2 = np.minimum(bboxes[..., 3], target[..., 3])
    A_I = np.maximum(I_x2 - I_x1 + 1, 0) * np.maximum(I_y2 - I_y1 + 1, 0)
    IoUs = A_I / (A_bboxes + A_target - A_I)
    assert(np.all(0 <= IoUs) and np.all(IoUs <= 1))
    return IoUs

# # all boxes are [num, height, width] binary array
def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U

def compute_bbox_max(bbox_file):
    with open(bbox_file) as f:
        for line in f:
            items = [int(x) for x in line.strip().split()]

    box1 = np.array(items[0::4]).T
    box2 = np.array(items[1::4]).T
    box3 = np.array(items[2::4]).T
    box4 = np.array(items[3::4]).T
    bboxes = np.array([box1, box2, box1+box3-1, box2+box4-1]).T

    col1 = np.min(np.array([bboxes[:,0], bboxes[:,2]]), axis=0)
    col2 = np.min(np.array([bboxes[:,1], bboxes[:,3]]), axis=0)
    col3 = np.max(np.array([bboxes[:,0], bboxes[:,2]]), axis=0)
    col4 = np.max(np.array([bboxes[:,1], bboxes[:,3]]), axis=0)
    bboxes = np.array([col1, col2, col3, col4]).T

    max_sz = 0
    max_box = bboxes[0, :]
    for i in range(bboxes.shape[0]): # for each bbox
        pred_box = bboxes[i, :]
        box_sz = (pred_box[2] - pred_box[0])*(pred_box[3] - pred_box[1])
        if box_sz > max_sz:
            max_sz = box_sz
            max_box = pred_box

    return max_box

