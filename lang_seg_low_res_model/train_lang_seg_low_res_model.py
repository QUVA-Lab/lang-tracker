import caffe
import numpy as np
import os
import sys

import lang_seg_low_res_model as segmodel
from util.processing_tools import *
import train_config


def train(config):
    with open('./lang_seg_low_res_model/proto_train.prototxt', 'w') as f:
        f.write(str(segmodel.generate_model('train', config)))

    caffe.set_device(config.gpu_id)
    caffe.set_mode_gpu()

    solver = caffe.get_solver('./lang_seg_low_res_model/solver.prototxt')
    if config.weights is not None:
        solver.net.copy_from(config.weights)

    cls_loss_avg = 0.0
    avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0.0, 0.0, 0.0
    decay = 0.99

    for it in range(config.max_iter):
        solver.step(1)

        cls_loss_val = solver.net.blobs['loss'].data
        scores_val = solver.net.blobs['fcn_scores'].data.copy()
        label_val = solver.net.blobs['label'].data.copy()

        cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val
        print('\titer = %d, cls_loss (cur) = %f, cls_loss (avg) = %f'
            % (it, cls_loss_val, cls_loss_avg))

        # Accuracy
        accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(scores_val, label_val)
        avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
        avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
        avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg
        print('\titer = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
              % (it, accuracy_all, accuracy_pos, accuracy_neg))
        print('\titer = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
              % (it, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

if __name__ == '__main__':
    config = train_config.Config()
    train(config)
