import caffe
import numpy as np
import os
import sys

import track_model_train as track_model
import train_config

def compute_accuracy(scores, labels):
    is_pos = (labels != 0)
    is_neg = np.logical_not(is_pos)
    num_all = labels.shape[0]
    num_pos = np.sum(is_pos)
    num_neg = num_all - num_pos

    is_correct = np.logical_xor(scores < 0, is_pos)
    accuracy_all = np.sum(is_correct) / float(num_all)
    accuracy_pos = np.sum(is_correct[is_pos]) / float(num_pos)
    accuracy_neg = np.sum(is_correct[is_neg]) / float(num_neg)
    return accuracy_all, accuracy_pos, accuracy_neg


def train(config):
    with open('./track_model/proto_train.prototxt', 'w') as f:
        f.write(str(track_model.generate_model('train', config)))

    caffe.set_device(config.gpu_id)
    caffe.set_mode_gpu()

    solver = caffe.get_solver('./track_model/solver_sgd.prototxt')
    solver.net.copy_from(config.weights)
    #solver.net.save('./snapshots/track_model/_iter_0.caffemodel')
    #solver.restore('./snapshots/track_model/_iter_50000.solverstate')

    loss_avg = 0.0
    avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
    decay = 0.99

    for it in range(config.max_iter):
        solver.step(1)

        loss_val = solver.net.blobs['loss'].data
        #scores_val = solver.net.blobs['fcn_scaled_scores'].data.copy()
        #label_val = solver.net.blobs['label'].data.copy()

        #import ipdb; ipdb.set_trace()

        loss_avg = decay*loss_avg + (1-decay)*loss_val
        if it % config.iter_display == 0:
            print('\titer = %d, cls_loss (cur) = %f, cls_loss (avg) = %f'
                % (it, loss_val, loss_avg))

        # Accuracy
        #accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(scores_val, label_val)
        #avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
        #avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
        #avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg
        #print('\titer = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
        #      % (it, accuracy_all, accuracy_pos, accuracy_neg))
        #print('\titer = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
        #      % (it, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

if __name__ == '__main__':
    config = train_config.Config()
    train(config)
