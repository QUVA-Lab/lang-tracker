import caffe
import numpy as np
import os
import sys

import track_model_train as track_model
import train_config

max_iter = 1000
def eval_avg_scores(config):
    with open('./track_model/scores.prototxt', 'w') as f:
        f.write(str(track_model.generate_scores('', config)))

    caffe.set_device(config.gpu_id)
    caffe.set_mode_gpu()

    # Load pretrained model
    scores_net = caffe.Net('./track_model/scores.prototxt',
                           config.weights,
                           caffe.TEST)

    #import ipdb; ipdb.set_trace()
    scores = 0
    num_sample = 0
    for it in range(max_iter):
        scores_net.forward()
        scores_val = scores_net.blobs['fcn_scores'].data[...].copy()

        scores += scores_val.sum()
        num_sample += scores_val.size

    # ALOV conv345 -> 0.01196
    # OTB50  scores = 72313495.437500, samples = 1936000, avg_score = 37.364085 -> 0.02676
    # ILSVRC scores = 66083375.812500, samples = 1936000, avg_score = 34.133975 -> 0.02929
    avg_score = scores / num_sample
    print('\tscores = %f, samples = %d, avg_score = %f\t'
          % (scores, num_sample, avg_score))
if __name__ == '__main__':
    config = train_config.Config()
    eval_avg_scores(config)
