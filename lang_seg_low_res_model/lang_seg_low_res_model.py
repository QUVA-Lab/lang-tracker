from __future__ import absolute_import, division, print_function

import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P

channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

###############################################################################
# Helper Methods
###############################################################################

def fc_sigmoid(bottom, nout, fix_param=False, finetune=False):
    if fix_param:
        mult = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
        fc = L.InnerProduct(bottom, num_output=nout, param=mult)
    else:
        if finetune:
            mult = [dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.2, decay_mult=0)]
            fc = L.InnerProduct(bottom, num_output=nout, param=mult)
        else:
            mult = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
            filler = dict(type='xavier')
            fc = L.InnerProduct(bottom, num_output=nout,
                                param=mult, weight_filler=filler)
    return fc, L.Sigmoid(fc, in_place=True)


def conv_relu(bottom, nout, ks=3, stride=1, pad=1, fix_param=False, finetune=False):
    if fix_param:
        mult = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, param=mult)
    else:
        if finetune:
            mult = [dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.2, decay_mult=0)]
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad, param=mult)
        else:
            mult = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
            filler = dict(type='xavier')
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad,
                                 param=mult, weight_filler=filler)
    return conv, L.ReLU(conv, in_place=True)


def conv(bottom, nout, ks=3, stride=1, pad=1, fix_param=False, finetune=False):
    if fix_param:
        mult = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, param=mult)
    else:
        if finetune:
            mult = [dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.2, decay_mult=0)]
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad, param=mult)
        else:
            mult = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
            filler = dict(type='xavier')
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad,
                                 param=mult, weight_filler=filler)
    return conv


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


################################################################################
# Model Generation
###############################################################################

def generate_model(split, config):
    n = caffe.NetSpec()
    batch_size = config.N
    mode_str = str(dict(split=split, batch_size=batch_size))
    n.language, n.cont, n.image, n.spatial, n.label = L.Python(module=config.data_provider,
                                                               layer=config.data_provider_layer,
                                                               param_str=mode_str,
                                                               ntop=5)

    # the base net (VGG-16)
    n.conv1_1, n.relu1_1 = conv_relu(n.image, 64,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    n.fcn_fc6, n.fcn_relu6 = conv_relu(n.pool5, 4096, ks=7, pad=3)
    if config.vgg_dropout:
        n.fcn_drop6 = L.Dropout(n.fcn_relu6, dropout_ratio=0.5, in_place=True)
        n.fcn_fc7, n.fcn_relu7 = conv_relu(n.fcn_drop6, 4096, ks=1, pad=0)
        n.fcn_drop7 = L.Dropout(n.fcn_relu7, dropout_ratio=0.5, in_place=True)
        n.fcn_fc8 = conv(n.fcn_drop7, 1000, ks=1, pad=0)
    else:
        n.fcn_fc7, n.fcn_relu7 = conv_relu(n.fcn_relu6, 4096, ks=1, pad=0)
        n.fcn_fc8 = conv(n.fcn_relu7, 1000, ks=1, pad=0)

    
    # embedding
    n.embed = L.Embed(n.language, input_dim=config.vocab_size,
                      num_output=config.embed_dim,
                      weight_filler=dict(type='uniform', min=-0.08, max=0.08))

    # LSTM
    n.lstm = L.LSTM(n.embed, n.cont,
                    recurrent_param=dict(num_output=config.lstm_dim,
                                         weight_filler=dict(type='uniform', min=-0.08, max=0.08),
                                         bias_filler=dict(type='constant', value=0)))
    tops = L.Slice(n.lstm, ntop=config.T, slice_param=dict(axis=0))
    for i in range(config.T - 1):
        n.__setattr__('slice'+str(i), tops[i])
        n.__setattr__('silence'+str(i), L.Silence(tops[i], ntop=0))
    n.lstm_out = tops[-1]
    n.lstm_feat = L.Reshape(n.lstm_out, reshape_param=dict(shape=dict(dim=[-1, config.lstm_dim])))

    # Dynamic conv filters
    n.dyn_l, n.dyn_sig = fc_sigmoid(n.lstm_feat, 1000+8)
    n.lstm_dyn_kernel = L.Reshape(n.dyn_sig, reshape_param=dict(shape=dict(dim=[-1, 1, config.lstm_dim+8, 1, 1])))

    # Tile LSTM feature
    #n.lstm_resh = L.Reshape(n.lstm_feat, reshape_param=dict(shape=dict(dim=[-1, config.lstm_dim, 1, 1])))
    #n.lstm_tile_1 = L.Tile(n.lstm_resh, axis=2, tiles=config.featmap_H)
    #n.lstm_tile_2 = L.Tile(n.lstm_tile_1, axis=3, tiles=config.featmap_W)

    # L2 Normalize image and language features
    #n.img_l2norm = L.L2Normalize(n.fcn_fc8)
    #n.lstm_l2norm = L.L2Normalize(n.lstm_tile_2)

    # Concatenate
    #n.feat_all = L.Concat(n.lstm_l2norm, n.img_l2norm, n.spatial, concat_param=dict(axis=1))
    n.feat_all = L.Concat(n.fcn_fc8, n.spatial, concat_param=dict(axis=1))

    # MLP Classifier over concatenated feature
    #n.fcn_l1, n.fcn_relu1 = conv_relu(n.feat_all, config.mlp_hidden_dims, ks=1, pad=0)
    #if config.mlp_dropout:
    #    n.fcn_drop1 = L.Dropout(n.fcn_relu1, dropout_ratio=0.5, in_place=True)
    #    n.fcn_scores = conv(n.fcn_drop1, 1, ks=1, pad=0)
    #else:
    #    n.fcn_scores = conv(n.fcn_relu1, 1, ks=1, pad=0)
    
    # Dyn conv layer
    n.fcn_scores = L.DynamicConvolution(n.feat_all, n.lstm_dyn_kernel,
                                        convolution_param=dict(num_output=1,
                                                               kernel_size=1,
                                                               stride=1,
                                                               pad=0,
                                                               bias_term=False))
    
    # Loss Layer
    n.loss = L.SigmoidCrossEntropyLoss(n.fcn_scores, n.label)

    return n.to_proto()


