from __future__ import absolute_import, division, print_function

import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P

channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

###############################################################################
# Helper Methods
###############################################################################

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, bias_term=True, fix_param=False, finetune=False):
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
                                 num_output=nout, pad=pad, bias_term=bias_term,
                                 param=mult, weight_filler=filler)
    return conv, L.ReLU(conv, in_place=True)


def conv(bottom, nout, ks=3, stride=1, pad=1, bias_term=True, fix_param=False, finetune=False):
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
                                 num_output=nout, pad=pad, bias_term=bias_term,
                                 param=mult, weight_filler=filler)
    return conv


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


################################################################################
# Model Generation
###############################################################################

def generate_conv_features(split, config):
    n = caffe.NetSpec()
    dataset = config.dataset
    batch_size = config.N
    mode_str = str(dict(dataset=dataset, split=split, batch_size=batch_size))
    n.image, n.label = L.Python(module=config.data_provider,
                                layer=config.data_provider_layer_1,
                                param_str=mode_str,
                                ntop=2)

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
    # spatial L2 norm
    n.pool3_lrn = L.LRN(n.pool3, local_size=513, alpha=513, beta=0.5, k=1e-16)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    # spatial L2 norm
    n.relu4_3_lrn = L.LRN(n.relu4_3, local_size=1025, alpha=1025, beta=0.5, k=1e-16)
    #n.pool4 = max_pool(n.relu4_3)

    #n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512,
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=(not config.fix_vgg))
    #n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512,
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=(not config.fix_vgg))
    #n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512,
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=(not config.fix_vgg))
    # upsampling feature map
    #n.relu5_3_upsampling = L.Deconvolution(n.relu5_3,
    #                                       convolution_param=dict(num_output=512,
    #                                                              group=512,
    #                                                              kernel_size=4,
    #                                                              stride=2,
    #                                                              pad=1,
    #                                                              bias_term=False,
    #                                                              weight_filler=dict(type='bilinear')),
    #                                       param=[dict(lr_mult=0, decay_mult=0)])
    # spatial L2 norm
    #n.relu5_3_lrn = L.LRN(n.relu5_3_upsampling, local_size=1025, alpha=1025, beta=0.5, k=1e-16)

    # concat all skip features
    #n.feat_all = n.relu4_3_lrn
    n.feat_all = L.Concat(n.pool3_lrn, n.relu4_3_lrn, concat_param=dict(axis=1))
    #n.feat_all = L.Concat(n.pool3_lrn, n.relu4_3_lrn, n.relu5_3_lrn, concat_param=dict(axis=1))

    return n.to_proto()

def generate_conv_scores(split, config):
    n = caffe.NetSpec()
    dataset = config.dataset
    batch_size = config.N
    mode_str = str(dict(dataset=dataset, split=split, batch_size=batch_size))
    n.image, n.label = L.Python(module=config.data_provider,
                                layer=config.data_provider_layer_2,
                                param_str=mode_str,
                                ntop=2)

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
    # spatial L2 norm
    n.pool3_lrn = L.LRN(n.pool3, local_size=513, alpha=513, beta=0.5, k=1e-16)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512,
                                     fix_param=config.fix_vgg,
                                     finetune=(not config.fix_vgg))
    # spatial L2 norm
    n.relu4_3_lrn = L.LRN(n.relu4_3, local_size=1025, alpha=1025, beta=0.5, k=1e-16)
    #n.pool4 = max_pool(n.relu4_3)

    #n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512,
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=(not config.fix_vgg))
    #n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512,
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=(not config.fix_vgg))
    #n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512,
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=(not config.fix_vgg))
    # upsampling feature map
    #n.relu5_3_upsampling = L.Deconvolution(n.relu5_3,
    #                                       convolution_param=dict(num_output=512,
    #                                                              group=512,
    #                                                              kernel_size=4,
    #                                                              stride=2,
    #                                                              pad=1,
    #                                                              bias_term=False,
    #                                                              weight_filler=dict(type='bilinear')),
    #                                       param=[dict(lr_mult=0, decay_mult=0)])
    # spatial L2 norm
    #n.relu5_3_lrn = L.LRN(n.relu5_3_upsampling, local_size=1025, alpha=1025, beta=0.5, k=1e-16)

    # concat all skip features
    #n.feat_all = n.relu4_3_lrn
    n.feat_all = L.Concat(n.pool3_lrn, n.relu4_3_lrn, concat_param=dict(axis=1))
    #n.feat_all = L.Concat(n.pool3_lrn, n.relu4_3_lrn, n.relu5_3_lrn, concat_param=dict(axis=1))

    # dyn conv / implement as normal conv during test
    n.scores = conv(n.feat_all, 1,
                    ks=11, stride=1, pad=5,
                    bias_term=False,
                    fix_param=config.fix_vgg,
                    finetune=(not config.fix_vgg))

    return n.to_proto()



