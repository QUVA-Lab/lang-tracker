from __future__ import absolute_import, division, print_function

import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P


channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

###############################################################################
# Helper Methods
###############################################################################

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, param_names=('conv_w', 'conv_b'), bias_term=True, fix_param=False, finetune=False):
    if fix_param:
        mult = [dict(name=param_names[0], lr_mult=0, decay_mult=0), dict(name=param_names[1], lr_mult=0, decay_mult=0)]
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, param=mult)
    else:
        if finetune:
            mult = [dict(name=param_names[0], lr_mult=0.1, decay_mult=1), dict(name=param_names[1], lr_mult=0.2, decay_mult=0)]
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad, param=mult)
        else:
            mult = [dict(name=param_names[0], lr_mult=1, decay_mult=1), dict(name=param_names[1], lr_mult=2, decay_mult=0)]
            filler = dict(type='xavier')
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad, bias_term=bias_term,
                                 param=mult, weight_filler=filler)
    return conv, L.ReLU(conv, in_place=True)


def conv(bottom, nout, ks=3, stride=1, pad=1, param_names=('conv_w', 'conv_b'), bias_term=True, fix_param=False, finetune=False):
    if fix_param:
        mult = [dict(name=param_names[0], lr_mult=0, decay_mult=0), dict(name=param_names[1], lr_mult=0, decay_mult=0)]
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, param=mult)
    else:
        if finetune:
            mult = [dict(name=param_names[0], lr_mult=0.1, decay_mult=1), dict(name=param_names[1], lr_mult=0.2, decay_mult=0)]
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad, param=mult)
        else:
            mult = [dict(name=param_names[0], lr_mult=1, decay_mult=1), dict(name=param_names[1], lr_mult=2, decay_mult=0)]
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

def generate_scores(split, config):
    n = caffe.NetSpec()
    dataset = config.dataset
    batch_size = config.N
    mode_str = str(dict(dataset=dataset, split=split, batch_size=batch_size))
    n.image1, n.image2, n.label, n.sample_weights, n.feat_crop = L.Python(module=config.data_provider,
                                                                          layer=config.data_provider_layer,
                                                                          param_str=mode_str,
                                                                          ntop=5)

    ################################
    # the base net (VGG-16) branch 1
    n.conv1_1, n.relu1_1 = conv_relu(n.image1, 64,
                                     param_names=('conv1_1_w', 'conv1_1_b'),
                                     fix_param=True,
                                     finetune=False)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64,
                                     param_names=('conv1_2_w', 'conv1_2_b'),
                                     fix_param=True,
                                     finetune=False)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128,
                                     param_names=('conv2_1_w', 'conv2_1_b'),
                                     fix_param=True,
                                     finetune=False)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128,
                                     param_names=('conv2_2_w', 'conv2_2_b'),
                                     fix_param=True,
                                     finetune=False)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256,
                                     param_names=('conv3_1_w', 'conv3_1_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256,
                                     param_names=('conv3_2_w', 'conv3_2_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256,
                                     param_names=('conv3_3_w', 'conv3_3_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    n.pool3 = max_pool(n.relu3_3)
    # spatial L2 norm
    n.pool3_lrn = L.LRN(n.pool3, local_size=513, alpha=513, beta=0.5, k=1e-16)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512,
                                     param_names=('conv4_1_w', 'conv4_1_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512,
                                     param_names=('conv4_2_w', 'conv4_2_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512,
                                     param_names=('conv4_3_w', 'conv4_3_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    # spatial L2 norm
    n.relu4_3_lrn = L.LRN(n.relu4_3, local_size=1025, alpha=1025, beta=0.5, k=1e-16)
    #n.pool4 = max_pool(n.relu4_3)

    #n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512,
    #                                 param_names=('conv5_1_w', 'conv5_1_b'),
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=config.finetune)
    #n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512,
    #                                 param_names=('conv5_2_w', 'conv5_2_b'),
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=config.finetune)
    #n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512,
    #                                 param_names=('conv5_3_w', 'conv5_3_b'),
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=config.finetune)
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
    #n.feat_all1 = n.relu4_3_lrn
    n.feat_all1 = L.Concat(n.pool3_lrn, n.relu4_3_lrn, concat_param=dict(axis=1))
    #n.feat_all1 = L.Concat(n.pool3_lrn, n.relu4_3_lrn, n.relu5_3_lrn, concat_param=dict(axis=1))
    n.feat_all1_crop = L.Crop(n.feat_all1, n.feat_crop, crop_param=dict(axis=2, offset=[config.query_featmap_H//3, config.query_featmap_W//3]))
    
    ################################
    # the base net (VGG-16) branch 2
    n.conv1_1_p, n.relu1_1_p = conv_relu(n.image2, 64,
                                         param_names=('conv1_1_w', 'conv1_1_b'),
                                         fix_param=True,
                                         finetune=False)
    n.conv1_2_p, n.relu1_2_p = conv_relu(n.relu1_1_p, 64,
                                         param_names=('conv1_2_w', 'conv1_2_b'),
                                         fix_param=True,
                                         finetune=False)
    n.pool1_p = max_pool(n.relu1_2_p)

    n.conv2_1_p, n.relu2_1_p = conv_relu(n.pool1_p, 128,
                                         param_names=('conv2_1_w', 'conv2_1_b'),
                                         fix_param=True,
                                         finetune=False)
    n.conv2_2_p, n.relu2_2_p = conv_relu(n.relu2_1_p, 128,
                                         param_names=('conv2_2_w', 'conv2_2_b'),
                                         fix_param=True,
                                         finetune=False)
    n.pool2_p = max_pool(n.relu2_2_p)

    n.conv3_1_p, n.relu3_1_p = conv_relu(n.pool2_p, 256,
                                         param_names=('conv3_1_w', 'conv3_1_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    n.conv3_2_p, n.relu3_2_p = conv_relu(n.relu3_1_p, 256,
                                         param_names=('conv3_2_w', 'conv3_2_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    n.conv3_3_p, n.relu3_3_p = conv_relu(n.relu3_2_p, 256,
                                         param_names=('conv3_3_w', 'conv3_3_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    n.pool3_p = max_pool(n.relu3_3_p)
    # spatial L2 norm
    n.pool3_lrn_p = L.LRN(n.pool3_p, local_size=513, alpha=513, beta=0.5, k=1e-16)

    n.conv4_1_p, n.relu4_1_p = conv_relu(n.pool3_p, 512,
                                         param_names=('conv4_1_w', 'conv4_1_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    n.conv4_2_p, n.relu4_2_p = conv_relu(n.relu4_1_p, 512,
                                         param_names=('conv4_2_w', 'conv4_2_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    n.conv4_3_p, n.relu4_3_p = conv_relu(n.relu4_2_p, 512,
                                         param_names=('conv4_3_w', 'conv4_3_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    # spatial L2 norm
    n.relu4_3_lrn_p = L.LRN(n.relu4_3_p, local_size=1025, alpha=1025, beta=0.5, k=1e-16)
    #n.pool4_p = max_pool(n.relu4_3_p)

    #n.conv5_1_p, n.relu5_1_p = conv_relu(n.pool4_p, 512,
    #                                     param_names=('conv5_1_w', 'conv5_1_b'),
    #                                     fix_param=config.fix_vgg,
    #                                     finetune=config.finetune)
    #n.conv5_2_p, n.relu5_2_p = conv_relu(n.relu5_1_p, 512,
    #                                     param_names=('conv5_2_w', 'conv5_2_b'),
    #                                     fix_param=config.fix_vgg,
    #                                     finetune=config.finetune)
    #n.conv5_3_p, n.relu5_3_p = conv_relu(n.relu5_2_p, 512,
    #                                     param_names=('conv5_3_w', 'conv5_3_b'),
    #                                     fix_param=config.fix_vgg,
    #                                     finetune=config.finetune)
    # upsampling feature map
    #n.relu5_3_upsampling_p = L.Deconvolution(n.relu5_3_p,
    #                                         convolution_param=dict(num_output=512,
    #                                                                group=512,
    #                                                                kernel_size=4,
    #                                                                stride=2,
    #                                                                pad=1,
    #                                                                bias_term=False,
    #                                                                weight_filler=dict(type='bilinear')),
    #                                         param=[dict(lr_mult=0, decay_mult=0)])
    # spatial L2 norm
    #n.relu5_3_lrn_p = L.LRN(n.relu5_3_upsampling_p, local_size=1025, alpha=1025, beta=0.5, k=1e-16)

    # concat all skip features
    #n.feat_all2 = n.relu4_3_lrn_p
    n.feat_all2 = L.Concat(n.pool3_lrn_p, n.relu4_3_lrn_p, concat_param=dict(axis=1))
    #n.feat_all2 = L.Concat(n.pool3_lrn_p, n.relu4_3_lrn_p, n.relu5_3_lrn_p, concat_param=dict(axis=1))

    # Dyn conv layer
    n.fcn_scores = L.DynamicConvolution(n.feat_all2, n.feat_all1_crop,
                                        convolution_param=dict(num_output=1,
                                                               kernel_size=11,
                                                               stride=1,
                                                               pad=5,
                                                               bias_term=False))
    return n.to_proto()

def generate_model(split, config):
    n = caffe.NetSpec()
    dataset = config.dataset
    batch_size = config.N
    mode_str = str(dict(dataset=dataset, split=split, batch_size=batch_size))
    n.image1, n.image2, n.label, n.sample_weights, n.feat_crop = L.Python(module=config.data_provider,
                                                                          layer=config.data_provider_layer,
                                                                          param_str=mode_str,
                                                                          ntop=5)

    ################################
    # the base net (VGG-16) branch 1
    n.conv1_1, n.relu1_1 = conv_relu(n.image1, 64,
                                     param_names=('conv1_1_w', 'conv1_1_b'),
                                     fix_param=True,
                                     finetune=False)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64,
                                     param_names=('conv1_2_w', 'conv1_2_b'),
                                     fix_param=True,
                                     finetune=False)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128,
                                     param_names=('conv2_1_w', 'conv2_1_b'),
                                     fix_param=True,
                                     finetune=False)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128,
                                     param_names=('conv2_2_w', 'conv2_2_b'),
                                     fix_param=True,
                                     finetune=False)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256,
                                     param_names=('conv3_1_w', 'conv3_1_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256,
                                     param_names=('conv3_2_w', 'conv3_2_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256,
                                     param_names=('conv3_3_w', 'conv3_3_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    n.pool3 = max_pool(n.relu3_3)
    # spatial L2 norm
    n.pool3_lrn = L.LRN(n.pool3, local_size=513, alpha=513, beta=0.5, k=1e-16)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512,
                                     param_names=('conv4_1_w', 'conv4_1_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512,
                                     param_names=('conv4_2_w', 'conv4_2_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512,
                                     param_names=('conv4_3_w', 'conv4_3_b'),
                                     fix_param=config.fix_vgg,
                                     finetune=config.finetune)
    # spatial L2 norm
    n.relu4_3_lrn = L.LRN(n.relu4_3, local_size=1025, alpha=1025, beta=0.5, k=1e-16)
    #n.pool4 = max_pool(n.relu4_3)

    #n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512,
    #                                 param_names=('conv5_1_w', 'conv5_1_b'),
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=config.finetune)
    #n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512,
    #                                 param_names=('conv5_2_w', 'conv5_2_b'),
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=config.finetune)
    #n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512,
    #                                 param_names=('conv5_3_w', 'conv5_3_b'),
    #                                 fix_param=config.fix_vgg,
    #                                 finetune=config.finetune)
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
    #n.feat_all1 = n.relu4_3_lrn
    n.feat_all1 = L.Concat(n.pool3_lrn, n.relu4_3_lrn, concat_param=dict(axis=1))
    #n.feat_all1 = L.Concat(n.pool3_lrn, n.relu4_3_lrn, n.relu5_3_lrn, concat_param=dict(axis=1))
    n.feat_all1_crop = L.Crop(n.feat_all1, n.feat_crop, crop_param=dict(axis=2, offset=[config.query_featmap_H//3, config.query_featmap_W//3]))

    ################################
    # the base net (VGG-16) branch 2
    n.conv1_1_p, n.relu1_1_p = conv_relu(n.image2, 64,
                                         param_names=('conv1_1_w', 'conv1_1_b'),
                                         fix_param=True,
                                         finetune=False)
    n.conv1_2_p, n.relu1_2_p = conv_relu(n.relu1_1_p, 64,
                                         param_names=('conv1_2_w', 'conv1_2_b'),
                                         fix_param=True,
                                         finetune=False)
    n.pool1_p = max_pool(n.relu1_2_p)

    n.conv2_1_p, n.relu2_1_p = conv_relu(n.pool1_p, 128,
                                         param_names=('conv2_1_w', 'conv2_1_b'),
                                         fix_param=True,
                                         finetune=False)
    n.conv2_2_p, n.relu2_2_p = conv_relu(n.relu2_1_p, 128,
                                         param_names=('conv2_2_w', 'conv2_2_b'),
                                         fix_param=True,
                                         finetune=False)
    n.pool2_p = max_pool(n.relu2_2_p)

    n.conv3_1_p, n.relu3_1_p = conv_relu(n.pool2_p, 256,
                                         param_names=('conv3_1_w', 'conv3_1_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    n.conv3_2_p, n.relu3_2_p = conv_relu(n.relu3_1_p, 256,
                                         param_names=('conv3_2_w', 'conv3_2_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    n.conv3_3_p, n.relu3_3_p = conv_relu(n.relu3_2_p, 256,
                                         param_names=('conv3_3_w', 'conv3_3_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    n.pool3_p = max_pool(n.relu3_3_p)
    # spatial L2 norm
    n.pool3_lrn_p = L.LRN(n.pool3_p, local_size=513, alpha=513, beta=0.5, k=1e-16)

    n.conv4_1_p, n.relu4_1_p = conv_relu(n.pool3_p, 512,
                                         param_names=('conv4_1_w', 'conv4_1_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    n.conv4_2_p, n.relu4_2_p = conv_relu(n.relu4_1_p, 512,
                                         param_names=('conv4_2_w', 'conv4_2_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    n.conv4_3_p, n.relu4_3_p = conv_relu(n.relu4_2_p, 512,
                                         param_names=('conv4_3_w', 'conv4_3_b'),
                                         fix_param=config.fix_vgg,
                                         finetune=config.finetune)
    # spatial L2 norm
    n.relu4_3_lrn_p = L.LRN(n.relu4_3_p, local_size=1025, alpha=1025, beta=0.5, k=1e-16)
    #n.pool4_p = max_pool(n.relu4_3_p)

    #n.conv5_1_p, n.relu5_1_p = conv_relu(n.pool4_p, 512,
    #                                     param_names=('conv5_1_w', 'conv5_1_b'),
    #                                     fix_param=config.fix_vgg,
    #                                     finetune=config.finetune)
    #n.conv5_2_p, n.relu5_2_p = conv_relu(n.relu5_1_p, 512,
    #                                     param_names=('conv5_2_w', 'conv5_2_b'),
    #                                     fix_param=config.fix_vgg,
    #                                     finetune=config.finetune)
    #n.conv5_3_p, n.relu5_3_p = conv_relu(n.relu5_2_p, 512,
    #                                     param_names=('conv5_3_w', 'conv5_3_b'),
    #                                     fix_param=config.fix_vgg,
    #                                     finetune=config.finetune)
    # upsampling feature map
    #n.relu5_3_upsampling_p = L.Deconvolution(n.relu5_3_p,
    #                                         convolution_param=dict(num_output=512,
    #                                                                group=512,
    #                                                                kernel_size=4,
    #                                                                stride=2,
    #                                                                pad=1,
    #                                                                bias_term=False,
    #                                                                weight_filler=dict(type='bilinear')),
    #                                         param=[dict(lr_mult=0, decay_mult=0)])
    # spatial L2 norm
    #n.relu5_3_lrn_p = L.LRN(n.relu5_3_upsampling_p, local_size=1025, alpha=1025, beta=0.5, k=1e-16)

    # concat all skip features
    #n.feat_all2 = n.relu4_3_lrn_p
    n.feat_all2 = L.Concat(n.pool3_lrn_p, n.relu4_3_lrn_p, concat_param=dict(axis=1))
    #n.feat_all2 = L.Concat(n.pool3_lrn_p, n.relu4_3_lrn_p, n.relu5_3_lrn_p, concat_param=dict(axis=1))

    # Dyn conv layer
    n.fcn_scores = L.DynamicConvolution(n.feat_all2, n.feat_all1_crop,
                                        convolution_param=dict(num_output=1,
                                                               kernel_size=11,
                                                               stride=1,
                                                               pad=5,
                                                               bias_term=False))
    
    # scale scores with zero mean 0.01196 -> 0.02677
    n.fcn_scaled_scores = L.Power(n.fcn_scores, power_param=dict(scale=0.01196,
                                                                 shift=-1.0,
                                                                 power=1))

    # Loss Layer
    n.loss = L.WeightedSigmoidCrossEntropyLoss(n.fcn_scaled_scores, n.label, n.sample_weights)

    return n.to_proto()


