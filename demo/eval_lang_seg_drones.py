# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import sys
import skimage.io
import numpy as np
import caffe
import json
import timeit
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#get_ipython().magic(u'matplotlib inline')
sys.path.append('../')
sys.path.append('../lang_seg_model')

from scipy.special import expit
from lang_seg_model import lang_seg_model as segmodel
from util import processing_tools, im_processing, text_processing, eval_tools
import otb100_config

from glob import glob, iglob


####################################################
def sigmoid(x):
    #return 1/(1+np.exp(-x))
    return expit(x)

def resize_and_pad(im, input_h, input_w):
    # Resize and pad im to input_h x input_w size
    im_h, im_w = im.shape[:2]
    scale = min(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    pad_h = int(np.floor(input_h - resized_h) / 2)
    pad_w = int(np.floor(input_w - resized_w) / 2)

    resized_im = skimage.transform.resize(im, [resized_h, resized_w])
    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
    new_im[pad_h:pad_h+resized_h, pad_w:pad_w+resized_w, ...] = resized_im
    
    return new_im, scale, pad_h, pad_w

def crop_and_resize(im, crop_h, crop_w, scale):
    # Crop and resize im
    im_h, im_w = im.shape[:2]
    cropped_h = im_h - 2*crop_h
    cropped_w = im_w - 2*crop_w
    if im.ndim > 2:
        cropped_im = np.zeros((cropped_h, cropped_w, im.shape[2]), dtype=im.dtype)
    else:
        cropped_im = np.zeros((cropped_h, cropped_w), dtype=im.dtype)

    cropped_im[...] = im[crop_h:im_h-crop_h, crop_w:im_w-crop_w, ...]

    resized_h = int(np.round(cropped_h * scale))
    resized_w = int(np.round(cropped_w * scale))
    resized_im = skimage.transform.resize(im, [resized_h, resized_w])

    return resized_im

####################################################   
# Load config
config = otb100_config.Config()

# Load the model
with open('./lang_seg_model.prototxt', 'w') as f:
    f.write(str(segmodel.generate_model('val', config)))

caffe.set_device(config.gpu_id)
caffe.set_mode_gpu()

net = caffe.Net('./lang_seg_model.prototxt', config.pretrained_model, caffe.TEST)

# Load vocabulary
vocab_dict = text_processing.load_vocab_dict_from_file(config.vocab_file)

####################################################
videofiles = sorted(glob('/home/zhenyang/Workspace/data/drones/frames/*'))
for videofile in videofiles:
    video = videofile.split('/')[-1]
    print(video)

    start_frame_id = 1

    # First, select query
    query = 'man in black on the left'

    frames = sorted(glob('/home/zhenyang/Workspace/data/drones/frames/'+video+'/*.jpg'))
    num_frames = len(frames)
    counter = 0
    results = np.zeros((num_frames, 4), np.int)
    
    for fi in range(start_frame_id, num_frames+start_frame_id):
        im_file = '/home/zhenyang/Workspace/data/drones/frames/' + video + '/%05d.jpg' % (fi,)      
        
        ###############################
        # Run on the input image and query text
        text_seq_val = np.zeros((config.T, config.N), dtype=np.float32)
        imcrop_val = np.zeros((config.N, config.input_H, config.input_W, 3), dtype=np.float32)

        # Preprocess image and text
        im = skimage.io.imread(im_file)
        #im_, pad_h, pad_w, scale = resize_and_pad(im, config.input_H, config.input_W)
        #processed_im = skimage.img_as_ubyte(im_)
        processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, config.input_H, config.input_W))
        if processed_im.ndim == 2:
            processed_im = np.tile(processed_im[:, :, np.newaxis], (1, 1, 3))
        imcrop_val[0, :] = processed_im.astype(np.float32) - segmodel.channel_mean
        imcrop_val = imcrop_val.transpose((0, 3, 1, 2))
        imcrop_val = imcrop_val[:, ::-1, :, :]

        spatial_val = processing_tools.generate_spatial_batch(config.N, config.featmap_H, config.featmap_W)
        spatial_val = spatial_val.transpose((0, 3, 1, 2))

        text_seq_val[:, 0] = text_processing.preprocess_sentence(query, vocab_dict, config.T)
        cont_val = text_processing.create_cont(text_seq_val)

        dummy_label = np.zeros((config.N, 1, config.input_H, config.input_W), dtype=np.float32)

        # Forward pass to get response map
        net.blobs['language'].data[...] = text_seq_val
        net.blobs['cont'].data[...] = cont_val
        net.blobs['image'].data[...] = imcrop_val
        net.blobs['spatial'].data[...] = spatial_val
        net.blobs['label'].data[...] = dummy_label

        net.forward()

        upscores = net.blobs['upscores'].data[...].copy()
        upscores = np.squeeze(upscores)

        # Final prediction
        upscores = sigmoid(upscores)
        #print( str(np.amax(upscores)) )
        score_thresh = np.amax(upscores) * 0.5
        prediction = im_processing.resize_and_crop(upscores>score_thresh, *im.shape[:2]).astype(np.bool)
        #print( str(np.sum(prediction)) )

        # save the results
        if not os.path.exists('/home/zhenyang/Workspace/data/drones/results/results_lang_seg_mask/'+video):
            os.makedirs('/home/zhenyang/Workspace/data/drones/results/results_lang_seg_mask/'+video)
        filename1 = '/home/zhenyang/Workspace/data/drones/results/results_lang_seg_mask/'+video+'/%05d.jpg' % (fi,)
        plt.imsave(filename1, np.array(prediction), cmap=cm.gray)

        if not os.path.exists('/home/zhenyang/Workspace/data/drones/results/results_lang_seg_bbox/'+video):
            os.makedirs('/home/zhenyang/Workspace/data/drones/results/results_lang_seg_bbox/'+video)
        filename2 = '/home/zhenyang/Workspace/data/drones/results/results_lang_seg_bbox/'+video+'/%05d.txt' % (fi,)
        bbox_threshold = [20, 100, 110]
        param_thershold = str(bbox_threshold[0])+' '+str(bbox_threshold[1])+' '+str(bbox_threshold[2])
        os.system("bboxgenerator/./dt_box " + filename1+' '+param_thershold+' '+filename2)

        prediction_box = np.copy(im)
        bbox = eval_tools.compute_bbox_max(filename2)
        cv2.rectangle(prediction_box, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        filename3 = '/home/zhenyang/Workspace/data/drones/results/results_lang_seg_bbox/'+video+'/%05d.jpg' % (fi,)
        plt.imsave(filename3, prediction_box)

