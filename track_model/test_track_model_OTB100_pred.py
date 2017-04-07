from __future__ import absolute_import, division, print_function

import sys
import os
import skimage.io
import numpy as np
import caffe
import json
import time
from glob import glob, iglob
from tqdm import tqdm
import StringIO

import track_model_test as track_model
import test_pred_config

from util import processing_tools, im_processing, text_processing, eval_tools


# setup dataset:
def make_data_dict(data_dir,pred_dir):
    d = dict()
    videos = sorted(glob(data_dir+'*'))
    for i, video in enumerate(videos):
        video_nr = i
        video_name = video.split('/')[-1]
        video_file = ''.join([data_dir, video_name])
        gt_box_file = ''.join([data_dir, video_name, '/groundtruth_rect.txt'])
        pred_box_file = ''.join([pred_dir, video_name, '_groundtruth_rect.txt'])
        d[video_nr] = (video_name, video_file, gt_box_file, pred_box_file)
    return d

def load_video_metadata(video_name, gt_box_file, pred_box_file):
    try:
        gt_boxes = np.loadtxt(gt_box_file, delimiter=',').reshape((-1, 4))
    except ValueError:
        gt_boxes = np.loadtxt(gt_box_file).reshape((-1, 4))
    try:
        pred_boxes = np.loadtxt(pred_box_file, delimiter=',').reshape((-1, 4))
    except ValueError:
        pred_boxes = np.loadtxt(pred_box_file).reshape((-1, 4))

    start_frame_id = 1
    if video_name == 'David':
        start_frame_id = 300
    elif video_name == 'Tiger1':
        start_frame_id = 6
        gt_boxes = gt_boxes[6-1:, :]
    elif video_name == 'BlurCar1':
        start_frame_id = 247
    elif video_name == 'BlurCar3':
        start_frame_id = 3
    elif video_name == 'BlurCar4':
        start_frame_id = 18
    return start_frame_id, gt_boxes, pred_boxes

################################################################################
# Evaluation network
################################################################################

def inference(config):
    with open('./track_model/conv_features.prototxt', 'w') as f:
        f.write(str(track_model.generate_conv_features('val', config)))
    with open('./track_model/conv_scores.prototxt', 'w') as f:
        f.write(str(track_model.generate_conv_scores('val', config)))

    caffe.set_device(config.gpu_id)
    caffe.set_mode_gpu()

    # Load pretrained model
    conv_features_net = caffe.Net('./track_model/conv_features.prototxt',
                                  config.pretrained_model,
                                  caffe.TEST)
    conv_scores_net = caffe.Net('./track_model/conv_scores.prototxt',
                           config.pretrained_model,
                           caffe.TEST)

    #import ipdb; ipdb.set_trace()
    ################################################################################
    # Load image and bounding box annotations
    ################################################################################
    data_dict = make_data_dict(config.data_dir, config.pred_dir)
    total = len(data_dict)
    print('In total %d videos to evaluate.'%total)
    dummy_label = np.zeros((config.N, 1))

    for video_nr in range(total):
        video_name, video_file, gt_box_file, pred_box_file = data_dict[video_nr]

        if video_name == 'ClifBar':
            print('Skip this video %s.'%(video_name,))
            continue
        
        start_frame_id, gt_boxes, pred_boxes = load_video_metadata(video_name, gt_box_file, pred_box_file)
        num_frames = gt_boxes.shape[0]
        print('Start processing video %s (%d frames).'%(video_name,num_frames))
        
        # query box
        init_box = pred_boxes[0,:].copy()
        init_box_w = init_box[2]
        init_box_h = init_box[3]
        init_box[2] = init_box[2] + init_box[0] - 1
        init_box[3] = init_box[3] + init_box[1] - 1

        # caffe.io.load_image(imagefile)*255.0
        #qimg = skimage.io.imread('%s/img/%04d.jpg' % (video_file, start_frame_id))
        if video_name == 'Board':
            qimg = skimage.io.imread('%s/img/%05d.jpg' % (video_file, start_frame_id))
        else:
            qimg = skimage.io.imread('%s/img/%04d.jpg' % (video_file, start_frame_id))

        if qimg.ndim == 2:
            qimg = np.tile(qimg[:, :, np.newaxis], (1, 1, 3))
        qimg_height, qimg_width = qimg.shape[:2]

        # select the triple larger box to include some context
        qbox = init_box.copy()
        qbox[0] = qbox[0] - 1.0*init_box_w
        qbox[1] = qbox[1] - 1.0*init_box_h
        qbox[2] = qbox[2] + 1.0*init_box_w
        qbox[3] = qbox[3] + 1.0*init_box_h
        qbox = np.round(qbox).astype(int).reshape((-1, 4))

        # extract query box features
        inputs = np.zeros((config.N, config.query_H, config.query_W, 3), dtype=np.float32)
        inputs[0, ...] = im_processing.crop_and_pad_bboxes_subtract_mean(
                            qimg, qbox, config.qimage_size*3, track_model.channel_mean)
        inputs_trans = inputs.transpose((0, 3, 1, 2))
        inputs_trans = inputs_trans[:, ::-1, :, :]
        conv_features_net.blobs['image'].data[...] = inputs_trans
        conv_features_net.blobs['label'].data[...] = dummy_label
        conv_features_net.forward()
        conv_features = conv_features_net.blobs['feat_all'].data[...].copy()

        # crop feature map
        qfeat = conv_features[0, ...].copy()
        qfeat_crop = im_processing.crop_featmap_from_center(qfeat, 3.0)
        qfeat_crop_resh = qfeat_crop.reshape((-1, qfeat_crop.shape[0], qfeat_crop.shape[1], qfeat_crop.shape[2]))

        # set up dyn filters
        #print('params nr %d'%(len(conv_scores_net.params['scores']), ))
        conv_scores_net.params['scores'][0].data[...] = qfeat_crop_resh

        ################################################################################
        # Start tracking target on each frame
        ################################################################################
        results = np.zeros((num_frames, 5))
        results[0, 0] = 1
        results[0, 1:] = init_box

        center_x = np.ceil((init_box[2]-init_box[0]+1)/2.0) + init_box[0] - 1
        center_y = np.ceil((init_box[3]-init_box[1]+1)/2.0) + init_box[1] - 1

        sz_times = config.sz_times
        sample_w = init_box_w
        sample_h = init_box_h
        
        prev_scale = 1
        counter = 1
        start_time = time.time()
        for ii in tqdm(range(start_frame_id+1, num_frames+start_frame_id)):
            #img = skimage.io.imread('%s/img/%04d.jpg' % (video_file, ii))
            if video_name == 'Board':
                img = skimage.io.imread('%s/img/%05d.jpg' % (video_file, ii))
            else:
                img = skimage.io.imread('%s/img/%04d.jpg' % (video_file, ii))
            if img.ndim == 2:
                img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
            img_height, img_width = img.shape[:2]

            # assemble box proposals with multiple scales
            boxes = np.zeros((config.scales.size, 4))
            for ss, scale in enumerate(config.scales):
                boxes[ss, 0] = center_x - 0.5*sz_times*scale*sample_w + 1
                boxes[ss, 1] = center_y - 0.5*sz_times*scale*sample_h + 1
                boxes[ss, 2] = center_x + 0.5*sz_times*scale*sample_w
                boxes[ss, 3] = center_y + 0.5*sz_times*scale*sample_h
            boxes = np.round(boxes).astype(int).reshape((-1, 4))
            num_boxes = boxes.shape[0]

            # extract query box features
            inputs = np.zeros((config.N, config.input_H, config.input_W, 3), dtype=np.float32)
            inputs[:num_boxes, ...] = im_processing.crop_and_pad_bboxes_subtract_mean(
                                         img, boxes, config.timage_size, track_model.channel_mean)
            inputs_trans = inputs.transpose((0, 3, 1, 2))
            inputs_trans = inputs_trans[:, ::-1, :, :]
            conv_scores_net.blobs['image'].data[...] = inputs_trans
            conv_scores_net.blobs['label'].data[...] = dummy_label
            conv_scores_net.forward()
            scores_val = conv_scores_net.blobs['scores'].data[...].copy()
            scores_val = scores_val[:num_boxes, ...]

            # obtain sizes
            map_h = scores_val.shape[2]
            map_w = scores_val.shape[3]
            map_size = map_h * map_w

            # scale change penalty
            scores_val = np.multiply(scores_val, config.scale_penalty)
            scores_val = scores_val.reshape(-1)

            max_idx = np.argmax(scores_val)+1
            max_score = scores_val[max_idx-1]
            s_idx = np.ceil(float(max_idx)/map_size)
            max_idx_within = np.fmod(max_idx,map_size)
            r_idx = 0
            c_idx = 0
            if max_idx_within == 0:
                r_idx = map_h
                c_idx = map_w
            else:
                r_idx = np.ceil(float(max_idx_within)/map_w)
                c_idx = np.fmod(max_idx_within,map_w)
                if c_idx == 0:
                    c_idx = map_w

            # obtain box prediction
            bbox = boxes[int(s_idx-1), :].copy()
            predict_box = bbox.copy()
            predict_box[0] = np.maximum(bbox[0] + (c_idx-1-5) * config.spatial_ratio / config.timage_size * (bbox[2]-bbox[0]+1), 1)
            predict_box[1] = np.maximum(bbox[1] + (r_idx-1-5) * config.spatial_ratio / config.timage_size * (bbox[3]-bbox[1]+1), 1)
            predict_box[2] = np.minimum(predict_box[0] + sample_w * config.scales[int(s_idx-1)] - 1, img_width) # Be careful when extended to multiple scales
            predict_box[3] = np.minimum(predict_box[1] + sample_h * config.scales[int(s_idx-1)] - 1, img_height)

            # record result
            results[counter, 0] = max_score
            results[counter, 1:] = predict_box

            # update center coordinates
            center_x = np.ceil((predict_box[2]-predict_box[0]+1)/2.0) + predict_box[0] - 1
            center_y = np.ceil((predict_box[3]-predict_box[1]+1)/2.0) + predict_box[1] - 1
            prev_scale = prev_scale*(1-config.scaleLP) + config.scales[int(s_idx-1)]*config.scaleLP

            sample_w = sample_w * prev_scale 
            sample_h = sample_h * prev_scale

            counter = counter + 1

        elapsed_time = time.time() - start_time
        print('[%d] %s done in %f seconds. [%f fps]'%(video_nr+1, video_name, elapsed_time, num_frames/elapsed_time))

        # save results to file
        filename = config.result_dir+video_name+'_'+config.signature+'.txt'
        if video_name == 'Tiger1':
            filename = config.result_dir+video_name+'_refined_'+config.signature+'.txt'
        fp = open(filename, 'w')
        for jj in range(num_frames):
            fp.write('%f %d %d %d %d\n'%(results[jj,0],int(results[jj,1]),int(results[jj,2]),int(results[jj,3]),int(results[jj,4])))
        fp.close()

    print('Finish evlaluation on the whole test set.')


if __name__ == '__main__':
    config = test_pred_config.Config()
    inference(config)

