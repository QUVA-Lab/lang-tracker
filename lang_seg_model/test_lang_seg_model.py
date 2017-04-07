from __future__ import absolute_import, division, print_function

import sys
import os;
import skimage.io
import numpy as np
import caffe
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lang_seg_model as segmodel
import test_config

from util import processing_tools, im_processing, text_processing, eval_tools
from util.io import load_referit_gt_mask as load_gt_mask

################################################################################
# Evaluation network
################################################################################

def inference(config):
    with open('./lang_seg_model/test.prototxt', 'w') as f:
        f.write(str(segmodel.generate_model('val', config)))

    caffe.set_device(config.gpu_id)
    caffe.set_mode_gpu()

    # Load pretrained model
    net = caffe.Net('./lang_seg_model/test.prototxt',
                    config.pretrained_model,
                    caffe.TEST)

    ################################################################################
    # Load annotations and bounding box proposals
    ################################################################################

    query_dict = json.load(open(config.query_file))
    bbox_dict = json.load(open(config.bbox_file))
    imcrop_dict = json.load(open(config.imcrop_file))
    imsize_dict = json.load(open(config.imsize_file))
    imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
    vocab_dict = text_processing.load_vocab_dict_from_file(config.vocab_file)

    ################################################################################
    # Flatten the annotations
    ################################################################################

    flat_query_dict = {imname: [] for imname in imlist}
    for imname in imlist:
        this_imcrop_names = imcrop_dict[imname]
        for imcrop_name in this_imcrop_names:
            gt_bbox = bbox_dict[imcrop_name]
            if imcrop_name not in query_dict:
                continue
            this_descriptions = query_dict[imcrop_name]
            for description in this_descriptions:
                flat_query_dict[imname].append((imcrop_name, gt_bbox, description))

    ################################################################################
    # Testing
    ################################################################################

    cum_I, cum_U = 0.0, 0.0
    eval_seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0.0

    # Pre-allocate arrays
    imcrop_val = np.zeros((config.N, config.input_H, config.input_W, 3), dtype=np.float32)
    text_seq_val = np.zeros((config.T, config.N), dtype=np.int32)

    num_im = len(imlist)
    for n_im in tqdm(range(num_im)):
        imname = imlist[n_im]

        # Extract visual features from all proposals
        im = skimage.io.imread(config.image_dir + imname)
        processed_im = skimage.img_as_ubyte(
            im_processing.resize_and_pad(im, config.input_H, config.input_W))
                                                                         
        if processed_im.ndim == 2:
            processed_im = np.tile(processed_im[:, :, np.newaxis], (1, 1, 3))

        imcrop_val[...] = processed_im.astype(np.float32) - segmodel.channel_mean
        imcrop_val_trans = imcrop_val.transpose((0, 3, 1, 2))
        imcrop_val_trans = imcrop_val_trans[:, ::-1, :, :]

        # Extract spatial features
        spatial_val = processing_tools.generate_spatial_batch(config.N,
                                                              config.featmap_H,
                                                              config.featmap_W)
        spatial_val = spatial_val.transpose((0, 3, 1, 2))

        if not os.path.exists('./results/referit/results_lang_seg_model/'+imname[:-4]):
            os.makedirs('./results/referit/results_lang_seg_model/'+imname[:-4])
        fp = open('./results/referit/results_lang_seg_model/'+imname[:-4]+'/query.txt', 'w')
        fq = open('./results/referit/results_lang_seg_model/'+imname[:-4]+'/IoU.txt', 'w')
        for imcrop_name, _, description in flat_query_dict[imname]:
            mask = load_gt_mask(config.mask_dir + imcrop_name + '.mat').astype(np.float32)
            labels = (mask > 0)
            processed_labels = im_processing.resize_and_pad(mask, config.input_H, config.input_W)
            processed_labels = processed_labels > 0

            text_seq_val[:, 0] = text_processing.preprocess_sentence(description, vocab_dict, config.T)
            cont_val = text_processing.create_cont(text_seq_val)

            net.blobs['language'].data[...] = text_seq_val
            net.blobs['cont'].data[...] = cont_val
            net.blobs['image'].data[...] = imcrop_val_trans
            net.blobs['spatial'].data[...] = spatial_val
            net.blobs['label'].data[...] = processed_labels

            #import ipdb; ipdb.set_trace()
            net.forward()
            upscores = net.blobs['upscores'].data[...].copy()
            upscores = np.squeeze(upscores)

            # Evaluate the segmentation performance of using bounding box segmentation
            pred_raw = (upscores >= config.score_thresh).astype(np.float32)
            predicts = im_processing.resize_and_crop(pred_raw, im.shape[0], im.shape[1])
            I, U = eval_tools.compute_mask_IU(predicts, labels)
            cum_I += I
            cum_U += U
            this_IoU = I/float(U)
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (I/float(U) >= eval_seg_iou)
            seg_total += 1

            # save the results
            filename = './results/referit/results_lang_seg_model/'+imname[:-4]+'/%s.jpg' % (imcrop_name,)
            plt.imsave(filename, np.array(predicts.astype(np.bool)), cmap=cm.gray)
            fp.write(description.encode('ascii','ignore')+'\n')
            fq.write(str(this_IoU)+'\n')

        fp.close()
        fq.close()

    # Print results
    print('Final results on the whole test set')
    result_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        result_str += 'precision@%s = %f\n' % \
            (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou]/seg_total)
    result_str += 'overall IoU = %f\n' % (cum_I/cum_U)
    print(result_str)

if __name__ == '__main__':
    config = test_config.Config()
    inference(config)
