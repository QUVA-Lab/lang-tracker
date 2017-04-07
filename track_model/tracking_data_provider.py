from __future__ import absolute_import, division, print_function

import caffe
import numpy as np
import os
import skimage.io
import threading
import Queue as queue
import ast

import train_config
import test_config
import track_model_train as track_model
from glob import glob, iglob

config = train_config.Config()
test_config = test_config.Config()

np.random.seed(1000)

def load_and_process_imgs(im_tuple):
    im1 = skimage.io.imread(im_tuple[0])
    im2 = skimage.io.imread(im_tuple[1])
    if im1.ndim == 2:
            im1 = np.tile(im1[:, :, np.newaxis], (1, 1, 3))
    if im2.ndim == 2:
            im2 = np.tile(im2[:, :, np.newaxis], (1, 1, 3))
    imcrop1 = im1.astype(np.float32) - track_model.channel_mean
    imcrop2 = im2.astype(np.float32) - track_model.channel_mean
    return (imcrop1, imcrop2)


def run_prefetch(prefetch_queue, im_tuples, num_tuple, batch_size, shuffle):
    n_tuple_prefetch = 0
    fetch_order = np.arange(num_tuple)
    while True:
        #import ipdb; ipdb.set_trace()
        # Shuffle the batch order for every epoch
        if n_tuple_prefetch == 0 and shuffle:
            fetch_order = np.random.permutation(num_tuple)

        # Load batch from file
        if (n_tuple_prefetch+batch_size) > num_tuple:
            tuple_ids = np.concatenate((fetch_order[n_tuple_prefetch:num_tuple], fetch_order[:n_tuple_prefetch+batch_size-num_tuple]))
        else:
            tuple_ids = fetch_order[n_tuple_prefetch:n_tuple_prefetch+batch_size]
        imcrop_val1 = np.zeros((batch_size, config.query_H, config.query_W, 3), np.float32)
        imcrop_val2 = np.zeros((batch_size, config.input_H, config.input_W, 3), np.float32)
        for k, tuple_id in enumerate(tuple_ids):
            im_tuple = im_tuples[tuple_id]
            imcrop1, imcrop2 = load_and_process_imgs(im_tuple)
            imcrop_val1[k, ...] = imcrop1
            imcrop_val2[k, ...] = imcrop2
        
        # process the batch
        imcrop_val1 = imcrop_val1.transpose((0, 3, 1, 2))
        imcrop_val1 = imcrop_val1[:, ::-1, :, :]
        imcrop_val2 = imcrop_val2.transpose((0, 3, 1, 2))
        imcrop_val2 = imcrop_val2[:, ::-1, :, :]
        # use template labels
        #label_val = ...
        #label_val = label_val.transpose((0, 3, 1, 2))

        data = { 'imcrop_batch1': imcrop_val1,
                 'imcrop_batch2': imcrop_val2 }
                #'label_batch': label_val }

        # add loaded batch to fetchqing queue
        prefetch_queue.put(data, block=True)

        # Move to next batch
        n_tuple_prefetch = np.minimum(n_tuple_prefetch+batch_size, num_tuple) % num_tuple


class DataReader:
    def __init__(self, folder_name, input_prefix, query_prefix, split, batch_size, shuffle=True, prefetch_num=8):
        self.folder_name = folder_name
        self.input_prefix = input_prefix
        self.query_prefix = query_prefix
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0

        # obtain the videos
        if not split:
            # search the folder to see the number of num_batch
            videolist = os.listdir(folder_name+input_prefix+'/')
        else:
            # load split filelist
            with open(folder_name+split+'.txt') as f:
                videolist = f.read().splitlines()

        num_tuple = 0
        im_tuples = []
        for video in videolist:
            frames = sorted([os.path.basename(f) for f in glob(folder_name+input_prefix+'/'+video+'/*.png')])
            num_frame = len(frames)
            for i in range(num_frame):
                for j in range(i+1, num_frame):
                    im_tuples.append((folder_name+query_prefix+'/'+video+'/'+frames[i],
                                      folder_name+input_prefix+'/'+video+'/'+frames[j]))
                    im_tuples.append((folder_name+query_prefix+'/'+video+'/'+frames[j],
                                      folder_name+input_prefix+'/'+video+'/'+frames[i]))
        num_tuple = len(im_tuples)
        num_batch = np.ceil(num_tuple / batch_size)

        if num_batch > 0:
            print('found %d tuples %d batches under %s with prefix "%s"' % (num_tuple, num_batch, folder_name, input_prefix))
        else:
            raise RuntimeError('no tuples/batches under %s with prefix "%s"' % (folder_name, input_prefix))
        self.num_batch = num_batch
        self.num_tuple = num_tuple
        self.im_tuples = im_tuples

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.im_tuples, self.num_tuple,
                  self.batch_size, self.shuffle))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def read_batch(self):

        if self.n_batch % config.iter_display == 0 or (self.n_batch + 1) % self.num_batch == 0:
            print('data reader: epoch = %d, batch = %d / %d' % (self.n_epoch, self.n_batch, self.num_batch))
            
            # Get a batch from the prefetching queue
            #if self.prefetch_queue.empty():
            #    print('data reader: waiting for file input (IO is slow)...')
        batch = self.prefetch_queue.get(block=True)
        self.n_batch = (self.n_batch + 1) % self.num_batch
        self.n_epoch += (self.n_batch == 0)

        return batch


class TrackingDataProviderLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = ast.literal_eval(self.param_str)
        self.dataset = params['dataset']
        self.batch_size = params['batch_size']
        self.split = params['split']
        top[0].reshape(self.batch_size, 3, config.query_H, config.query_W)
        top[1].reshape(self.batch_size, 3, config.input_H, config.input_W)
        top[2].reshape(self.batch_size, 1, config.input_featmap_H, config.input_featmap_W)
        top[3].reshape(self.batch_size, 1, config.input_featmap_H, config.input_featmap_W)
        top[4].reshape(self.batch_size, 256+512, config.query_featmap_H//3, config.query_featmap_W//3)

        if self.split == 'test':
            pass
        else:
            self.reader = DataReader(config.data_dir, config.data_input_prefix, config.data_query_prefix,
                                     self.split, self.batch_size)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        if self.split == 'test':
            pass
        else:
            batch = self.reader.read_batch()
            imcrop_val1 = batch['imcrop_batch1']
            imcrop_val2 = batch['imcrop_batch2']
            label_val = config.template_labels
            sample_weights = config.sample_weights
            
            top[0].data[...] = imcrop_val1
            top[1].data[...] = imcrop_val2
            top[2].data[...] = label_val
            top[3].data[...] = sample_weights

    def backward(self, top, propagate_down, bottom):
        pass


class TossLayer1(caffe.Layer):
    def setup(self, bottom, top):
        params = ast.literal_eval(self.param_str)
        self.dataset = params['dataset']
        self.batch_size = params['batch_size']
        self.split = params['split']
        top[0].reshape(self.batch_size, 3, test_config.query_H, test_config.query_W)
        top[1].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class TossLayer2(caffe.Layer):
    def setup(self, bottom, top):
        params = ast.literal_eval(self.param_str)
        self.dataset = params['dataset']
        self.batch_size = params['batch_size']
        self.split = params['split']
        top[0].reshape(self.batch_size, 3, test_config.input_H, test_config.input_W)
        top[1].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


