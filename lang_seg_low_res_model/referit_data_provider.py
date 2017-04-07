import caffe
import numpy as np
import os
import threading
import Queue as queue
import ast

import train_config
import lang_seg_low_res_model as segmodel
from util.processing_tools import *

config = train_config.Config()


def create_cont(text_seq_batch):
    cont_batch = np.zeros_like(text_seq_batch)
    max_length = text_seq_batch.shape[0]
    for i in range(text_seq_batch.shape[1]):
        text = text_seq_batch[:, i]
        cont = np.zeros(text.shape[0])
        sent_begin = False
        for j, word in enumerate(text):
            if sent_begin:
                cont[j] = 1
            if word != 0:
                sent_begin = True
        cont_batch[:, i] = cont
    return cont_batch

def run_prefetch(prefetch_queue, folder_name, prefix, num_batch, shuffle):
    n_batch_prefetch = 0
    fetch_order = np.arange(num_batch)
    while True:
        # Shuffle the batch order for every epoch
        if n_batch_prefetch == 0 and shuffle:
            fetch_order = np.random.permutation(num_batch)

        # Load batch from file
        batch_id = fetch_order[n_batch_prefetch]
        save_file = os.path.join(folder_name, prefix+'_'+str(batch_id)+'.npz')
        npz_filemap = np.load(save_file)
        batch = dict(npz_filemap)
        npz_filemap.close()
        
        # process the batch
        text_seq_val = batch['text_seq_batch']
        cont_val = create_cont(text_seq_val)
        imcrop_val = batch['imcrop_batch'].astype(np.float32) - segmodel.channel_mean
        imcrop_val = imcrop_val.transpose((0, 3, 1, 2))
        imcrop_val = imcrop_val[:, ::-1, :, :]
        spatial_val = generate_spatial_batch(config.N, config.featmap_H, config.featmap_W)
        spatial_val = spatial_val.transpose((0, 3, 1, 2))
        label_val = batch['label_coarse_batch'].astype(np.float32)
        label_val = label_val.transpose((0, 3, 1, 2))

        data = { 'text_seq_batch': text_seq_val,
                 'cont_batch': cont_val,
                 'imcrop_batch': imcrop_val,
                 'spatial_batch': spatial_val,
                 'label_batch': label_val }

        # add loaded batch to fetchqing queue
        prefetch_queue.put(data, block=True)

        # Move to next batch
        n_batch_prefetch = (n_batch_prefetch + 1) % num_batch

class DataReader:
    def __init__(self, folder_name, prefix, shuffle=True, prefetch_num=8):
        self.folder_name = folder_name
        self.prefix = prefix
        self.shuffle = shuffle
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0

        # Search the folder to see the number of num_batch
        filelist = os.listdir(folder_name)
        num_batch = 0
        while (prefix + '_' + str(num_batch) + '.npz') in filelist:
            num_batch += 1
        if num_batch > 0:
            print('found %d batches under %s with prefix "%s"' % (num_batch, folder_name, prefix))
        else:
            raise RuntimeError('no batches under %s with prefix "%s"' % (folder_name, prefix))
        self.num_batch = num_batch

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.folder_name, self.prefix,
                  self.num_batch, self.shuffle))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def read_batch(self):
        print('data reader: epoch = %d, batch = %d / %d' % (self.n_epoch, self.n_batch, self.num_batch))

        # Get a batch from the prefetching queue
        if self.prefetch_queue.empty():
            print('data reader: waiting for file input (IO is slow)...')
        batch = self.prefetch_queue.get(block=True)
        self.n_batch = (self.n_batch + 1) % self.num_batch
        self.n_epoch += (self.n_batch == 0)

        return batch


class ReferitDataProviderLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = ast.literal_eval(self.param_str)
        self.batch_size = params['batch_size']
        self.split = params['split']
        top[0].reshape(config.T, self.batch_size)
        top[1].reshape(config.T, self.batch_size)
        top[2].reshape(self.batch_size, 3, config.input_H, config.input_W)
        top[3].reshape(self.batch_size, 8, config.featmap_H, config.featmap_W)
        top[4].reshape(self.batch_size, 1, config.featmap_H, config.featmap_W)

        if self.split == 'val' or self.split == 'test':
            pass
        else:
            self.reader = DataReader(config.data_folder, config.data_prefix)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        if self.split == 'val' or self.split == 'test':
            pass
        else:
            batch = self.reader.read_batch()
            text_seq_val = batch['text_seq_batch']
            cont_val = batch['cont_batch']
            imcrop_val = batch['imcrop_batch']
            spatial_val = batch['spatial_batch']
            label_val = batch['label_batch']

            top[0].data[...] = text_seq_val
            top[1].data[...] = cont_val
            top[2].data[...] = imcrop_val
            top[3].data[...] = spatial_val
            top[4].data[...] = label_val

    def backward(self, top, propagate_down, bottom):
        pass


