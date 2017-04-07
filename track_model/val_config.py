import numpy as np

class Config():
    def __init__(self):
        # Model Params
        self.N = 3
        self.input_H = 160
        self.input_W = 160
        self.fix_vgg = True
        self.vgg_dropout = False


        # Tracker Params
        self.qimage_size = 80 # be careful when extended for multiple scales
        self.timage_size = 160
        self.sz_times = 2 # sz_times * w, sz_times * h, (w,h) could be the init width and height
        self.spatial_ratio = 8
        self.scales = np.array([0.9639, 1.0000, 1.0375], dtype=np.float32)
        self.scale_penalty = np.full((self.scales.size, 1, 
                            (self.timage_size-self.qimage_size)/self.spatial_ratio+1,
                            (self.timage_size-self.qimage_size)/self.spatial_ratio+1), 0.962)
        self.scale_penalty[1, ...] = 1.0
        self.scaleLP = 0.34


        # Testing Params
        self.gpu_id = 0
        #self.pretrained_model = './VGG16.v2.caffemodel'
        self.pretrained_model = './snapshots/sint_model/'
        self.snapshots = range(0, 20001, 1000)


        # Data Params
        self.dataset = 'ALOV'
        self.data_provider = 'tracking_data_provider'
        self.data_provider_layer = 'TossLayer'

        self.signature = 'vgg16_conv345_fullyconv_lrn_finetune'
        self.data_dir = './ALOV/imagedata/'
        self.meta_dir = './ALOV/alov_metadata/'
        self.result_dir = './ALOV/results/'
        self.split = 'val'

