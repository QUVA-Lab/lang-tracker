import numpy as np

class Config():
    def __init__(self):
        # Model Params
        self.N = 3
        self.input_H = 352
        self.input_W = 352
        self.query_H = 264
        self.query_W = 264
        self.input_featmap_H = (self.input_H // 8)
        self.input_featmap_W = (self.input_W // 8)
        self.query_featmap_H = (self.query_H // 8)
        self.query_featmap_W = (self.query_W // 8)
        self.fix_vgg = True
        self.vgg_dropout = False


        # Tracker Params
        self.qimage_size = 88 # be careful when extended for multiple scales
        self.timage_size = 88*4
        self.sz_times = 4 # sz_times * w, sz_times * h, (w,h) could be the init width and height
        self.spatial_ratio = 8
        self.scales = np.array([0.9639, 1.0000, 1.0375], dtype=np.float32)
        self.scale_penalty = np.full((self.scales.size, 1, 
                                      self.input_featmap_H, self.input_featmap_W), 0.962)
        self.scale_penalty[1, ...] = 1.0
        self.scaleLP = 0.34


        # Testing Params
        self.gpu_id = 0
        #self.pretrained_model = './VGG16.v2.caffemodel'
        self.pretrained_model = './VGG_ILSVRC_16_layers.caffemodel'


        # Data Params
        self.dataset = 'ILSVRC'
        self.data_provider = 'tracking_data_provider'
        self.data_provider_layer_1 = 'TossLayer1'
        self.data_provider_layer_2 = 'TossLayer2'

        self.signature = 'vgg16_ILSVRC_conv34_fullyconv_lrn_crop4_fix'
        self.test_file = './ILSVRC/test.txt'
        self.data_dir = './ILSVRC/ImageNetTracker/'
        self.pred_dir = './ILSVRC/preds/results_lang_seg_sigmoid_thresh0.4/'
        self.result_dir = './ILSVRC/pred_track_results/results_'+self.signature+'/'

