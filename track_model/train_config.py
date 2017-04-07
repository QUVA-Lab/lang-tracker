import numpy as np

def generate_template_labels(ov):
    if isinstance(ov, list):
        ov = np.array(ov)
    ov = ov.reshape((44, 44))

    l = np.zeros(ov.shape, dtype=np.float32)
    l[ov > 0.5] = 1.0
    return l

def generate_cost_weights(ov):
    if isinstance(ov, list):
        ov = np.array(ov)
    ov = ov.reshape((44, 44))
    scale_factor = 0.1

    w1 = np.zeros(ov.shape, dtype=np.float32)
    w1[ov >= 0.7] = 1.0
    w1 = w1 / w1.sum()
    w2 = np.zeros(ov.shape, dtype=np.float32)
    w2[ov < 0.5] = 1.0
    w2 = w2 / w2.sum()
    w = (w1 + w2) * scale_factor
    return w

class Config():
    def __init__(self):
        # Model Params
        self.N = 10
        self.input_H = 352
        self.input_W = 352
        self.query_H = 264
        self.query_W = 264
        self.input_featmap_H = (self.input_H // 8)
        self.input_featmap_W = (self.input_W // 8)
        self.query_featmap_H = (self.query_H // 8)
        self.query_featmap_W = (self.query_W // 8)

        # Training Params
        self.gpu_id = 0
        self.max_iter = 35000
        self.iter_display = 100

        #import ipdb; ipdb.set_trace()
        #self.ov = np.loadtxt('./track_model/ov.txt', delimiter=',').reshape((44, 44))
        #self.template_labels = np.tile(generate_template_labels(self.ov)[np.newaxis, np.newaxis, :, :], (self.N, 1, 1, 1))
        #self.sample_weights = np.tile(generate_cost_weights(self.ov)[np.newaxis, np.newaxis, :, :], (self.N, 1, 1, 1))
        
        self.weights = './VGG16.v2.caffemodel'
        self.fix_vgg = False  # set False to finetune VGG net
        self.finetune = False
        self.vgg_dropout = False

        # Data Params
        self.dataset = 'OTB50'
        #self.dataset = 'ILSVRC'
        self.data_provider = 'tracking_data_provider'
        self.data_provider_layer = 'TrackingDataProviderLayer'

        self.data_dir = './OTB50/'
        self.data_input_prefix = 'OTB50_cropped_frames_cropscale4_resize352'
        self.data_query_prefix = 'OTB50_cropped_frames_cropscale3_resize264'
        #self.data_dir = './ILSVRC/'
        #self.data_input_prefix = 'ILSVRC_cropped_frames_cropscale4_resize352'
        #self.data_query_prefix = 'ILSVRC_cropped_frames_cropscale3_resize264'
