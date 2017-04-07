class Config():
    def __init__(self):
        # Model Params
        self.T = 20
        self.N = 1
        self.input_H = 512 
        self.input_W = 512 
        self.featmap_H = (self.input_H // 32)
        self.featmap_W = (self.input_W // 32)
        self.vocab_size = 8803
        self.embed_dim = 1000
        self.lstm_dim = 1000
        self.mlp_hidden_dims = 500
        self.fix_vgg = True
        self.vgg_dropout = False
        self.mlp_dropout = False


        # Testing Params
        self.gpu_id = 0
        self.pretrained_model = './snapshots/lang_high_res_seg/_iter_25000.caffemodel'
        self.score_thresh = 1e-9


        # Data Params
        self.data_provider = 'referit_data_provider'
        self.data_provider_layer = 'ReferitDataProviderLayer'

        self.image_dir = './referit/referit-dataset/images/'
        self.mask_dir = './referit/referit-dataset/mask/'
        self.query_file = './referit/data/referit_query_test.json'
        self.bbox_file = './referit/data/referit_bbox.json'
        self.imcrop_file = './referit/data/referit_imcrop.json'
        self.imsize_file = './referit/data/referit_imsize.json'
        self.vocab_file = './referit/data/vocabulary_referit.txt'


