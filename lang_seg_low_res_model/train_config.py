class Config():
    def __init__(self):
        # Model Params
        self.T = 20
        self.N = 10 
        self.input_H = 512
        self.input_W = 512
        self.featmap_H = (self.input_H // 32)
        self.featmap_W = (self.input_W // 32)
        self.vocab_size = 8803
        self.embed_dim = 1000
        self.lstm_dim = 1000
        self.mlp_hidden_dims = 500

        # Training Params
        self.gpu_id = 0 
        self.max_iter = 30000

        self.weights = './snapshots/lang_det_model_dyn_sigmoid/_iter_30000.caffemodel' # set as None if training from scratch
        self.fix_vgg = True  # set False to finetune VGG net
        self.vgg_dropout = False
        self.mlp_dropout = False

        # Data Params
        self.data_provider = 'referit_data_provider'
        self.data_provider_layer = 'ReferitDataProviderLayer'

        self.data_folder = './referit/data/train_batch_seg/'
        self.data_prefix = 'referit_train_seg'


