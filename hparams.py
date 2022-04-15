class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def create_hparams():
    """Create model hyperparameters"""

    hparams = AttrDict({
        ################################
        # Experiment Parameters        #
        ################################
        "epochs": 200,
        "max_training_steps": 1000,
        "iters_per_checkpoint":50,
        "seed":1234,
        "fp16_run":False,
        "cudnn_enabled":True,
        "cudnn_benchmark":False,
        "ignore_layers":['output.weight', 'output.bias'],

        ################################
        # Data Parameters             #
        ################################
        "training_files":'recordings/train_filelist.txt',
        "validation_files":'recordings/valid_filelist.txt',
        "n_frames_per_utt": 60,
        "n_labels": 5,

        ################################
        # Audio Parameters             #
        ################################
        "sample_rate":16000,
        "n_fft":1024,
        "hop_length":256,
        "win_length":1024,
        "n_mels":80,
        "f_min":0.0,
        "f_max":8000.0,
        "mel_normalized":True,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate":False,
        "learning_rate":1e-3,
        "weight_decay":1e-6,
        "grad_clip_thresh":1.0,
        "batch_size":4,
        "mask_padding":True  # set model's padded outputs to padded values
    })

    return hparams
