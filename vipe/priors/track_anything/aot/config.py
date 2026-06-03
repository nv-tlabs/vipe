import os


class DefaultModelConfig:
    def __init__(self):
        self.MODEL_NAME = "AOTDefault"

        self.MODEL_VOS = "aot"
        self.MODEL_ENGINE = "aotengine"
        self.MODEL_ALIGN_CORNERS = True
        self.MODEL_ENCODER = "mobilenetv2"
        self.MODEL_ENCODER_PRETRAIN = "./pretrain_models/mobilenet_v2-b0353104.pth"
        self.MODEL_ENCODER_DIM = [24, 32, 96, 1280]  # 4x, 8x, 16x, 16x
        self.MODEL_ENCODER_EMBEDDING_DIM = 256
        self.MODEL_DECODER_INTERMEDIATE_LSTT = True
        self.MODEL_FREEZE_BN = True
        self.MODEL_FREEZE_BACKBONE = False
        self.MODEL_MAX_OBJ_NUM = 10
        self.MODEL_SELF_HEADS = 8
        self.MODEL_ATT_HEADS = 8
        self.MODEL_LSTT_NUM = 1
        self.MODEL_EPSILON = 1e-5
        self.MODEL_USE_PREV_PROB = False

        self.TRAIN_LONG_TERM_MEM_GAP = 9999
        self.TRAIN_AUG_TYPE = "v1"

        self.TEST_LONG_TERM_MEM_GAP = 9999

        self.TEST_SHORT_TERM_MEM_SKIP = 1


class DeAOTModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = "DeAOTDefault"

        self.MODEL_VOS = "deaot"
        self.MODEL_ENGINE = "deaotengine"

        self.MODEL_DECODER_INTERMEDIATE_LSTT = False

        self.MODEL_SELF_HEADS = 1
        self.MODEL_ATT_HEADS = 1

        self.TRAIN_AUG_TYPE = "v2"


class DeAOTTModelConfig(DeAOTModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = "DeAOTT"


class DeAOTSModelConfig(DeAOTModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = "DeAOTS"
        self.MODEL_LSTT_NUM = 2


class DeAOTBModelConfig(DeAOTModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = "DeAOTB"
        self.MODEL_LSTT_NUM = 3


class DeAOTLModelConfig(DeAOTModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = "DeAOTL"
        self.MODEL_LSTT_NUM = 3
        self.TRAIN_LONG_TERM_MEM_GAP = 2
        self.TEST_LONG_TERM_MEM_GAP = 5


class R50DeAOTLModelConfig(DeAOTLModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = "R50_DeAOTL"

        self.MODEL_ENCODER = "resnet50"
        self.MODEL_ENCODER_DIM = [256, 512, 1024, 1024]  # 4x, 8x, 16x, 16x

        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5


MyModelConfig = R50DeAOTLModelConfig


MODEL_CONFIGS = {
    "deaott": DeAOTTModelConfig,
    "deaots": DeAOTSModelConfig,
    "deaotb": DeAOTBModelConfig,
    "deaotl": DeAOTLModelConfig,
    "r50_deaotl": R50DeAOTLModelConfig,
}


class DefaultEngineConfig:
    def __init__(self, exp_name="default", model="r50_deaotl"):

        try:
            model_cfg = MODEL_CONFIGS[model.lower()]()
        except KeyError as exc:
            supported = ", ".join(sorted(MODEL_CONFIGS))
            raise ValueError(f"unsupported AOT model {model!r}; expected one of: {supported}") from exc

        self.__dict__.update(model_cfg.__dict__)  # add model config

        self.EXP_NAME = exp_name + "_" + self.MODEL_NAME

        self.STAGE_NAME = "YTB"

        self.DATASETS = ["youtubevos"]
        self.DATA_WORKERS = 8
        self.DATA_RANDOMCROP = (465, 465) if self.MODEL_ALIGN_CORNERS else (464, 464)
        self.DATA_RANDOMFLIP = 0.5
        self.DATA_MAX_CROP_STEPS = 10
        self.DATA_SHORT_EDGE_LEN = 480
        self.DATA_MIN_SCALE_FACTOR = 0.7
        self.DATA_MAX_SCALE_FACTOR = 1.3
        self.DATA_RANDOM_REVERSE_SEQ = True
        self.DATA_SEQ_LEN = 5
        self.DATA_DAVIS_REPEAT = 5
        self.DATA_RANDOM_GAP_DAVIS = (
            12  # max frame interval between two sampled frames for DAVIS (24fps)
        )
        self.DATA_RANDOM_GAP_YTB = (
            3  # max frame interval between two sampled frames for YouTube-VOS (6fps)
        )
        self.DATA_DYNAMIC_MERGE_PROB = 0.3

        self.PRETRAIN = True
        self.PRETRAIN_FULL = False  # if False, load encoder only
        self.PRETRAIN_MODEL = "./data_wd/pretrain_model/mobilenet_v2.pth"
        # self.PRETRAIN_MODEL = './pretrain_models/mobilenet_v2-b0353104.pth'

        self.TRAIN_TOTAL_STEPS = 100000
        self.TRAIN_START_STEP = 0
        self.TRAIN_WEIGHT_DECAY = 0.07
        self.TRAIN_WEIGHT_DECAY_EXCLUSIVE = {
            # 'encoder.': 0.01
        }
        self.TRAIN_WEIGHT_DECAY_EXEMPTION = [
            "absolute_pos_embed",
            "relative_position_bias_table",
            "relative_emb_v",
            "conv_out",
        ]
        self.TRAIN_LR = 2e-4
        self.TRAIN_LR_MIN = 2e-5 if "mobilenetv2" in self.MODEL_ENCODER else 1e-5
        self.TRAIN_LR_POWER = 0.9
        self.TRAIN_LR_ENCODER_RATIO = 0.1
        self.TRAIN_LR_WARM_UP_RATIO = 0.05
        self.TRAIN_LR_COSINE_DECAY = False
        self.TRAIN_LR_RESTART = 1
        self.TRAIN_LR_UPDATE_STEP = 1
        self.TRAIN_AUX_LOSS_WEIGHT = 1.0
        self.TRAIN_AUX_LOSS_RATIO = 1.0
        self.TRAIN_OPT = "adamw"
        self.TRAIN_SGD_MOMENTUM = 0.9
        self.TRAIN_GPUS = 4
        self.TRAIN_BATCH_SIZE = 16
        self.TRAIN_TBLOG = False
        self.TRAIN_TBLOG_STEP = 50
        self.TRAIN_LOG_STEP = 20
        self.TRAIN_IMG_LOG = True
        self.TRAIN_TOP_K_PERCENT_PIXELS = 0.15
        self.TRAIN_SEQ_TRAINING_FREEZE_PARAMS = ["patch_wise_id_bank"]
        self.TRAIN_SEQ_TRAINING_START_RATIO = 0.5
        self.TRAIN_HARD_MINING_RATIO = 0.5
        self.TRAIN_EMA_RATIO = 0.1
        self.TRAIN_CLIP_GRAD_NORM = 5.0
        self.TRAIN_SAVE_STEP = 5000
        self.TRAIN_MAX_KEEP_CKPT = 8
        self.TRAIN_RESUME = False
        self.TRAIN_RESUME_CKPT = None
        self.TRAIN_RESUME_STEP = 0
        self.TRAIN_AUTO_RESUME = True
        self.TRAIN_DATASET_FULL_RESOLUTION = False
        self.TRAIN_ENABLE_PREV_FRAME = False
        self.TRAIN_ENCODER_FREEZE_AT = 2
        self.TRAIN_LSTT_EMB_DROPOUT = 0.0
        self.TRAIN_LSTT_ID_DROPOUT = 0.0
        self.TRAIN_LSTT_DROPPATH = 0.1
        self.TRAIN_LSTT_DROPPATH_SCALING = False
        self.TRAIN_LSTT_DROPPATH_LST = False
        self.TRAIN_LSTT_LT_DROPOUT = 0.0
        self.TRAIN_LSTT_ST_DROPOUT = 0.0

        self.TEST_GPU_ID = 0
        self.TEST_GPU_NUM = 1
        self.TEST_FRAME_LOG = False
        self.TEST_DATASET = "youtubevos"
        self.TEST_DATASET_FULL_RESOLUTION = False
        self.TEST_DATASET_SPLIT = "val"
        self.TEST_CKPT_PATH = None
        # if "None", evaluate the latest checkpoint.
        self.TEST_CKPT_STEP = None
        self.TEST_FLIP = False
        self.TEST_MULTISCALE = [1]
        self.TEST_MAX_SHORT_EDGE = None
        self.TEST_MAX_LONG_EDGE = 800 * 1.3
        self.TEST_WORKERS = 4

        # GPU distribution
        self.DIST_ENABLE = True
        self.DIST_BACKEND = "nccl"  # "gloo"
        self.DIST_URL = "tcp://127.0.0.1:13241"
        self.DIST_START_GPU = 0

    def init_dir(self):
        self.DIR_DATA = "../VOS02/datasets"  #'./datasets'
        self.DIR_DAVIS = os.path.join(self.DIR_DATA, "DAVIS")
        self.DIR_YTB = os.path.join(self.DIR_DATA, "YTB")
        self.DIR_STATIC = os.path.join(self.DIR_DATA, "Static")

        self.DIR_ROOT = "./"  #'./data_wd/youtube_vos_jobs'

        self.DIR_RESULT = os.path.join(
            self.DIR_ROOT, "result", self.EXP_NAME, self.STAGE_NAME
        )
        self.DIR_CKPT = os.path.join(self.DIR_RESULT, "ckpt")
        self.DIR_EMA_CKPT = os.path.join(self.DIR_RESULT, "ema_ckpt")
        self.DIR_LOG = os.path.join(self.DIR_RESULT, "log")
        self.DIR_TB_LOG = os.path.join(self.DIR_RESULT, "log", "tensorboard")
        self.DIR_IMG_LOG = "./img_logs"
        self.DIR_EVALUATION = "./results"


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name, model="r50_deaotl"):
        super().__init__(exp_name, model)
        self.STAGE_NAME = "PRE_YTB_DAV"

        self.init_dir()

        self.DATASETS = ["youtubevos", "davis2017"]

        pretrain_stage = "PRE"
        pretrain_ckpt = "save_step_100000.pth"
        self.PRETRAIN_FULL = True  # if False, load encoder only
        self.PRETRAIN_MODEL = os.path.join(
            self.DIR_ROOT,
            "result",
            self.EXP_NAME,
            pretrain_stage,
            "ema_ckpt",
            pretrain_ckpt,
        )
