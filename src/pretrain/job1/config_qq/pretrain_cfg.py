# Pretrain file num
PRETRAIN_FILE_NUM = 1
LOAD_DATA_TYPE = 'mem'#'fluid'
# Training params
NUM_FOLDS = 1
SEED = 2021
BATCH_SIZE = 24
NUM_EPOCHS = 10
WARMUP_RATIO = 0.06
REINIT_LAYER = 0
WEIGHT_DECAY = 0.01
LR = {'others':5e-4, 'roberta':5e-5, 'newfc_videoreg':5e-4}
LR_LAYER_DECAY = 1.0
PRETRAIN_TASK = [ 'mlm','mfm','itm']#'tag','mfm'