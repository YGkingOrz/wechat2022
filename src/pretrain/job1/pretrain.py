#%%writefile pretrain.py
import os, math, random, time, sys, gc,  sys, json#, psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from imp import reload
reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)

import numpy as np
import pandas as pd
from config import parse_args
from config_qq.data_cfg import *
from config_qq.model_cfg import *
from config_qq.pretrain_cfg import *
from data_qq.record_trans import record_transform
from data_qq.qq_dataset import MultiModalDataset#QQDataset,
from qq_model.qq_uni_model import QQUniModel
from optim.create_optimizer import create_optimizer
from utils.eval_spearman import evaluate_emb_spearman
from utils.utils import set_random_seed

from tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset
# from tqdm import tqdm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ChainDataset
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

gc.enable()
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
               
set_random_seed(SEED)

def get_pred_and_loss(model, item, task=None):
    """Get pred and loss for specific task"""
    video_feature = item['frame_features'].to(DEVICE)
#     print(item['id'])
    input_ids = item['id'].to(DEVICE)
    attention_mask = item['mask'].to(DEVICE)
    video_mask = item['frame_mask'].to(DEVICE)
    
    target = None
    if 'target' in item:
        target = item['target'].to(DEVICE)

    pred, emb, loss,masked_lm_loss,masked_vm_loss,itm_loss= model(video_feature, video_mask, input_ids, attention_mask, target, task)

    return pred, emb, loss,masked_lm_loss,masked_vm_loss,itm_loss

def eval(model, data_loader, get_pred_and_loss, compute_loss=True, eval_max_num=99999):
    """Evaluates the |model| on |data_loader|"""
    model.eval()
    loss_l, emb_l, vid_l = [], [], []

    with torch.no_grad():
        for batch_num, item in enumerate(data_loader):
            pred, emb, loss ,masked_lm_loss,masked_vm_loss,itm_loss= get_pred_and_loss(model, item, task='tag')
#             print(loss)
            
            if loss is not None:
                loss_l.append(loss.to("cpu"))
                
            emb_l += emb.to("cpu").tolist()
            
            vid_l.append(item['vid'][0].numpy())
            
            if (batch_num + 1) * emb.shape[0] >= eval_max_num:
                break
            
    return np.mean(loss_l), np.array(emb_l), np.concatenate(vid_l)

def train(model, model_path, 
          train_loader,val_loader, 
          optimizer, get_pred_and_loss, scheduler=None, 
          num_epochs=5):
    best_val_loss, best_epoch, step = None, 0, 0
    start = time.time()
    num_total_steps = len(train_loader) * num_epochs
    for epoch in range(num_epochs):
        print('epoch------------------',epoch)
        for batch_num, item in tqdm(enumerate(train_loader)):
            model.train()
            optimizer.zero_grad()
            pred, emb, loss,masked_lm_loss,masked_vm_loss,itm_loss = get_pred_and_loss(model, item)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            if step % 100 == 0:
                time_per_step = (time.time() - start) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}  masked_lm_loss {masked_lm_loss:.3f}  masked_vm_loss {masked_vm_loss:.3f}  itm_loss {itm_loss:.3f}")

            step += 1

        torch.save(model.state_dict(),f"model_state_dict{epoch}_{loss}")

    return 0

# Show config
logging.info("Start")
for fname in ['pretrain', 'model', 'data']:
    logging.info('=' * 66)
    with open(f'config_qq/{fname}_cfg.py') as f:
        logging.info(f"Config - {fname}:" + '\n' + f.read().strip())
    
list_val_loss = []
logging.info(f"Model_type = {MODEL_TYPE}")
trans = record_transform(model_path=BERT_PATH, 
                         tag_file=f'{DATA_PATH}/tag_list.txt', 
                         get_tagid=True)

for fold in range(NUM_FOLDS):

    logging.info('=' * 66)
    model_path = f"model_pretrain_10w.pth"
    logging.info(f"Fold={fold + 1}/{NUM_FOLDS} seed={SEED+fold}")
    
    set_random_seed(SEED + fold)

    if LOAD_DATA_TYPE == 'fluid':
        # load data on fly, low memory required
        logging.info("Load data on fly")
        sample_dict = dict(zip([f'/pointwise/pretrain_{k}' for k in range(PRETRAIN_FILE_NUM)], [1/ (PRETRAIN_FILE_NUM + 2)]*(PRETRAIN_FILE_NUM)))
        sample_dict['/pairwise/pairwise'] = 2 / (PRETRAIN_FILE_NUM + 2)
        logging.info(sample_dict)
        train_dataset = MultiTFRecordDataset(data_pattern=DATA_PATH + "{}.tfrecords",
                                       index_pattern=None,
                                       splits=sample_dict,
                                       description=DESC,
                                       transform=trans.transform,
                                       infinite=False,
                                       shuffle_queue_size=1024)
        val_dataset = TFRecordDataset(data_path=f"{DATA_PATH}/pairwise/pairwise.tfrecords",
                              index_path=None,
                              description=DESC, 
                              transform=trans.transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, num_workers=1)

        total_steps = NUM_EPOCHS * (PRETRAIN_FILE_NUM * 50000 + 63573) // BATCH_SIZE
    else:
        # load data into memory, need about 60-70g memory
        args = parse_args()
        logging.info("Load data into memory")
        train_dataset_list = [f"{DATA_PATH}/pointwise/pretrain_{ix}.tfrecords" for ix in range(PRETRAIN_FILE_NUM)] + [f"{DATA_PATH}/pairwise/pairwise.tfrecords"]
        val_dataset = MultiModalDataset(args,f"../../data/annotations/labeled.json", f"../../data/zip_feats/labeled.zip")
        train_dataset = MultiModalDataset(args,f"../../data/annotations/unlabeled.json", f"../../data/zip_feats/unlabeled.zip")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, num_workers=4)

        
        total_steps = NUM_EPOCHS * len(train_dataset) // BATCH_SIZE
    
    warmup_steps = int(WARMUP_RATIO * total_steps)


    # model

    model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=PRETRAIN_TASK)
    model.to(DEVICE)

    # optimizer
    optimizer = create_optimizer(model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

    # schedueler
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)

    # train
    #val_loss = 
    train(model, model_path, train_loader, val_loader, optimizer, 
                     get_pred_and_loss=get_pred_and_loss,
                     scheduler=scheduler, num_epochs=NUM_EPOCHS)
    
    del train_dataset, val_dataset
    gc.collect()

