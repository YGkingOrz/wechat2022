#%%writefile data/qq_dataset.py

from config_qq.data_cfg import *
from config_qq.model_cfg import *
from config_qq.pretrain_cfg import *
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import AutoTokenizer
import torch
from sklearn.preprocessing import MultiLabelBinarizer
# from category_id_map import category_id_to_lv2id
from config import parse_args
# from config_weixin import parse_args
from transformers import BertTokenizer
import pandas as pd
import json
import random
import zipfile
from io import BytesIO
from functools import partial



class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        #加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str,asr: str,ocr:str) -> tuple:
        if len(ocr)==1:
            encoded_inputs = self.tokenizer(text+'seq'+asr[0:320]+'seq'+ocr[0]['text'], max_length=self.bert_seq_length, padding='max_length', truncation=True)
        elif len(ocr)==2:
            encoded_inputs = self.tokenizer(text+'seq'+asr[0:320]+'seq'+ocr[0]['text']+ocr[1]['text'], max_length=self.bert_seq_length, padding='max_length', truncation=True)
        elif len(ocr)>=3:
            encoded_inputs = self.tokenizer(text+'seq'+asr[0:320]+'seq'+ocr[0]['text']+ocr[1]['text']+ocr[2]['text'], max_length=self.bert_seq_length, padding='max_length', truncation=True)   
        else:
            encoded_inputs = self.tokenizer(text+'seq'+asr[0:512], max_length=self.bert_seq_length, padding='max_length', truncation=True)  
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_feats(idx)

        # Step 2, load title tokens
        title_input, title_mask = self.tokenize_text(self.anns[idx]['title'],self.anns[idx]['asr'],self.anns[idx]['ocr'])


        # Step 3, summarize into a dictionary
        data = dict(
            frame_features=frame_input,
            frame_mask=frame_mask,
            id=title_input,
            mask=title_mask,
        )


        return data
