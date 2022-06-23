import json
import random
import zipfile
from io import BytesIO
from functools import partial
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from category_id_map import category_id_to_lv2id
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from tqdm import tqdm

def create_dataloaders(args):
    dataset= MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    
    size = len(dataset)
    val_size = int(size * args.val_ratio)    
    #手动五折
    ran_1 = [i for i in range(0,10000)]
    ran_2 = [i for i in range(10000,20000)]
    ran_3 = [i for i in range(20000,30000)]
    ran_4 = [i for i in range(30000,40000)]
    ran_5 = [i for i in range(40000,50000)]
    ran_6 = [i for i in range(50000,60000)]
    ran_7 = [i for i in range(60000,70000)]
    ran_8 = [i for i in range(70000,80000)]
    ran_9 = [i for i in range(80000,90000)]
    ran_10 = [i for i in range(90000,100000)]

    
    dataset_1 = torch.utils.data.Subset(dataset,ran_1)
    dataset_2 = torch.utils.data.Subset(dataset, ran_2)
    dataset_3 = torch.utils.data.Subset(dataset,ran_3)
    dataset_4 = torch.utils.data.Subset(dataset, ran_4)
    dataset_5 = torch.utils.data.Subset(dataset, ran_5)
    dataset_6 = torch.utils.data.Subset(dataset,ran_6)
    dataset_7 = torch.utils.data.Subset(dataset, ran_7)
    dataset_8 = torch.utils.data.Subset(dataset,ran_8)
    dataset_9 = torch.utils.data.Subset(dataset, ran_9)
    dataset_10 = torch.utils.data.Subset(dataset, ran_10)
    
    train_dataset_1 = torch.utils.data.ConcatDataset([dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7,dataset_8,dataset_9,dataset_10])

    train_dataset_2 = torch.utils.data.ConcatDataset([dataset_1,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7,dataset_8,dataset_9,dataset_10])
    train_dataset_3 = torch.utils.data.ConcatDataset([dataset_1,dataset_2,dataset_4,dataset_5,dataset_6,dataset_7,dataset_8,dataset_9,dataset_10])
    train_dataset_4 = torch.utils.data.ConcatDataset([dataset_1,dataset_2,dataset_3,dataset_5,dataset_6,dataset_7,dataset_8,dataset_9,dataset_10])
    train_dataset_5 = torch.utils.data.ConcatDataset([dataset_1,dataset_2,dataset_3,dataset_4,dataset_6,dataset_7,dataset_8,dataset_9,dataset_10])
    train_dataset_6 = torch.utils.data.ConcatDataset([dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_7,dataset_8,dataset_9,dataset_10])

    train_dataset_7 = torch.utils.data.ConcatDataset([dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_8,dataset_9,dataset_10])
    train_dataset_8 = torch.utils.data.ConcatDataset([dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7,dataset_9,dataset_10])
    train_dataset_9 = torch.utils.data.ConcatDataset([dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7,dataset_8,dataset_10])
    train_dataset_10 = torch.utils.data.ConcatDataset([dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7,dataset_8,dataset_9])
    
    #train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
    #                                    s                       generator=torch.Generator().manual_seed(args.seed))
    if args.fold == 1:
        train_dataset = train_dataset_1
        val_dataset = dataset_1
    if args.fold == 2:
        train_dataset = train_dataset_2
        val_dataset = dataset_2
    if args.fold == 3:
        train_dataset = train_dataset_3
        val_dataset = dataset_3
    if args.fold == 4:
        train_dataset = train_dataset_4
        val_dataset = dataset_4
    if args.fold == 5:
        train_dataset = train_dataset_5
        val_dataset = dataset_5
    if args.fold == 6:
        train_dataset = train_dataset_6
        val_dataset = dataset_6
    if args.fold == 7:
        train_dataset = train_dataset_7
        val_dataset = dataset_7
    if args.fold == 8:
        train_dataset = train_dataset_8
        val_dataset = dataset_8
    if args.fold == 9:
        train_dataset = train_dataset_9
        val_dataset = dataset_9
    if args.fold == 10:
        train_dataset = train_dataset_10
        val_dataset = dataset_10
    
    
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader


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
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # load annotations
        #with open(ann_path, 'r', encoding='utf8') as f:
        #    self.anns = json.load(f)
        
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        # initialize the text tokenizer
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

    def tokenize_text(self, text: str,) -> tuple:
        encoded_inputs = self.tokenizer('[CLS]' + text + '[SEP]' + asr + '[SEP]' + ocr + '[SEP]',
                                        max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask
    def tokenize_three_text(self, text1: str,text2: str,text3: str) -> tuple:
        encoded_inputs = self.tokenizer('[CLS]'+text1+'[SEP]'+text2+'[SEP]'+text3+'[SEP]', max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        token_type_ids = torch.LongTensor(encoded_inputs['token_type_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids,token_type_ids,mask
    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_feats(idx)

        # Step 2, load title tokens
        if(len(self.anns[idx]['title'])>126):
            self.anns[idx]['title']=self.anns[idx]['title'][:126]


        self.anns[idx]['sum_o']=''
        for i in range(len(self.anns[idx]['ocr'])):
            self.anns[idx]['sum_o']=self.anns[idx]['sum_o']+self.anns[idx]['ocr'][i]['text']
        if len(self.anns[idx]['sum_o'])>126:
            self.anns[idx]['sum_ocr']=self.anns[idx]['sum_o'][:126]
        else:
            self.anns[idx]['sum_ocr']=self.anns[idx]['sum_o']

            
        if len(self.anns[idx]['asr'])>126:
            self.anns[idx]['asr']=self.anns[idx]['asr'][:126]
        input_ids,token_type_ids,text_mask = self.tokenize_three_text(self.anns[idx]['title'],self.anns[idx]['asr'],self.anns[idx]['sum_ocr'])

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            input_ids=input_ids,
#             position_ids=position_ids,
            token_type_ids=token_type_ids,
            text_mask=text_mask,
        )
        
        data1 = copy.deepcopy(data)
        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])
        
        return data
