import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal
from math import sqrt
import numpy as np
from tqdm import tqdm


def inference():
    models=['./save/v1/model_the_1_10_zhe_epoch_3__mean_f1_0.6716.bin',
            './save/v1/model_the_2_10_zhe_epoch_3__mean_f1_0.6647.bin',
           './save/v1/model_the_3_10_zhe_epoch_3__mean_f1_0.6671.bin',
           './save/v1/model_the_4_10_zhe_epoch_3__mean_f1_0.671.bin',
           './save/v1/model_the_5_10_zhe_epoch_3__mean_f1_0.6672.bin',
           './save/v1/model_the_6_10_zhe_epoch_3__mean_f1_0.6638.bin',
           './save/v1/model_the_7_10_zhe_epoch_3__mean_f1_0.6655.bin',
           './save/v1/model_the_8_10_zhe_epoch_3__mean_f1_0.6707.bin',
           './save/v1/model_the_9_10_zhe_epoch_3__mean_f1_0.6782.bin',
           './save/v1/model_the_10_10_zhe_epoch_3__mean_f1_0.6691.bin',]

    sum_pre=[]
    for i in tqdm(range(0,10)):
        args = parse_args()
        args.num_workers = 4
        # 1. load data
        dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=args.test_batch_size,
                                sampler=sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)

        # 2. load model
        model = MultiModal(args)
        checkpoint = torch.load(models[i], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()

        # 3. inference
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                if args.is5zhe:

                    pre = model(batch, inference=True)
                    predictions.extend(pre.cpu().numpy())
                else:
                    pred_label_id = model(batch, inference=True)
                    predictions.extend(pred_label_id.cpu().numpy())
        sum_pre.append(predictions)
        
    pro = np.mean(sum_pre,axis=0)
    predictions = list(pro)
    print("pro.shape",pro.shape)
    print("pro.type",type(pro))
    print("len",len(predictions))
    np.savetxt( "prodiction_B.csv", pro, delimiter="," )

    predictions = np.argmax(predictions, axis=1)
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
        


if __name__ == '__main__':
    inference()
