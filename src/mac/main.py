import logging
import os
import time
import torch
from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from ema import ExponentialMovingAverage,FGM,EMA,PGD
from tqdm import tqdm
import gc
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ["CUDA_VISIBLE_DEVICE"]='1'

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)
    model.train()
    return loss, results


def train_and_validate(args):
    for fold in range(1,11):
        print(f"--------第{fold}折--------")
        args.fold = fold
        swa_start = 0
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
        # 1. load data
        train_dataloader, val_dataloader = create_dataloaders(args)
        # 2. build model and optimizers
        model = MultiModal(args)
        
        optimizer, scheduler = build_optimizer(args, model)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05) # 添加   当SWA开始的时候，使用的学习率策略
        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))
        ema = EMA(model, decay=0.999)
        ema.register()
        fgm = FGM(model)
#         swa_model = AveragedModel(model)
#         swa_scheduler = SWALR(optimizer, swa_lr=0.05) # 添加   当SWA开始的时候，使用的学习率策略
        pgd = PGD(model)
        pgd_k=3
        # 3. training
        step = 0
        best_score = args.best_score
        start_time = time.time()
        num_total_steps = len(train_dataloader) * args.max_epochs
        for epoch in range(args.max_epochs):
            for batch in tqdm(train_dataloader):
                with autocast():
                    model.train()
                    loss, accuracy, _, _ = model(batch)
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                scaler.scale(loss).backward()
                fgm.attack() # 在embedding上添加对抗扰动
                with autocast():
                    model.train()
                    loss_adv, accuracy, _, _ = model(batch)
                    loss_adv = loss_adv.mean()
                    accuracy = accuracy.mean()      
                scaler.scale(loss_adv).backward() # 反向传播，并在正常
                fgm.restore()
                scaler.step(optimizer)
                #--添加
#                 if epoch > swa_start:
#                     swa_model.update_parameters(model)
#                     swa_scheduler.step()
#                 else:
                scheduler.step()
                optimizer.zero_grad()
    #             scaler.step(scheduler)
                ema.update()
                scaler.update()  
                step += 1
                if step % args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
    #             if step % 500==0:
    #                 ema.apply_shadow()
    #                 loss, results = validate(model, val_dataloader)
    #                 results = {k: round(v, 4) for k, v in results.items()}
    #                 logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

    #                 # 5. save checkpoint
    #                 mean_f1 = results['mean_f1']
    #                 if mean_f1 > 0.66:
    #                     best_score = mean_f1
    #                     torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
    #                                f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
    #                 ema.restore()
            # 4. validation
#             torch.optim.swa_utils.update_bn(train_dataloader, swa_model)
            ema.apply_shadow()
            loss, results = validate(model, val_dataloader)
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

            # 5. save checkpoint
            mean_f1 = results['mean_f1']
            if mean_f1 > best_score:
                best_score = mean_f1
                torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                           f'{args.savedmodel_path}/model_the_{fold}_10_zhe_epoch_{epoch}__mean_f1_{mean_f1}.bin')

            ema.restore()
            torch.cuda.empty_cache()
            gc.collect()

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
