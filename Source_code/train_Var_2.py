import os
import sys

# Hack: disable transformers import to avoid side-effects.
sys.modules['transformers'] = None

from cal_composite import cal_composite_score
# 必须添加：解决显存碎片问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from monai.transforms import Compose, ScaleIntensity, RandCropByPosNegLabel, RandRotate, RandFlip, ToTensor
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, DiceHelper
from monai.data import DataLoader, CacheDataset, list_data_collate, decollate_batch
from monai.apps import DecathlonDataset

from models.Var_2 import Var_2

from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
import matplotlib.pyplot as plt
import time
import logging
from tqdm import tqdm
from dataloader import dataload
from datetime import datetime
import torch.nn as nn
from bdLoss import BoundaryLoss
import numpy as np

COARSE_WEIGHT = 0.8  # 保持一致

TRAIN_AMP = True
VAL_AMP = True
change = datetime.now().strftime("%m%d%H%M")
print(f"Training started at: {change}")
print(f"Architecture: Var_2 (3D Coarse + Old Fine + BBox Crop)")
print(f"AMP enabled: Train={TRAIN_AMP}, Val={VAL_AMP}")

def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(160, 160, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast(enabled=True):
            return _compute(input)
    else:
        return _compute(input)

def drawGraphs(loss_list, metric_values):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(loss_list))]
    y = loss_list
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")

    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [i + 1 for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")

    plt.savefig('/home/pyy/3D_MRI/two_stage/graphs/loss_metric_var2_' + change + '.png')

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    best_composite_score = -1
    best_composite_epoch = -1
    
    logger = logging.getLogger()
    handler = logging.FileHandler('/home/pyy/3D_MRI/two_stage/logs/var2_' + change + '.txt')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    root_dir = '/home/pyy/ori_data'
    model_dir = '/home/pyy/3D_MRI/two_stage/weights'

    train_loader, val_loader, _ = dataload(tr_batchsize=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Var_2(
        num_out_classes=3, 
        input_channels=4,
        coarse_base_channels=8,
        downsample_factor=4
    ).to(device)
    
    logger.info(f"Model: Var_2 (Ablation Study)")
    logger.info(f"  Stage 1: 3D CoarseNet (New)")
    logger.info(f"  Stage 2: FineNet (Old)")
    logger.info(f"  Cropping: BBox (New)")
    logger.info(f"  Deep Supervision: False (Disabled for fairness with Old FineNet)")

    loss_function = DiceLoss(sigmoid=False, smooth_nr=1e-5, smooth_dr=1e-5)
    # 模型输出已经过sigmoid，BoundaryLoss不需要再次sigmoid
    bdLoss = BoundaryLoss(apply_sigmoid=False)
    loss_mse = nn.MSELoss()

    def fuse_loss(inputs, labels):
        return 0.5 * loss_mse(inputs, labels) + loss_function(inputs, labels) + 0.5 * bdLoss(inputs, labels)

    optimizer = torch.optim.AdamW([
        {'params': model.coarse_unet.parameters(), 'lr': 0.00003},
        {'params': model.fine_unet.parameters(), 'lr': 0.0001}
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,
        eta_min=1e-6
    )
    
    warmup_epochs = 5
    warmup_factor = 0.01

    if hasattr(torch.cuda.amp, 'GradScaler'):
        scaler = torch.cuda.amp.GradScaler(enabled=TRAIN_AMP)
    else:
        scaler = torch.amp.GradScaler('cuda', enabled=TRAIN_AMP)

    post_pred = AsDiscrete(threshold=0.5)
    post_label = AsDiscrete(threshold=0.5)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    max_epochs = 200
    best_metric = -1
    best_metric_epoch = -1
    loss_list = []
    metric_values = []

    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        logger.info("-" * 10)
        logger.info(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} - Training", unit="batch") as tqdm_lodaer:
            for batch_data in tqdm_lodaer:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=TRAIN_AMP):
                    coarse_outputs, fine_outputs = model(inputs)
                    
                    labels_merge = labels.sum(dim=1, keepdim=True)
                    labels_merge = (labels_merge > 0.5).float()
                    
                    loss_coarse = fuse_loss(coarse_outputs, labels_merge)
                    
                    loss_fine = fuse_loss(fine_outputs, labels)
                    
                    loss = COARSE_WEIGHT * loss_coarse + loss_fine

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                step += 1
                tqdm_lodaer.set_postfix(loss=f"{loss.item():.4f}")

        if epoch < warmup_epochs:
            warmup_ratio = warmup_factor + (1 - warmup_factor) * (epoch + 1) / warmup_epochs
            coarse_lr = 0.00003 * warmup_ratio
            fine_lr = 0.0001 * warmup_ratio
            optimizer.param_groups[0]['lr'] = coarse_lr
            optimizer.param_groups[1]['lr'] = fine_lr
        else:
            scheduler.step()

        current_epoch_loss = epoch_loss / step
        loss_list.append(current_epoch_loss)
        logger.info(f"Epoch {epoch + 1} Training Loss: {current_epoch_loss:.4f}")

        should_validate = False
        if epoch < 50: should_validate = (epoch + 1) % 5 == 0
        elif epoch < 100: should_validate = (epoch + 1) % 3 == 0
        else: should_validate = (epoch + 1) % 2 == 0
        
        if should_validate:
            model.eval()
            with torch.no_grad():
                composite_scores = []
                with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} - Validation", unit="batch") as tqdm_val:
                    for val_data in tqdm_val:
                        val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                        coarse_outputs, fine_outputs = inference(val_inputs, model)
                        
                        fine_outputs_binary = post_pred(fine_outputs)
                        val_labels_binary = post_label(val_labels)
                        
                        dice_metric(y_pred=fine_outputs_binary, y=val_labels_binary)
                        dice_metric_batch(y_pred=fine_outputs_binary, y=val_labels_binary)

                        _, _, _, metrics = cal_composite_score(
                            fine_outputs, coarse_outputs, val_labels
                        )
                        composite_scores.append(metrics)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                
                avg_composite = np.mean([m["composite_score"] for m in composite_scores])
                avg_hd95 = np.mean([m["HD95_avg"] for m in composite_scores])
                
                dice_metric.reset()
                dice_metric_batch.reset()

                is_new_best_composite = avg_composite > best_composite_score
                is_new_best_dice = metric > best_metric

                if is_new_best_composite:
                    best_composite_score = avg_composite
                    best_composite_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_composite_var2_" + change + ".pth"))
                
                if is_new_best_dice:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_dice_var2_" + change + ".pth"))

                print("=" * 80)
                print(f"Epoch {epoch + 1}/{max_epochs} Validation Metrics:")
                print(f"  Mean Dice: {metric:.4f}")
                print(f"  Composite Score: {avg_composite:.4f}")
                print(f"  HD95 (mm): {avg_hd95:.2f}")
                print(f"  Best Dice: {best_metric:.4f} (Epoch {best_metric_epoch})")
                print(f"  Best Composite: {best_composite_score:.4f} (Epoch {best_composite_epoch})")
                print("=" * 80)
                
                logger.info(f"Epoch {epoch + 1} - Dice: {metric:.4f}, Composite: {avg_composite:.4f}, HD95: {avg_hd95:.2f}")

    drawGraphs(loss_list, metric_values)
