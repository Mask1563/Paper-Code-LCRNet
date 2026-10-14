import os
# 必须添加：解决显存碎片问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
from monai.transforms import RandScaleIntensityd, RandShiftIntensityd, RandFlipd
import torch
from monai.transforms import Compose, ScaleIntensity, RandCropByPosNegLabel, RandRotate, RandFlip, ToTensor
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, DiceHelper
from monai.data import DataLoader, CacheDataset, list_data_collate, decollate_batch
from monai.apps import DecathlonDataset
from cal_composite import cal_composite_score
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
    ConvertToMultiChannelBasedOnBratsClassesd,
)
from models.cascadeNet import CascadeUNet
from models.cascade_both import CascadeBoth
import matplotlib.pyplot as plt
import time
import os
import logging
from tqdm import tqdm
from dataloader import dataload
from datetime import datetime
import torch.nn as nn
from bdLoss import BoundaryLoss

VAL_AMP = False
change = datetime.now().strftime("%m%d%H%M")
print(change)


# 注意：这个函数将在模型定义后重新定义，以避免作用域问题
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
        with torch.cuda.amp.autocast():
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

    plt.savefig('D:/wjh/3D_MRI/two_stage/graphs/loss_metric' + change + '.png')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CascadeBoth().to(device)

def calculate_composite_scores(model, val_loader, device):
    """计算验证集综合评分."""
    model.eval()
    composite_scores = []
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            coarse_outputs, fine_outputs = inference(val_inputs, model)

            # 计算综合评分
            composite_score, _, _, metrics = cal_composite_score(
                fine_outputs, coarse_outputs, val_labels
            )
            composite_scores.append(composite_score)
            composite_scores.append(metrics)

    return composite_scores


def get_targeted_augmentations(composite_scores, current_transform):
    """基于综合评分返回新的数据增强策略."""
    hd95_scores = [m["HD95_mm"] for m in composite_scores]
    r_low_scores = [m["R_low"] for m in composite_scores]

    HD95_THRESHOLD = 10.0
    R_LOW_THRESHOLD = 0.3

    new_transforms = []

    if np.mean(hd95_scores) > HD95_THRESHOLD:
        print(f"HD95较高: {np.mean(hd95_scores):.2f} > {HD95_THRESHOLD}, 增加边界增强")
        new_transforms.append(RandScaleIntensityd(keys="image", factors=0.1, prob=1.0))
        new_transforms.append(RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0))
        new_transforms.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0))

    if np.mean(r_low_scores) > R_LOW_THRESHOLD:
        print(f"R_low较高: {np.mean(r_low_scores):.2f} > {R_LOW_THRESHOLD}, 增加低置信度区域增强")
        new_transforms.append(RandScaleIntensityd(keys="image", factors=0.15, prob=1.0))
        new_transforms.append(RandShiftIntensityd(keys="image", offsets=0.15, prob=1.0))

    if not new_transforms:
        print("使用当前数据增强策略")
        return current_transform

    return Compose(new_transforms)


if __name__ == '__main__':
    best_composite_score = -1
    best_composite_metrics_epochs_and_time = [[], [], []]  # 用于记录最佳综合评分的指标
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    logger = logging.getLogger()
    handler = logging.FileHandler('D:/wjh/3D_MRI/two_stage/logs/' + change + '.txt')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    root_dir = 'D:/wjh/ori_data'
    model_dir = 'D:/wjh/3D_MRI/two_stage/weights'  # best.pth和end.pth都会保存在这里面
    os.makedirs(model_dir, exist_ok=True)

    # 定义默认的训练和验证转换
    default_train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[160, 160, 128], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )

    default_val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[160, 160, 128], random_size=False),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    current_train_transform = default_train_transform

    train_loader, val_loader, _ = dataload(
        tr_batchsize=3,
        train_transform=current_train_transform,
        val_transform=default_val_transform
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CascadeBoth().to(device)

    loss_function = DiceLoss(sigmoid=True)
    bdLoss = BoundaryLoss()
    loss_bce = nn.BCEWithLogitsLoss()


    def fuse_loss(inputs, labels):
        return 0.5 * loss_bce(inputs, labels) + loss_function(inputs, labels) + 0.5 * bdLoss(inputs, labels)


    optimizer = torch.optim.Adam(model.parameters(), 0.000075)
    # 添加指数型学习率调度器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    print("begin to train.....")

    # 训练循环
    max_epochs = 200
    val_interval = 1
    best_metric = -1
    loss_list = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    best_metrics_epochs_and_time = [[], [], []]

    logger.info(f"this train is about both 2d and 3d, fuseLoss(+boundaryLoss)")
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
                step_start = time.time()
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

                # 前向传播
                coarse_outputs, fine_outputs = model(inputs)

                # 混合label的类别计算loss
                labels_merge = labels.sum(dim=1, keepdim=True)
                labels_merge = torch.where(labels_merge > 0.5, torch.tensor(1.0), torch.tensor(1e-3))

                # 计算损失
                loss_coarse = fuse_loss(coarse_outputs, labels_merge)
                loss_fine = fuse_loss(fine_outputs, labels)
                loss = loss_coarse + loss_fine

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 清理GPU内存
                torch.cuda.empty_cache()

                epoch_loss += loss.item()
                step += 1

                # tqdm_lodaer.set_postfix()

        # 每个epoch结束后更新学习率
        scheduler.step()

        # 保存损失
        loss_list.append(epoch_loss / step)

        # 验证
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    coarse_outputs, fine_outputs = inference(val_inputs, model)

                    dice_metric(y_pred=fine_outputs, y=val_labels)
                    dice_metric_batch(y_pred=fine_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_values_tc.append(metric_tc)
                metric_wt = metric_batch[1].item()
                metric_values_wt.append(metric_wt)
                metric_et = metric_batch[2].item()
                metric_values_et.append(metric_et)

                dice_metric.reset()
                dice_metric_batch.reset()

                # 计算综合评分
                composite_scores = calculate_composite_scores(model, val_loader, device)
                hd95_scores = [m["HD95_mm"] for m in composite_scores]
                r_low_scores = [m["R_low"] for m in composite_scores]
                # 计算平均综合评分
                avg_composite = np.mean([m["composite_score"] for m in composite_scores])

                # 打印综合评分指标
                logger.info(f"Composite Score Metrics (Validation):")
                logger.info(f" Avg Composite Score: {avg_composite:.4f}")
                logger.info(f"  Avg HD95: {np.mean(hd95_scores):.2f} mm")
                logger.info(f"  Avg R_low: {np.mean(r_low_scores):.2f}")


                # 基于综合评分动态调整训练增强策略
                current_train_transform = get_targeted_augmentations(
                    composite_scores,
                    current_train_transform
                )

                # 保存当前增强策略
                torch.save(current_train_transform, os.path.join(model_dir, f"augmentations_epoch{epoch}.pt"))
                logger.info(f"Updated training augmentations saved to {model_dir}/augmentations_epoch{epoch}.pt")

                # 保存基于综合评分的最佳模型
                if avg_composite > best_composite_score:
                    best_composite_score = avg_composite
                    best_composite_metrics_epochs_and_time[0].append(best_composite_score)
                    best_composite_metrics_epochs_and_time[1].append(epoch + 1)
                    best_composite_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_dir, "best_composite" + change + ".pth"),
                    )
                    logger.info("saved new best composite score model")

                # 保存最佳模型
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_dir, "best" + change + ".pth"),
                    )
                    logger.info("saved new best metric model")

            # 重新创建训练数据加载器，使用新的增强策略
            train_loader, _, _ = dataload(
                tr_batchsize=3,
                train_transform=current_train_transform,
                val_transform=default_val_transform
            )

        # 保存最终模型
        if epoch + 1 == max_epochs:
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, "end" + change + ".pth"),
            )
            logger.info(f"Saved end model to {model_dir}/end{change}.pth")

        # 日志输出
        logger.info(f"Loss: {epoch_loss / step}")
        logger.info(
            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
            f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
            f"best mean dice: {best_metric:.4f}"
            f" at epoch: {best_metric_epoch}"
        )
        logger.info(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    total_time = time.time() - total_start
    logger.info(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

    # 保存训练指标
    np.save(os.path.join(model_dir, f"loss_list_{change}.npy"), loss_list)
    np.save(os.path.join(model_dir, f"metric_values_{change}.npy"), metric_values)

    # 绘制图表
    drawGraphs(loss_list, metric_values)
    print("training completed.")