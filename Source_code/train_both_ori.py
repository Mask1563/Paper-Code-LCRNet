import os

# 必须添加：解决显存碎片问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from monai.transforms import Compose, ScaleIntensity, RandCropByPosNegLabel, RandRotate, RandFlip, ToTensor
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, DiceHelper
from monai.data import DataLoader, CacheDataset, list_data_collate, decollate_batch
from monai.apps import DecathlonDataset
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


if __name__ == '__main__':
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

    train_loader, val_loader, _ = dataload(tr_batchsize=3)

    # 定义网络模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = UNet(dimensions=3, in_channels=1, out_channels=1).to(device)
    # model = CascadeUNet(num_out_classes=3, input_channels=4, init_coarse_channels=4, init_fine_channels=16, testing=False).to(device)
    model = CascadeBoth().to(device)
    # model.load_state_dict(torch.load("D:/wjh/3D_MRI/two_stage/models/200epoch.pth")) 当前无训练模型

    # 定义损失函数、优化器和评价指标
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

                # print('inputs, labels:', inputs.shape, labels.shape)
                coarse_outputs, fine_outputs = model(inputs)
                # 混合label的类别计算loss
                labels_merge = labels.sum(dim=1, keepdim=True)
                labels_merge = torch.where(labels_merge > 0.5, torch.tensor(1.0), torch.tensor(1e-3))
                # cal loss
                loss_coarse = fuse_loss(coarse_outputs, labels_merge)
                loss_fine = fuse_loss(fine_outputs, labels)

                loss = loss_coarse + loss_fine

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # 清理GPU内存
                torch.cuda.empty_cache()
                epoch_loss += loss.item()
                step += 1
                #
                tqdm_lodaer.set_postfix()

        # 每个epoch结束后更新学习率
        scheduler.step()

        # loss
        loss_list.append(epoch_loss / step)

        # val
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    coarse_outputs, fine_outputs = inference(val_inputs, model)
                    # fine_outputs = inference(val_inputs, model)
                    # val_outputs = [post_trans(i) for i in decollate_batch(fine_outputs)]

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

                # save model if it is best metric
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

                if epoch + 1 == max_epochs:
                    # save the end of this train
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_dir, "end" + change + ".pth"),
                    )

                logger.info(f"Loss: {epoch_loss / step}")
                logger.info(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"best mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}")

        logger.info(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    total_time = time.time() - total_start
    logger.info(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

    drawGraphs(loss_list, metric_values)

    print("training completed.")


