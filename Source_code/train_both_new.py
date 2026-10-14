import os
import sys
# Hack: disable transformers import to avoid numpy/pytree issues.
sys.modules['transformers'] = None

from cal_composite import cal_composite_score
# 必须添加：解决显存碎片问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from monai.transforms import Compose, ScaleIntensity, RandCropByPosNegLabel, RandRotate, RandFlip, ToTensor
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, DiceHelper
from monai.data import DataLoader, CacheDataset, list_data_collate, decollate_batch
from monai.apps import DecathlonDataset
from models.cascade_both_new import CascadeBoth_New  # 假设新模型在cascade_both_new.py中
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
    # 新增：激进数据增强
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandSimulateLowResolutiond,
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
import numpy as np

# ========== 新架构配置 ==========
USE_NEW_ARCHITECTURE = True  # 是否使用新架构CascadeBoth_New
USE_DEEP_SUPERVISION = True  # 是否启用深度监督（仅新架构）
DEEP_SUPERVISION_WEIGHTS = [1.0, 0.5, 0.25]  # 深度监督权重 [final, deep1, deep2]
COARSE_WEIGHT = 0.8  # 粗分割损失权重（增加以强化粗网络训练）

# 混合精度训练配置
TRAIN_AMP = True  # 训练时启用AMP
VAL_AMP = True  # 验证时也启用AMP
change = datetime.now().strftime("%m%d%H%M")
print(f"Training started at: {change}")
print(f"Architecture: {'CascadeBoth_New' if USE_NEW_ARCHITECTURE else 'CascadeBoth'}")
print(f"Deep Supervision: {USE_DEEP_SUPERVISION if USE_NEW_ARCHITECTURE else 'N/A'}")
print(f"AMP enabled: Train={TRAIN_AMP}, Val={VAL_AMP}")


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

    plt.savefig('/home/pyy/3D_MRI/two_stage/graphs/loss_metric' + change + '.png')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    best_composite_score = -1
    best_composite_epoch = -1
    best_composite_metrics_epochs_and_time = [[], [], []]  # 用于记录最佳综合评分的指标

    logger = logging.getLogger()
    handler = logging.FileHandler('/home/pyy/3D_MRI/two_stage/logs/' + change + '.txt')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    root_dir = '/home/pyy/ori_data'
    model_dir = '/home/pyy/3D_MRI/two_stage/weights'  # best.pth和end.pth都会保存在这里面

    train_loader, val_loader, _ = dataload(tr_batchsize=3)

    # 定义网络模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # BraTS 有 3 个肿瘤子区域（TC, WT, ET），输入是 4 个模态
    if USE_NEW_ARCHITECTURE:
        model = CascadeBoth_New(
            num_out_classes=3, 
            input_channels=4,
            use_deep_supervision=USE_DEEP_SUPERVISION,
            downsample_factor=4  # 粗分割下采样4倍
        ).to(device)
        logger.info(f"Model: CascadeBoth_New (3D Coarse + ConvNeXt-3D Fine)")
        logger.info(f"Deep Supervision: {USE_DEEP_SUPERVISION}")
    else:
        model = CascadeBoth(num_out_classes=3, input_channels=4).to(device)
        logger.info(f"Model: CascadeBoth (Original 2D+3D)")
    
    # 从检查点恢复（设置为None禁用，或提供完整路径启用）
    RESUME_CHECKPOINT = None  # "D:/wjh/3D_MRI/two_stage/weights/best_composite11230039.pth"
    START_EPOCH = 0  # 检查点对应的epoch编号
    
    logger.info(f"Expected label shape: (B, 3, H, W, D) for TC/WT/ET")

    # 定义损失函数、优化器和评价指标
    # 注意：新模型输出已经过sigmoid，不需要再次sigmoid
    loss_function = DiceLoss(sigmoid=False, smooth_nr=1e-5, smooth_dr=1e-5)
    bdLoss = BoundaryLoss(apply_sigmoid=False)  # 模型输出已过sigmoid，不需要再次sigmoid
    # MSELoss替代BCELoss（AMP安全，且对sigmoid输出效果相似）
    loss_mse = nn.MSELoss()


    def fuse_loss(inputs, labels):
        """融合损失：MSE + Dice + Boundary (AMP安全版本)"""
        # 使用MSE替代BCE：AMP安全，对sigmoid输出效果相近，保持三项损失完整性
        return 0.5 * loss_mse(inputs, labels) + loss_function(inputs, labels) + 0.5 * bdLoss(inputs, labels)
    
    
    def compute_deep_supervision_loss(outputs_list, labels, weights=None):
        """计算深度监督的加权损失"""
        if weights is None:
            weights = DEEP_SUPERVISION_WEIGHTS
        
        total_loss = 0.0
        for i, (output, weight) in enumerate(zip(outputs_list, weights)):
            loss = fuse_loss(output, labels)
            total_loss += weight * loss
        return total_loss


    # Step 3: 使用AdamW + 差异化学习率
    # 粗网络需要更低学习率，避免过拟合为全1输出
    optimizer = torch.optim.AdamW([
        {'params': model.coarse_unet.parameters(), 'lr': 0.00003},  # 粗网络：更低学习率
        {'params': model.fine_unet.parameters(), 'lr': 0.0001}      # 精细网络：标准学习率
    ], weight_decay=1e-5)

    # 余弦退火调度器：不重启，持续衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,  # 总训练轮数
        eta_min=1e-6  # 最小学习率
    )
    
    # 从检查点恢复完整训练状态（在定义optimizer和scheduler之后）
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整检查点格式（包含优化器、调度器等）
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epoch' in checkpoint:
                START_EPOCH = checkpoint['epoch']
            if 'best_metric' in checkpoint:
                best_metric = checkpoint['best_metric']
                best_metric_epoch = checkpoint.get('best_metric_epoch', START_EPOCH)
            if 'best_composite_score' in checkpoint:
                best_composite_score = checkpoint['best_composite_score']
                best_composite_epoch = checkpoint.get('best_composite_epoch', START_EPOCH)
            logger.info(f"✓ Resumed from checkpoint: {RESUME_CHECKPOINT}")
            logger.info(f"  Starting epoch: {START_EPOCH + 1}")
            logger.info(f"  Best Dice: {best_metric:.4f} (Epoch {best_metric_epoch})")
            logger.info(f"  Best Composite: {best_composite_score:.4f} (Epoch {best_composite_epoch})")
        else:
            # 仅权重格式（向后兼容）
            model.load_state_dict(checkpoint)
            logger.info(f"⚠ Loaded model weights only from: {RESUME_CHECKPOINT}")
            logger.info(f"  Optimizer and scheduler will start fresh")
    
    # Warmup调度器（前5个epoch线性增加学习率）
    warmup_epochs = 5
    warmup_factor = 0.01  # 从1%开始（1e-6），逐渐增加到100%（1e-4）

    # Step 1: 初始化GradScaler用于混合精度训练
    # 兼容旧版本PyTorch (如1.12等)
    if hasattr(torch.cuda.amp, 'GradScaler'):
        scaler = torch.cuda.amp.GradScaler(enabled=TRAIN_AMP)
    else:
        scaler = torch.amp.GradScaler('cuda', enabled=TRAIN_AMP)

    # 注意：BraTS标签已经是3个肿瘤通道（TC, WT, ET），没有额外的背景通道
    # include_background=True 表示计算所有通道（这里是3个肿瘤通道）
    # 关键：模型输出已过sigmoid，需要用AsDiscrete转换为one-hot格式
    from monai.transforms import AsDiscrete
    post_pred = AsDiscrete(threshold=0.5)
    post_label = AsDiscrete(threshold=0.5)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    print("begin to train.....")

    # 训练循环
    max_epochs = 200
    # 阶段性验证策略：早期稀疏，后期密集
    # val_interval = 1  # 原配置：每轮验证
    best_metric = -1
    best_metric_epoch = -1
    loss_list = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    best_metrics_epochs_and_time = [[], [], []]

    arch_name = "CascadeBoth_New (3D Coarse + ConvNeXt)" if USE_NEW_ARCHITECTURE else "CascadeBoth (2D Fuse)"
    logger.info(f"Training: {arch_name}")
    logger.info(f"Validation Strategy: Staged (Epoch 0-50: every 5, 50-100: every 3, 100-200: every 2)")
    logger.info(f"Loss: fuseLoss (0.5*MSE + Dice + 0.5*Boundary), Coarse Weight: {COARSE_WEIGHT}")
    logger.info(f"Note: MSE replaces BCE for AMP compatibility (similar performance)")
    if USE_NEW_ARCHITECTURE and USE_DEEP_SUPERVISION:
        logger.info(f"Deep Supervision: Weights {DEEP_SUPERVISION_WEIGHTS}")
    logger.info(f"Optimizer: AdamW (coarse_lr=3e-5, fine_lr=1e-4, weight_decay=1e-5)")
    logger.info(f"Scheduler: CosineAnnealingLR (T_max=200, eta_min=1e-6, no restart)")
    logger.info(f"Mixed Precision (AMP): Train={TRAIN_AMP}, Val={VAL_AMP}")
    logger.info(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    total_start = time.time()
    for epoch in range(START_EPOCH, max_epochs):
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

                optimizer.zero_grad()

                # Step 1: 使用混合精度训练
                with torch.cuda.amp.autocast(enabled=TRAIN_AMP):
                    coarse_outputs, fine_outputs = model(inputs)
                    
                    # 调试：打印第一个batch的详细信息
                    if epoch == 0 and step == 1:
                        print(f"\n{'='*80}")
                        print(f"[DEBUG] Shape Information:")
                        print(f"  Input shape: {inputs.shape}")
                        print(f"  Labels shape: {labels.shape}")
                        print(f"  Coarse output shape: {coarse_outputs.shape}")
                        if isinstance(fine_outputs, list):
                            print(f"  Fine outputs (list) shapes: {[f.shape for f in fine_outputs]}")
                        else:
                            print(f"  Fine output shape: {fine_outputs.shape}")
                        
                        print(f"\n[DEBUG] Value Range:")
                        print(f"  Coarse output range: [{coarse_outputs.min():.4f}, {coarse_outputs.max():.4f}]")
                        print(f"  Coarse output mean: {coarse_outputs.mean():.4f}")
                        print(f"  Coarse output > 0.5 ratio: {(coarse_outputs > 0.5).float().mean():.4f}")
                        print(f"  Labels unique values: {torch.unique(labels)}")
                        print(f"  Labels tumor ratio: {labels.sum() / labels.numel():.4f}")
                        
                        print(f"\n[DEBUG] Merged Labels for Coarse:")
                        labels_merge_debug = labels.sum(dim=1, keepdim=True)
                        labels_merge_debug = (labels_merge_debug > 0.5).float()
                        print(f"  Merged labels shape: {labels_merge_debug.shape}")
                        print(f"  Merged labels tumor ratio: {labels_merge_debug.mean():.4f}")
                        print(f"{'='*80}\n")
                    
                    # 混合label的类别计算粗分割loss（整体肿瘤）
                    labels_merge = labels.sum(dim=1, keepdim=True)
                    labels_merge = (labels_merge > 0.5).float()
                    
                    # 计算粗分割损失
                    loss_coarse = fuse_loss(coarse_outputs, labels_merge)
                    
                    # 计算精细分割损失（处理深度监督）
                    if USE_NEW_ARCHITECTURE and USE_DEEP_SUPERVISION and isinstance(fine_outputs, list):
                        # 深度监督模式：fine_outputs是list
                        loss_fine = compute_deep_supervision_loss(fine_outputs, labels)
                    else:
                        # 普通模式：fine_outputs是tensor
                        loss_fine = fuse_loss(fine_outputs, labels)
                    
                    # 总损失
                    loss = COARSE_WEIGHT * loss_coarse + loss_fine

                # 使用GradScaler进行反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                step += 1
                tqdm_lodaer.set_postfix(loss=f"{loss.item():.4f}")

        # 每个epoch结束后更新学习率
        if epoch < warmup_epochs:
            # Warmup阶段：线性增加学习率（差异化）
            warmup_ratio = warmup_factor + (1 - warmup_factor) * (epoch + 1) / warmup_epochs
            coarse_lr = 0.00003 * warmup_ratio
            fine_lr = 0.0001 * warmup_ratio
            optimizer.param_groups[0]['lr'] = coarse_lr  # 粗网络
            optimizer.param_groups[1]['lr'] = fine_lr    # 精细网络
            logger.info(f"Learning rate (warmup): coarse={coarse_lr:.6f}, fine={fine_lr:.6f}")
        else:
            # Warmup后使用余弦退火
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")

        # loss
        current_epoch_loss = epoch_loss / step
        loss_list.append(current_epoch_loss)
        # 补充记录：确保每个epoch都记录训练loss，方便后期画图
        logger.info(f"Epoch {epoch + 1} Training Loss: {current_epoch_loss:.4f}")

        # 阶段性验证：根据epoch动态调整验证频率
        should_validate = False
        if epoch < 50:
            should_validate = (epoch + 1) % 5 == 0  # 前50轮：每5轮验证
        elif epoch < 100:
            should_validate = (epoch + 1) % 3 == 0  # 50-100轮：每3轮验证
        else:
            should_validate = (epoch + 1) % 2 == 0  # 100-200轮：每2轮验证
        
        # val
        if should_validate:
            model.eval()
            with torch.no_grad():
                composite_scores = []
                # 验证阶段也使用AMP加速
                print(f"\n{'='*80}")
                print(f"Validating Epoch {epoch + 1}/{max_epochs}...")
                print(f"{'='*80}")
                with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} - Validation", unit="batch") as tqdm_val:
                    for val_data in tqdm_val:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        coarse_outputs, fine_outputs = inference(val_inputs, model)
                        
                        # 处理深度监督的输出（推理时应该返回单个tensor，但以防万一）
                        if isinstance(fine_outputs, list):
                            fine_outputs = fine_outputs[0]  # 只使用最终输出

                        # 应用后处理：将概率值二值化为0/1
                        fine_outputs_binary = post_pred(fine_outputs)
                        val_labels_binary = post_label(val_labels)
                        
                        dice_metric(y_pred=fine_outputs_binary, y=val_labels_binary)
                        dice_metric_batch(y_pred=fine_outputs_binary, y=val_labels_binary)

                        # 计算综合评分
                        _, _, _, metrics = cal_composite_score(
                            fine_outputs, coarse_outputs, val_labels
                        )
                        composite_scores.append(metrics)

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

                # 2. 提取指标（直接使用 composite_scores，避免重复计算）
                hd95_scores = [m["HD95_avg"] for m in composite_scores]
                r_low_scores = [m["R_low"] for m in composite_scores]
                dice_scores = [m["Mean Dice"] for m in composite_scores]  # 无需重复计算

                # 3. 计算平均综合评分
                avg_composite = np.mean([m["composite_score"] for m in composite_scores])
                avg_hd95 = np.mean(hd95_scores)
                avg_r_low = np.mean(r_low_scores)
                avg_dice_composite = np.mean(dice_scores)

                # ============ 先更新最佳指标（在打印之前）============
                # 标记是否有新的最佳模型
                is_new_best_composite = avg_composite > best_composite_score
                is_new_best_dice = metric > best_metric

                # 更新综合评分最佳记录
                if is_new_best_composite:
                    best_composite_score = avg_composite
                    best_composite_epoch = epoch + 1
                    best_composite_metrics_epochs_and_time[0].append(best_composite_score)
                    best_composite_metrics_epochs_and_time[1].append(best_composite_epoch)
                    best_composite_metrics_epochs_and_time[2].append(time.time() - total_start)

                # 更新Dice最佳记录
                if is_new_best_dice:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)

                # ============ 然后打印（显示最新的最佳值）============
                print("=" * 80)
                print(f"Epoch {epoch + 1}/{max_epochs} Validation Metrics:")
                print(f"  Loss: {epoch_loss / step:.4f}")
                print(f"  Mean Dice: {metric:.4f} (TC: {metric_tc:.4f}, WT: {metric_wt:.4f}, ET: {metric_et:.4f})")
                print(f"  Composite Score: {avg_composite:.4f}")
                print(f"  HD95 (mm): {avg_hd95:.2f}")
                print(f"  R_low: {avg_r_low:.4f}")
                print(f"  Dice (from composite): {avg_dice_composite:.4f}")
                print(f"  Best Dice: {best_metric:.4f} (Epoch {best_metric_epoch})" + 
                      (" ← NEW BEST!" if is_new_best_dice else ""))
                print(f"  Best Composite: {best_composite_score:.4f} (Epoch {best_composite_epoch})" +
                      (" ← NEW BEST!" if is_new_best_composite else ""))
                print("=" * 80)

                # 只记录简要信息到日志
                logger.info(
                    f"Epoch {epoch + 1} - Loss: {epoch_loss / step:.4f}, Dice: {metric:.4f}, Composite: {avg_composite:.4f}, HD95: {avg_hd95:.2f}mm, R_low: {avg_r_low:.4f}")

                # ============ 最后保存模型（已在更新之后）============
                if is_new_best_composite:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_metric': best_metric,
                        'best_metric_epoch': best_metric_epoch,
                        'best_composite_score': best_composite_score,
                        'best_composite_epoch': best_composite_epoch,
                        'loss': epoch_loss / step,
                    }, os.path.join(model_dir, "best_composite_cascade_new" + change + ".pth"))
                    logger.info(
                        f"saved new best composite score model: {best_composite_score:.4f} at epoch {best_composite_epoch}")

                if is_new_best_dice:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_metric': best_metric,
                        'best_metric_epoch': best_metric_epoch,
                        'best_composite_score': best_composite_score,
                        'best_composite_epoch': best_composite_epoch,
                        'loss': epoch_loss / step,
                    }, os.path.join(model_dir, "best_dice_cascade_new" + change + ".pth"))
                    logger.info("saved new best metric model")

                if epoch + 1 == max_epochs:
                    # save the end of this train（完整检查点）
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_metric': best_metric,
                        'best_metric_epoch': best_metric_epoch,
                        'best_composite_score': best_composite_score,
                        'best_composite_epoch': best_composite_epoch,
                        'loss': epoch_loss / step,
                    }, os.path.join(model_dir, "end_cascade_new" + change + ".pth"))
                    logger.info(f"Saved final model to {model_dir}/end{change}.pth")

        # 每个epoch结束后打印时间
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s ({epoch_time / 60:.2f}min)")
        logger.info("-" * 60)

    total_time = time.time() - total_start
    logger.info(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, "
        f"best_composite_score: {best_composite_score:.4f} at epoch: {best_composite_epoch}, total time: {total_time}.")

    drawGraphs(loss_list, metric_values)

    print("training completed.")


