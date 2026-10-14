
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice_coeff(pred_mask, gt_mask):
    """Dice coefficient for binary masks."""
    pred_mask = (pred_mask > 0.5).float()
    gt_mask = (gt_mask > 0.5).float()

    intersection = (pred_mask * gt_mask).sum(dim=(2, 3, 4))
    sum_volumes = pred_mask.sum(dim=(2, 3, 4)) + gt_mask.sum(dim=(2, 3, 4))  # |A| + |B|

    dice = (2 * intersection / (sum_volumes + 1e-6)).clamp(0.0, 1.0)

    # 移除了调试信息

    return dice


def hd95_func(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)):
    """HD95 in mm with fallback for empty/tiny masks."""
    pred_mask = (pred_mask > 0.5).float()
    gt_mask = (gt_mask > 0.5).float()

    batch_size = pred_mask.shape[0]

    hd95_values = torch.zeros(batch_size)

    for b in range(batch_size):
        pred = pred_mask[b, 0].cpu().numpy()
        gt = gt_mask[b, 0].cpu().numpy()

        pred_voxels = np.sum(pred)
        gt_voxels = np.sum(gt)

        if pred_voxels == 0 or gt_voxels == 0:
            hd95_values[b] = 20.0  # 设置为最大可接受误差
            continue

        pred_voxel_coords = np.argwhere(pred)  # 体素坐标
        gt_voxel_coords = np.argwhere(gt)  # 体素坐标
        
        if pred_voxels < 10 or gt_voxels < 10:
            pred_centroid = pred_voxel_coords.mean(axis=0) * np.array(spacing)
            gt_centroid = gt_voxel_coords.mean(axis=0) * np.array(spacing)
            centroid_dist = np.linalg.norm(pred_centroid - gt_centroid)
            hd95_values[b] = min(centroid_dist * 2.0, 20.0)  # 质心距离×2作为估计，上限20mm
            continue

        pred_points = pred_voxel_coords * np.array(spacing)
        gt_points = gt_voxel_coords * np.array(spacing)

        pred_min = pred_points.min(axis=0)
        pred_max = pred_points.max(axis=0)
        gt_min = gt_points.min(axis=0)
        gt_max = gt_points.max(axis=0)

        overlap_min = np.maximum(pred_min, gt_min)
        overlap_max = np.minimum(pred_max, gt_max)
        overlap_volume = np.prod(np.maximum(0, overlap_max - overlap_min))

        pred_volume = np.prod(pred_max - pred_min)
        gt_volume = np.prod(gt_max - gt_min)
        total_volume = pred_volume + gt_volume - overlap_volume

        overlap_ratio = overlap_volume / total_volume if total_volume > 0 else 0

        if overlap_ratio < 0.05:
            pred_centroid = pred_voxel_coords.mean(axis=0) * np.array(spacing)
            gt_centroid = gt_voxel_coords.mean(axis=0) * np.array(spacing)
            centroid_dist = np.linalg.norm(pred_centroid - gt_centroid)
            hd95_values[b] = min(centroid_dist, 20.0)
            continue

        try:
            from scipy.spatial.distance import directed_hausdorff
            from scipy.spatial import cKDTree

            gt_tree = cKDTree(gt_points)
            pred_tree = cKDTree(pred_points)

            dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
            dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)

            hd95_pred_to_gt = np.percentile(dist_pred_to_gt, 95)
            hd95_gt_to_pred = np.percentile(dist_gt_to_pred, 95)

            hd95_values[b] = max(hd95_pred_to_gt, hd95_gt_to_pred)

            hd95_values[b] = min(hd95_values[b], 20.0)

        except Exception as e:
            hd95_values[b] = 20.0

    return hd95_values.to(pred_mask.device)


def extract_boundary_morphology(mask, thickness=2):
    """Extract boundary mask using erosion."""
    from scipy.ndimage import binary_erosion
    batch_size = mask.shape[0]
    boundary_batch = []
    
    for b in range(batch_size):
        mask_np = (mask[b, 0] > 0.5).cpu().numpy()
        if mask_np.sum() == 0:
            boundary_batch.append(torch.zeros_like(mask[b, 0]))
            continue

        eroded = binary_erosion(mask_np, iterations=thickness)
        boundary = (mask_np.astype(float) - eroded.astype(float))
        boundary_batch.append(torch.from_numpy(boundary).float().to(mask.device))
    
    return torch.stack(boundary_batch, dim=0).unsqueeze(1)


def cal_composite_score(pc, wt_mask, gt_mask, alpha=0.6, beta=0.25, gamma=0.15, low_conf_threshold=0.70, low_conf_threshold_overlap=0.50, spacing=(1.0, 1.0, 1.0)):
    """Composite score for [TC, WT, ET] channels."""

    # Channel order: [TC, WT, ET]
    pred_tc_mask = (pc[:, 0:1, ...] > 0.5).float()
    gt_tc_mask = gt_mask[:, 0:1, ...]  # TC通道
    dice_tc = dice_coeff(pred_tc_mask, gt_tc_mask)

    pred_wt_mask = (pc[:, 1:2, ...] > 0.5).float()
    gt_wt_mask = gt_mask[:, 1:2, ...]  # WT通道
    dice_wt = dice_coeff(pred_wt_mask, gt_wt_mask)

    pred_et_mask = (pc[:, 2:3, ...] > 0.5).float()
    gt_et_mask = gt_mask[:, 2:3, ...]  # ET通道
    dice_et = dice_coeff(pred_et_mask, gt_et_mask)

    dice_avg = (dice_tc + dice_wt + dice_et) / 3.0

    hd95_tc = hd95_func(pred_tc_mask, gt_tc_mask, spacing=spacing)
    hd95_wt = hd95_func(pred_wt_mask, gt_wt_mask, spacing=spacing)
    hd95_et = hd95_func(pred_et_mask, gt_et_mask, spacing=spacing)
    hd95_avg = (hd95_tc + hd95_wt + hd95_et) / 3.0

    score_hd95 = torch.exp(-hd95_avg / 15.0)  # k=15mm

    normalized_hd95 = torch.clamp(hd95_avg / 20.0, 0.0, 1.0)  # HD95最大可接受误差为20mm

    max_pc = pc.max(dim=1, keepdim=True)[0]  # 每个体素的最大类别概率 (B, 1, H, W, D)
    sum_pc = pc.sum(dim=1, keepdim=True)  # 三个通道概率之和（检测重叠区域）

    tumor_mask_fine = (pc[:, 1:2, ...] > 0.5).float()  # 精细分割WT通道定义的肿瘤区域

    # Dual-threshold for single vs overlap regions.
    low_conf_single = ((max_pc < low_conf_threshold) & (sum_pc < 1.2)).float()  # 单区域：使用较严格阈值（0.70）
    low_conf_overlap = ((max_pc < low_conf_threshold_overlap) & (sum_pc >= 1.2)).float()  # 多区域：使用较宽松阈值（0.50）
    low_conf_mask = (low_conf_single + low_conf_overlap).clamp(0.0, 1.0) * tumor_mask_fine

    tumor_voxels = tumor_mask_fine.sum(dim=(2, 3, 4))  # 精细分割肿瘤区域体素数

    rlow = torch.where(
        tumor_voxels > 0, 
        low_conf_mask.sum(dim=(2, 3, 4)) / tumor_voxels, 
        torch.zeros_like(tumor_voxels)
    ).clamp(0.0, 1.0)

    # Boundary entropy from WT channel.
    wedge = pc[:, 1:2, ...]  # 精细分割的WT通道（整体肿瘤概率）
    wedge_binary = (wedge > 0.5).float()  # 二值化用于边界提取
    edge_mask = extract_boundary_morphology(wedge_binary, thickness=2)  # 从精细分割提取边界

    wedge_clamped = wedge.clamp(1e-6, 1.0 - 1e-6)  # 避免log(0)
    voxel_entropy = -(wedge_clamped * wedge_clamped.log2() + (1 - wedge_clamped) * (1 - wedge_clamped).log2())

    h_wedge = (voxel_entropy * edge_mask).sum(dim=(2, 3, 4)) / (edge_mask.sum(dim=(2, 3, 4)) + 1e-6)

    overlap_voxels = (low_conf_mask * edge_mask).sum(dim=(2, 3, 4))  # 边界内的低置信度体素数
    total_low_conf = low_conf_mask.sum(dim=(2, 3, 4))  # 肿瘤内总低置信度体素数

    roverlap = torch.where(
        total_low_conf > 0,
        (overlap_voxels / total_low_conf).clamp(0.0, 1.0),
        torch.ones_like(total_low_conf)  # 没有低置信度体素时，给予满分
    )

    conf_quality = 1.0 - rlow  # 整体置信度质量
    boundary_quality = 1.0 - h_wedge  # 边界清晰度
    conf_distribution = roverlap  # 低置信度分布合理性

    rlow_weight = torch.clamp(rlow * 3.0, 0.0, 1.0)  # rlow的影响权重

    w1 = 0.4 + 0.1 * (1.0 - rlow_weight)  # 置信度权重：0.4-0.5（rlow低时增加）
    w2 = 0.4 + 0.1 * (1.0 - rlow_weight)  # 边界权重：0.4-0.5（rlow低时增加）
    w3 = 0.2 * rlow_weight  # 分布权重：0-0.2（rlow低时减少）
    
    score_reliability = conf_quality * w1 + boundary_quality * w2 + conf_distribution * w3

    Accuracy_score = alpha * dice_avg + beta * score_hd95

    composite_score = Accuracy_score + gamma * score_reliability

    batch_size = composite_score.shape[0]
    metrics = {
        "Mean Dice": round(dice_avg.mean().item(), 4),
        "TC Dice": round(dice_tc.mean().item(), 4),
        "WT Dice": round(dice_wt.mean().item(), 4),
        "ET Dice": round(dice_et.mean().item(), 4),
        "HD95_avg": round(hd95_avg.mean().item(), 2),
        "Normalized_HD95": round(normalized_hd95.mean().item(), 4),
        "Score_HD95": round(score_hd95.mean().item(), 4),
        "R_low": round(rlow.mean().item(), 4),
        "H_Wedge": round(h_wedge.mean().item(), 4),
        "R_overlap": round(roverlap.mean().item(), 4),
        "Reliability_Score": round(score_reliability.mean().item(), 4),
        "composite_score": round(composite_score.mean().item(), 4),
    }

    return composite_score, Accuracy_score, score_reliability, metrics
