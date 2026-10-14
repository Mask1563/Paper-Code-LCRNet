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

    return dice


def hd95_func_paper(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)):
    """Standard HD95 without truncation or centroid fallback."""
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
            hd95_values[b] = 375.0
            continue

        pred_voxel_coords = np.argwhere(pred)
        gt_voxel_coords = np.argwhere(gt)

        pred_points = pred_voxel_coords * np.array(spacing)
        gt_points = gt_voxel_coords * np.array(spacing)

        try:
            from scipy.spatial import cKDTree

            gt_tree = cKDTree(gt_points)
            pred_tree = cKDTree(pred_points)

            dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
            dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)

            hd95_pred_to_gt = np.percentile(dist_pred_to_gt, 95)
            hd95_gt_to_pred = np.percentile(dist_gt_to_pred, 95)

            hd95_values[b] = max(hd95_pred_to_gt, hd95_gt_to_pred)

        except Exception as e:
            hd95_values[b] = 375.0

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


def cal_composite_score_for_paper(pc, wt_mask, gt_mask, alpha=0.3, beta=0.3, gamma=0.4, low_conf_threshold=0.85,
                                  low_conf_threshold_overlap=0.50, spacing=(1.0, 1.0, 1.0)):
    """Composite score for paper experiments (strict)."""

    # Channel order: [TC, WT, ET]

    # 计算TC的Dice系数（通道0）
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

    hd95_tc = hd95_func_paper(pred_tc_mask, gt_tc_mask, spacing=spacing)
    hd95_wt = hd95_func_paper(pred_wt_mask, gt_wt_mask, spacing=spacing)
    hd95_et = hd95_func_paper(pred_et_mask, gt_et_mask, spacing=spacing)
    hd95_avg = (hd95_tc + hd95_wt + hd95_et) / 3.0

    normalized_hd95 = torch.clamp(hd95_avg / 20.0, 0.0, 1.0)  # 归一化基准仍为20mm
    score_hd95 = torch.exp(-hd95_avg / 15.0)  # k=15mm，使得HD95=10mm时得分约0.513

    max_pc = pc.max(dim=1, keepdim=True)[0]  # 每个体素的最大类别概率 (B, 1, H, W, D)
    sum_pc = pc.sum(dim=1, keepdim=True)  # 三个通道概率之和（检测重叠区域）

    tumor_mask_fine = (pc[:, 1:2, ...] > 0.5).float()

    # Dual-threshold for single vs overlap regions.
    low_conf_single = ((max_pc < low_conf_threshold) & (sum_pc < 1.2)).float()
    low_conf_overlap = ((max_pc < low_conf_threshold_overlap) & (sum_pc >= 1.2)).float()
    low_conf_mask = (low_conf_single + low_conf_overlap).clamp(0.0, 1.0) * tumor_mask_fine

    tumor_voxels = tumor_mask_fine.sum(dim=(2, 3, 4))

    rlow = torch.where(
        tumor_voxels > 0,
        low_conf_mask.sum(dim=(2, 3, 4)) / tumor_voxels,
        torch.zeros_like(tumor_voxels)
    ).clamp(0.0, 1.0)

    wedge = pc[:, 1:2, ...]  # 精细分割的WT通道
    wedge_binary = (wedge > 0.5).float()
    edge_mask = extract_boundary_morphology(wedge_binary, thickness=2)

    wedge_clamped = wedge.clamp(1e-6, 1.0 - 1e-6)
    voxel_entropy = -(wedge_clamped * wedge_clamped.log2() + (1 - wedge_clamped) * (1 - wedge_clamped).log2())

    h_wedge = (voxel_entropy * edge_mask).sum(dim=(2, 3, 4)) / (edge_mask.sum(dim=(2, 3, 4)) + 1e-6)

    overlap_voxels = (low_conf_mask * edge_mask).sum(dim=(2, 3, 4))
    total_low_conf = low_conf_mask.sum(dim=(2, 3, 4))

    roverlap = torch.where(
        total_low_conf > 0,
        (overlap_voxels / total_low_conf).clamp(0.0, 1.0),
        torch.ones_like(total_low_conf)
    )

    conf_quality = 1.0 - rlow
    boundary_quality = 1.0 - h_wedge
    conf_distribution = roverlap

    rlow_weight = torch.clamp(rlow * 3.0, 0.0, 1.0)

    w1 = 0.4 + 0.1 * (1.0 - rlow_weight)
    w2 = 0.4 + 0.1 * (1.0 - rlow_weight)
    w3 = 0.2 * rlow_weight

    score_reliability = conf_quality * w1 + boundary_quality * w2 + conf_distribution * w3

    # 8. 计算精度得分
    Accuracy_score = alpha * dice_avg + beta * score_hd95

    # 9. 计算最终综合评分（CompositeScore）
    composite_score = Accuracy_score + gamma * score_reliability

    # 10. 将指标打包为字典（论文版本：包含三个区域的独立HD95）
    batch_size = composite_score.shape[0]
    metrics = {
        "Mean Dice": round(dice_avg.mean().item(), 4),
        "TC Dice": round(dice_tc.mean().item(), 4),
        "WT Dice": round(dice_wt.mean().item(), 4),
        "ET Dice": round(dice_et.mean().item(), 4),
        "HD95_TC": round(hd95_tc.mean().item(), 2),  # 新增：TC区域HD95
        "HD95_WT": round(hd95_wt.mean().item(), 2),  # 新增：WT区域HD95
        "HD95_ET": round(hd95_et.mean().item(), 2),  # 新增：ET区域HD95
        "HD95_avg": round(hd95_avg.mean().item(), 2),
        "Normalized_HD95": round(normalized_hd95.mean().item(), 4),
        "Score_HD95": round(score_hd95.mean().item(), 4),
        "R_low": round(rlow.mean().item(), 4),
        "H_Wedge": round(h_wedge.mean().item(), 4),
        "R_overlap": round(roverlap.mean().item(), 4),
        "Reliability_Score": round(score_reliability.mean().item(), 4),
        "composite_score": round(composite_score.mean().item(), 4),
        # 逐样本原始值，便于严格模式下筛选案例（避免仅有均值信息）
        "dice_avg_per_sample": [float(x) for x in dice_avg.detach().cpu().flatten()],
        "comp_per_sample": [float(x) for x in composite_score.detach().cpu().flatten()],
    }

    return composite_score, Accuracy_score, score_reliability, metrics
