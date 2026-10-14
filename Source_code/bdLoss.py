import torch.nn as nn
import torch
import torch.nn.functional as F
class BoundaryLoss(nn.Module):
    """Boundary loss from Bokhovkin et al., 2019."""

    def __init__(self, theta0=3, theta=5, apply_sigmoid=True):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.apply_sigmoid = apply_sigmoid

    def forward(self, pred, gt):

        n, c, _, _, _ = pred.shape

        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)
        one_hot_gt = gt
        gt_b = F.max_pool3d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool3d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        gt_b_ext = F.max_pool3d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool3d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        BF1 = 2 * P * R / (P + R + 1e-7)

        loss = torch.mean(1 - BF1)

        return loss