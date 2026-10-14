"""Training script for the improved cascade architecture."""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, decollate_batch
from monai.transforms import AsDiscrete
import logging
from datetime import datetime
from tqdm import tqdm

from models.cascade_both_new import CascadeBoth_New
from dataloader import dataload
from bdLoss import BoundaryLoss
from cal_composite import cal_composite_score

DATA_ROOT = "D:/wjh/ori_data"
WEIGHTS_DIR = "D:/wjh/3D_MRI/two_stage/weights"
LOG_DIR = "./logs"

BATCH_SIZE = 1
NUM_EPOCHS = 300
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
CROP_SIZE = 128

USE_TINY_COARSE = False  # Set True for faster but less accurate coarse stage
USE_DEEP_SUPERVISION = True  # Enable deep supervision
DOWNSAMPLE_FACTOR = 4  # Downsample factor for coarse stage (4x recommended)

COARSE_WEIGHT = 0.5  # Weight for coarse stage loss
DEEP_SUPERVISION_WEIGHTS = [1.0, 0.5, 0.25]  # Weights for [final, deep1, deep2]

TRAIN_AMP = True
VAL_AMP = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"train_cascade_new_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("Training New Cascade Architecture with 3D Coarse Stage")
logger.info(f"Deep Supervision: {USE_DEEP_SUPERVISION}")
logger.info(f"Tiny Coarse Model: {USE_TINY_COARSE}")
logger.info(f"Downsample Factor: {DOWNSAMPLE_FACTOR}")
logger.info("="*80)


class CombinedLoss(nn.Module):
    """Combined loss: BCE + Dice + Boundary."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(sigmoid=False, smooth_nr=1e-5, smooth_dr=1e-5)
        self.boundary = BoundaryLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        bd_loss = self.boundary(pred, target)
        return 0.5 * bce_loss + dice_loss + 0.5 * bd_loss


def compute_deep_supervision_loss(outputs, targets, loss_fn, weights=None):
    """Compute weighted loss for deep supervision."""
    if weights is None:
        weights = [1.0, 0.5, 0.25]
    
    total_loss = 0.0
    for i, (output, weight) in enumerate(zip(outputs, weights)):
        loss = loss_fn(output, targets)
        total_loss += weight * loss
    
    return total_loss


logger.info("Loading data...")
train_loader, val_loader = dataload(
    data_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    num_workers=4,
    crop_size=160  # Original resolution input
)
logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")


logger.info("Initializing model...")
model = CascadeBoth_New(
    num_out_classes=3,
    input_channels=4,
    coarse_base_channels=8,
    use_tiny_coarse=USE_TINY_COARSE,
    use_deep_supervision=USE_DEEP_SUPERVISION,
    downsample_factor=DOWNSAMPLE_FACTOR,
    testing=False
).to(device)

coarse_params = sum(p.numel() for p in model.coarse_unet.parameters()) / 1e6
fine_params = sum(p.numel() for p in model.fine_unet.parameters()) / 1e6
total_params = sum(p.numel() for p in model.parameters()) / 1e6

logger.info(f"Model Parameters:")
logger.info(f"  Coarse Network: {coarse_params:.2f}M")
logger.info(f"  Fine Network: {fine_params:.2f}M")
logger.info(f"  Total: {total_params:.2f}M")


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

logger.info(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
logger.info(f"Scheduler: CosineAnnealingWarmRestarts (T_0=50, eta_min=1e-6)")


coarse_loss_fn = DiceLoss(sigmoid=False, smooth_nr=1e-5, smooth_dr=1e-5)
fine_loss_fn = CombinedLoss()

dice_metric = DiceMetric(include_background=False, reduction="mean")
post_trans = AsDiscrete(threshold=0.5)


scaler = torch.cuda.amp.GradScaler(enabled=TRAIN_AMP)


def train_epoch(epoch):
    model.train()
    epoch_loss = 0.0
    epoch_coarse_loss = 0.0
    epoch_fine_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=TRAIN_AMP):
            coarse_output, fine_output = model(inputs)
            
            coarse_target = (labels.sum(dim=1, keepdim=True) > 0).float()
            loss_coarse = coarse_loss_fn(coarse_output, coarse_target)
            
            if isinstance(fine_output, list):
                loss_fine = compute_deep_supervision_loss(
                    fine_output, labels, fine_loss_fn, DEEP_SUPERVISION_WEIGHTS
                )
            else:
                loss_fine = fine_loss_fn(fine_output, labels)
            
            total_loss = COARSE_WEIGHT * loss_coarse + loss_fine
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += total_loss.item()
        epoch_coarse_loss += loss_coarse.item()
        epoch_fine_loss += loss_fine.item()
        
        progress_bar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'coarse': f'{loss_coarse.item():.4f}',
            'fine': f'{loss_fine.item():.4f}'
        })
    
    avg_loss = epoch_loss / len(train_loader)
    avg_coarse = epoch_coarse_loss / len(train_loader)
    avg_fine = epoch_fine_loss / len(train_loader)
    
    logger.info(f"Epoch {epoch} Training - Loss: {avg_loss:.4f}, Coarse: {avg_coarse:.4f}, Fine: {avg_fine:.4f}")
    
    return avg_loss


def validate(epoch):
    model.eval()
    val_loss = 0.0
    dice_metric.reset()
    
    all_composite_scores = []
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validating"):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            with torch.cuda.amp.autocast(enabled=VAL_AMP):
                coarse_output, fine_output = model(inputs)
                
                if isinstance(fine_output, list):
                    fine_output = fine_output[0]
                
                coarse_target = (labels.sum(dim=1, keepdim=True) > 0).float()
                loss_coarse = coarse_loss_fn(coarse_output, coarse_target)
                loss_fine = fine_loss_fn(fine_output, labels)
                total_loss = COARSE_WEIGHT * loss_coarse + loss_fine
                
                val_loss += total_loss.item()
            
            fine_output_discrete = [post_trans(i) for i in decollate_batch(fine_output)]
            dice_metric(y_pred=fine_output_discrete, y=labels)
            
            # Composite score
            try:
                pred_np = fine_output.cpu().numpy()
                label_np = labels.cpu().numpy()
                composite_score, accuracy_score, reliability_score, metrics_dict = cal_composite_score(
                    pred_np, label_np, voxel_spacing=[1.0, 1.0, 1.0]
                )
                all_composite_scores.append(composite_score)
            except Exception as e:
                logger.warning(f"Composite score calculation failed: {e}")
    
    # Validation statistics
    avg_val_loss = val_loss / len(val_loader)
    mean_dice = dice_metric.aggregate().item()
    mean_composite = sum(all_composite_scores) / len(all_composite_scores) if all_composite_scores else 0.0
    
    logger.info(f"Epoch {epoch} Validation - Loss: {avg_val_loss:.4f}, Dice: {mean_dice:.4f}, Composite: {mean_composite:.4f}")
    
    # Real-time console output for key metrics
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Validation Results:")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"  Composite Score: {mean_composite:.4f}")
    print(f"  Validation Loss: {avg_val_loss:.4f}")
    print(f"{'='*60}\n")
    
    return mean_dice, mean_composite, avg_val_loss


# ========== Main Training Loop ==========
best_dice = 0.0
best_composite = 0.0
best_dice_epoch = 0
best_composite_epoch = 0

logger.info("Starting training...")

for epoch in range(1, NUM_EPOCHS + 1):
    logger.info(f"\n{'='*80}")
    logger.info(f"Epoch {epoch}/{NUM_EPOCHS}")
    logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    logger.info(f"{'='*80}")
    
    # Training
    train_loss = train_epoch(epoch)
    
    # Validation
    if epoch % 1 == 0:  # Validate every epoch
        mean_dice, mean_composite, val_loss = validate(epoch)
        
        # Save best models
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_dice_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': mean_dice,
                'composite': mean_composite,
            }, os.path.join(WEIGHTS_DIR, f'cascade_new_best_dice_{timestamp}.pth'))
            logger.info(f"✓ Saved new best Dice model: {mean_dice:.4f}")
        
        if mean_composite > best_composite:
            best_composite = mean_composite
            best_composite_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': mean_dice,
                'composite': mean_composite,
            }, os.path.join(WEIGHTS_DIR, f'cascade_new_best_composite_{timestamp}.pth'))
            logger.info(f"✓ Saved new best Composite model: {mean_composite:.4f}")
    
    # Step scheduler
    scheduler.step()
    
    # Save checkpoint every 20 epochs
    if epoch % 20 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(WEIGHTS_DIR, f'cascade_new_checkpoint_epoch{epoch}_{timestamp}.pth'))
        logger.info(f"Checkpoint saved at epoch {epoch}")

logger.info("\n" + "="*80)
logger.info("Training completed!")
logger.info(f"Best Dice: {best_dice:.4f} at epoch {best_dice_epoch}")
logger.info(f"Best Composite: {best_composite:.4f} at epoch {best_composite_epoch}")
logger.info("="*80)
