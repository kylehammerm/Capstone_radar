"""
Micro-Doppler CNN for AWR2944 ASL Recognition
==============================================
Designed for tensors from realtime_rdi_ra_md_awr2944_v4.py

Tensor characteristics:
- Shape: (128, 200) - Doppler bins × Time frames
- Values: dB scale, clipped at -10 dB floor
- Range: Typically -10 dB to +40 dB (motion) 
- Structure: Doppler profile across 200 time steps

Author: Your advisor demo ready
"""

import logging
from pathlib import Path
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
import json

# ==============================
# LOGGING CONFIGURATION
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# File handler for persistent logs
fh = logging.FileHandler('cnn_detailed.log', encoding='utf-8')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# ==============================
# CONFIGURATION
# ==============================
DATASET_DIR = "cnn_tensors"
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 5e-4
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
NUM_WORKERS = 0  # Windows compatibility

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"=== Micro-Doppler ASL CNN ===")
print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_DIR}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Train ratio: {TRAIN_RATIO*100}%")
print("="*50)

# ==============================
# DATASET
# ==============================
class MicroDopplerDataset(Dataset):
    """Dataset for v4-generated Micro-Doppler tensors"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_names = []
        
        # Get all class folders
        classes = sorted([d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.class_names = classes
        
        print(f"\nLoading dataset from {root_dir}")
        print(f"Found classes: {classes}")
        
        # Load all .npy files
        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            npy_files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
            
            for fname in npy_files:
                filepath = os.path.join(cls_path, fname)
                self.samples.append(filepath)
                self.labels.append(self.class_to_idx[cls])
            
            print(f"  {cls}: {len(npy_files)} samples")
        
        print(f"Total samples: {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load tensor (128, 200)
        tensor = np.load(self.samples[idx]).astype(np.float32)
        
        # Normalize to [0, 1] range for stable training
        # Values are typically -10 to +40 dB
        tensor = (tensor + 10) / 50  # -10dB -> 0, +40dB -> 1
        tensor = np.clip(tensor, 0, 1)
        
        # Add channel dimension: (1, 128, 200)
        tensor = tensor[np.newaxis, :, :]
        
        # Convert to torch tensor
        tensor = torch.from_numpy(tensor)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return tensor, label, self.samples[idx]  # Return filename for debugging

# ==============================
# CNN ARCHITECTURE
# ==============================
class MicroDopplerCNN(nn.Module):
    """
    Specialized CNN for 128×200 Micro-Doppler spectrograms.
    Designed for AWR2944 radar data with -10 dB floor.
    """
    
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        
        # Input: (batch, 1, 128, 200)
        
        # Block 1: Doppler feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 7), padding=(2, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(5, 7), padding=(2, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (64, 100)
            nn.Dropout2d(0.1)
        )
        
        # Block 2: Temporal pattern extraction
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (32, 50)
            nn.Dropout2d(0.2)
        )
        
        # Block 3: High-level features
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),  # (4, 8) - fixed size
            nn.Dropout2d(0.2)
        )
        
        # Calculate flattened size
        self.feature_size = 128 * 4 * 8
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """Extract features for visualization"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return x

# ==============================
# TRAINING ENGINE
# ==============================
class Trainer:
    def __init__(self, model, device, save_dir='checkpoints'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        self.best_val_acc = 0
        self.best_epoch = 0
        self.patience_counter = 0
        
    def train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc='Training', leave=False)
        for batch_idx, (inputs, targets, _) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(loader), 100. * correct / total
    
    def validate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        all_probs = []
        all_filenames = []
        
        with torch.no_grad():
            for inputs, targets, filenames in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store for metrics
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_filenames.extend(filenames)
        
        return (total_loss / len(loader), 
                100. * correct / total,
                all_targets,
                all_predictions,
                np.array(all_probs),
                all_filenames)
    
    def fit(self, train_loader, val_loader, epochs, criterion, optimizer, scheduler):
        logger.info("="*60)
        logger.info("TRAINING STARTED")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        logger.info(f"Device: {self.device}")
        logger.info("="*60)
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, _, _, _, _ = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # LOG EVERY EPOCH
            logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | ")
            
            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc)
                    if optimizer.param_groups[0]['lr'] < current_lr:
                        logger.info(f"  → Learning rate reduced to {optimizer.param_groups[0]['lr']:.2e}")
                else:
                    scheduler.step()
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_names': train_loader.dataset.class_names,
                }
                torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
                logger.info(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= 5:  # Log when patience is building
                    logger.info(f"  Patience: {self.patience_counter}/15")
            
            # Early stopping
            if self.patience_counter >= 15:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        logger.info("="*60)
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")
        logger.info("="*60)
        
        # Load best model
        checkpoint = torch.load(os.path.join(self.save_dir, 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.model

# ==============================
# VISUALIZATION
# ==============================
class Visualizer:
    def __init__(self, class_names, save_dir='figures'):
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_history(self, trainer):
        """Plot training curves - FIXED SPACING"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)
        
        epochs = range(1, len(trainer.train_accs) + 1)
        
        # Accuracy
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(epochs, trainer.train_accs, 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, trainer.val_accs, 'r-', label='Validation', linewidth=2)
        ax1.axhline(y=trainer.best_val_acc, color='g', linestyle='--', 
                    label=f'Best: {trainer.best_val_acc:.1f}%')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold', pad=15)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        ax1.tick_params(labelsize=10)
        
        # Loss
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(epochs, trainer.train_losses, 'b-', label='Train', linewidth=2)
        ax2.plot(epochs, trainer.val_losses, 'r-', label='Validation', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.set_title('Training & Validation Loss', fontsize=12, fontweight='bold', pad=15)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)
        
        # Learning rate
        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(epochs[:len(trainer.learning_rates)], trainer.learning_rates, 'g-', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Learning Rate', fontsize=11)
        ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        ax3.tick_params(labelsize=10)
        
        # Accuracy distribution
        ax4 = plt.subplot(gs[1, 1])
        ax4.hist([trainer.train_accs, trainer.val_accs], bins=15, 
                label=['Train', 'Validation'], alpha=0.7, 
                color=['blue', 'red'], edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Accuracy (%)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Accuracy Distribution', fontsize=12, fontweight='bold', pad=15)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=10)
        
        plt.suptitle('Training History', fontsize=14, fontweight='bold', y=1.02)
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, probs=None):
        """Plot confusion matrix with statistics - FIXED SPACING"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Better figure layout
        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 0.9, 1], wspace=0.3)
        
        # 1. Confusion matrix - larger
        ax1 = plt.subplot(gs[0])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax1, cbar_kws={'label': 'Count', 'shrink': 0.8},
                annot_kws={'size': 11})
        ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold', pad=15)
        ax1.set_ylabel('True Label', fontsize=11)
        ax1.set_xlabel('Predicted Label', fontsize=11)
        ax1.tick_params(labelsize=10)
        
        # 2. Class-wise accuracy
        ax2 = plt.subplot(gs[1])
        class_acc = []

        # Convert to numpy arrays for boolean indexing
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        for i, class_name in enumerate(self.class_names):
            total = np.sum(y_true_arr == i)
            if total > 0:
                correct = np.sum((y_true_arr == i) & (y_pred_arr == i))
                acc = 100 * correct / total
            else:
                acc = 0
            class_acc.append(acc)

        bars = ax2.barh(self.class_names, class_acc, color='skyblue', edgecolor='navy', linewidth=0.5)
        ax2.set_xlabel('Accuracy (%)', fontsize=11)
        ax2.set_title('Class-wise Accuracy', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlim([0, 100])
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.tick_params(labelsize=10)

        # Add accuracy labels
        for bar, acc in zip(bars, class_acc):
            ax2.text(min(acc + 3, 97), bar.get_y() + bar.get_height()/2, 
                    f'{acc:.1f}%', va='center', ha='left', fontsize=10, fontweight='bold')
        
        # 3. Confidence histogram
        ax3 = plt.subplot(gs[2])
        if probs is not None:
            max_probs = np.max(probs, axis=1)
            correct_conf = max_probs[np.array(y_true) == np.array(y_pred)]
            incorrect_conf = max_probs[np.array(y_true) != np.array(y_pred)]
            
            ax3.hist(correct_conf, bins=20, alpha=0.7, label=f'Correct (n={len(correct_conf)})',
                    color='green', edgecolor='black', linewidth=0.5)
            ax3.hist(incorrect_conf, bins=20, alpha=0.7, label=f'Incorrect (n={len(incorrect_conf)})',
                    color='red', edgecolor='black', linewidth=0.5)
            ax3.set_xlabel('Confidence Score', fontsize=11)
            ax3.set_ylabel('Count', fontsize=11)
            ax3.set_title('Prediction Confidence', fontsize=13, fontweight='bold', pad=15)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(labelsize=10)
        
        overall_acc = 100 * np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
        plt.suptitle(f'Model Performance Summary (Overall Accuracy: {overall_acc:.1f}%)', 
                    fontsize=14, fontweight='bold', y=1.05)
        #plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        return class_acc
    
    def plot_predictions(self, model, loader, device, num_samples=12):
        """Plot sample predictions with spectrograms - FIXED SPACING"""
        model.eval()
        
        # Get batch
        inputs, targets, filenames = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, 1)
        
        # Setup plot with better spacing
        n_cols = 4
        n_rows = (num_samples + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(18, 4.5 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.3)
        
        for idx in range(min(num_samples, len(inputs))):
            row = idx // n_cols
            col = idx % n_cols
            ax = plt.subplot(gs[row, col])
            
            # Get spectrogram (denormalize for visualization)
            spec = inputs[idx, 0].cpu().numpy()
            spec = spec * 50 - 10  # Back to dB scale
            
            # Plot with better aspect ratio
            im = ax.imshow(spec, aspect='auto', cmap='viridis',
                        extent=[0, 200, 64, -64], vmin=-10, vmax=40)
            
            # Labels
            true_label = self.class_names[targets[idx].item()]
            pred_label = self.class_names[predictions[idx].item()]
            confidence = confidences[idx].item()
            correct = true_label == pred_label
            
            # Title with better formatting
            title_color = 'green' if correct else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                        color=title_color, fontsize=10, fontweight='bold',
                        pad=10)
            
            ax.set_xlabel('Time (frames)', fontsize=8)
            ax.set_ylabel('Doppler (bin)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            # Add filename short
            fname = os.path.basename(filenames[idx])
            if len(fname) > 20:
                fname = fname[:15] + '...'
            ax.text(0.5, -0.2, fname, transform=ax.transAxes,
                fontsize=7, ha='center', va='top', style='italic')
        
        # Hide empty subplots
        for idx in range(num_samples, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(plt.subplot(gs[row, col]))
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Power (dB)', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        plt.suptitle('Micro-Doppler Spectrograms: Predictions vs Ground Truth', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.savefig(os.path.join(self.save_dir, 'sample_predictions.png'), 
                    dpi=150, bbox_inches='tight')
        plt.show()
        

# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(description='Micro-Doppler CNN for ASL Recognition')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    args = parser.parse_args()
    
    # Create dataset
    print("\n=== Loading Dataset ===")
    dataset = MicroDopplerDataset(DATASET_DIR)
    
    # Split dataset
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # Preserve class_names attribute
    train_dataset.class_names = dataset.class_names
    val_dataset.class_names = dataset.class_names

    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Validation: {len(val_dataset)} samples ({(1-TRAIN_RATIO)*100:.0f}%)")
    print(f"  Classes: {dataset.class_names}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    # Create model
    num_classes = len(dataset.class_names)
    model = MicroDopplerCNN(num_classes, dropout_rate=0.5).to(DEVICE)
    
    print(f"\n=== Model Architecture ===")
    print(f"Input: (batch, 1, 128, 200)")
    print(f"Output: {num_classes} classes")
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {train_params:,}")
    
    # Training
    if args.train:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                             patience=5, min_lr=1e-6)
        
        # Trainer
        trainer = Trainer(model, DEVICE)
        model = trainer.fit(train_loader, val_loader, args.epochs, 
                           criterion, optimizer, scheduler)
        
        # Visualize training
        viz = Visualizer(dataset.class_names)
        viz.plot_training_history(trainer)
        
        # Save training summary
        summary = {
            'best_val_acc': trainer.best_val_acc,
            'best_epoch': trainer.best_epoch,
            'total_epochs': len(trainer.train_accs),
            'train_accs': trainer.train_accs,
            'val_accs': trainer.val_accs,
            'final_train_acc': trainer.train_accs[-1],
            'final_val_acc': trainer.val_accs[-1],
            'class_names': dataset.class_names,
            'num_classes': num_classes,
            'total_samples': len(dataset),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'model_params': total_params,
        }
        
        with open('training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining summary saved to training_summary.json")
    
    # Testing
    if args.test:
        print("\n=== Testing Model ===")
        
        # Load best model
        checkpoint_path = os.path.join('checkpoints', 'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint['epoch']}")
            print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            print("No trained model found. Train first with --train")
            return
        
        # Evaluate
        trainer = Trainer(model, DEVICE)
        test_loss, test_acc, targets, predictions, probs, filenames = \
            trainer.validate(val_loader, nn.CrossEntropyLoss())
        
        print(f"\n=== Test Results ===")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Samples: {len(targets)}")
        
        # Classification report
        print(f"\n=== Classification Report ===")
        print(classification_report(targets, predictions, 
                                   target_names=dataset.class_names,
                                   digits=3))
        
        # After classification report, add:
        print(f"\n=== Detailed Test Summary ===")
        logger.info("="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Samples: {len(targets)}")
        logger.info("\nClass-wise Accuracy:")
        for i, cls in enumerate(dataset.class_names):
            total = np.sum(np.array(targets) == i)
            correct = np.sum((np.array(targets) == i) & (np.array(predictions) == i))
            acc = 100 * correct / total if total > 0 else 0
            logger.info(f"  {cls:8s}: {acc:5.2f}% ({correct}/{total})")

        # Confusion matrix as text
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(targets, predictions)
        cm_text = "\n     " + " ".join([f"{c:6s}" for c in dataset.class_names])
        for i, row in enumerate(cm):
            cm_text += f"\n{dataset.class_names[i]:4s}: " + " ".join([f"{val:6d}" for val in row])
        logger.info(cm_text)

        # Misclassified samples
        logger.info("\nMisclassified Samples:")
        misclassified_idx = np.where(np.array(targets) != np.array(predictions))[0]
        for idx in misclassified_idx[:10]:  # Show first 10
            true = dataset.class_names[targets[idx]]
            pred = dataset.class_names[predictions[idx]]
            conf = np.max(probs[idx])
            fname = os.path.basename(filenames[idx])
            logger.info(f"  {fname:30s} | True: {true:6s} | Pred: {pred:6s} | Conf: {conf:.3f}")

        if len(misclassified_idx) > 10:
            logger.info(f"  ... and {len(misclassified_idx) - 10} more")
        # Visualize results
        viz = Visualizer(dataset.class_names)
        class_acc = viz.plot_confusion_matrix(targets, predictions, probs)
        viz.plot_predictions(model, val_loader, DEVICE, num_samples=12)
        
        # Save results
        results = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'class_names': dataset.class_names,
            'class_accuracies': class_acc,
            'confusion_matrix': confusion_matrix(targets, predictions).tolist(),
            'total_tested': len(targets),
            'correct_predictions': int(np.sum(np.array(targets) == np.array(predictions))),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest results saved to test_results.json")

if __name__ == "__main__":
    main()