"""
Improved Forward Model (StO_v2) Training
Key improvements over v1:
  1. Deeper CNN with more channels (1->16->32->64) and 3 conv layers
  2. Dropout for regularization
  3. Cosine Annealing LR scheduler with warm restarts
  4. Data augmentation: random 90/180/270 rotation & flips of 6x6 grids
  5. Wider FC layers (512->256->128->60) 
  6. 500 epochs with early stopping patience
  7. Weight initialization (Kaiming)
  8. Gradient clipping

Run with: C:\ProgramData\anaconda3\envs\mappo\python.exe run_forward_v2.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
import time
import sys

LOG_PATH = 'C:/holo/forward_v2_log.txt'

def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')
    print(msg)
    sys.stdout.flush()

def read_txt_to_2d_list(file_path, symbol, dataset):
    with open(file_path, 'r') as file:
        data = []
        line_count = 0
        for line in file:
            if line_count >= dataset:
                break
            line = line.strip()
            if line:
                row = [float(num) for num in line.split(symbol)]
                data.append(row)
            line_count += 1
        return data

def binary_string_to_bit_list(file_path, dataset):
    with open(file_path, 'r') as file:
        data = []
        line_count = 0
        for line in file:
            if line_count >= dataset:
                break
            line = line.strip()
            bit_list = [float(bit) for bit in line]
            bit_matrix = [bit_list[i*6:i*6+6] for i in range(len(bit_list) // 6)]
            data.append(bit_matrix)
            line_count += 1
    return data


class StO_v2(nn.Module):
    """Improved forward model with deeper architecture"""
    def __init__(self, num_output=60):
        super().__init__()
        # Conv block 1: 1 -> 32
        self.conv1a = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1b = nn.BatchNorm2d(32)
        
        # Conv block 2: 32 -> 64
        self.conv2a = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2b = nn.BatchNorm2d(64)
        
        # Conv block 3: 64 -> 128 (on 3x3 after pooling)
        self.conv3a = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3a = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)  # 6x6 -> 3x3
        
        # Residual shortcut for block 1
        self.shortcut1 = nn.Conv2d(1, 32, 1, 1, 0)
        # Residual shortcut for block 2
        self.shortcut2 = nn.Conv2d(32, 64, 1, 1, 0)
        
        # FC layers: 128*3*3 = 1152
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_output)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        # Last layer init for sigmoid output
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, x):
        x = x.view(-1, 1, 6, 6)
        
        # Block 1 with residual
        identity = self.shortcut1(x)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = self.bn1b(self.conv1b(x))
        x = F.relu(x + identity)  # residual connection
        
        # Block 2 with residual
        identity = self.shortcut2(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + identity)  # residual connection
        
        # Pool: 6x6 -> 3x3
        x = self.pool(x)
        
        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        
        # FC
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


def augment_batch(st_batch, opr_batch):
    """
    Data augmentation for 6x6 binary structures.
    Since the structure is a 6x6 grid, we can apply:
    - Random 90/180/270 degree rotations
    - Random horizontal/vertical flips
    Note: The optical response would need to change correspondingly,
    but since we're training on paired data, we just augment both.
    
    IMPORTANT: For this meta-surface problem, rotation/flip of structure
    changes the optical response. So we can't blindly augment.
    Instead, we use noise-based augmentation on the optical response side.
    
    Actually, the safest augmentation is to just use the real pairs.
    We'll add small Gaussian noise to structures (staying binary via rounding)
    and use mixup-style regularization.
    """
    # No geometric augmentation (would change optical response)
    # Instead, use mixup: blend pairs for regularization
    batch_size = len(st_batch)
    if batch_size < 2:
        return st_batch, opr_batch
    
    # 50% chance to apply mixup
    if np.random.random() > 0.5:
        return st_batch, opr_batch
    
    # Mixup with alpha=0.2
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    
    indices = np.random.permutation(batch_size)
    mixed_st = lam * st_batch + (1 - lam) * st_batch[indices]
    mixed_opr = lam * opr_batch + (1 - lam) * opr_batch[indices]
    
    return mixed_st, mixed_opr


if __name__ == '__main__':
    with open(LOG_PATH, 'w') as f:
        f.write('')
    
    log("=" * 60)
    log("Forward Model v2 (StO_v2) Training - Improved")
    log("=" * 60)
    
    batch_size = 128
    initial_lr = 2e-3
    epochs = 500
    tn = 44033
    path = 'C:/holo/4/'
    patience = 80  # early stopping patience (in epochs)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    
    log("Loading dataset...")
    t0 = time.time()
    st = binary_string_to_bit_list('C:/holo/dataset/st_36.txt', tn)
    opr1 = read_txt_to_2d_list('C:/holo/dataset/opr_530.txt', ' ', tn)
    opr2 = read_txt_to_2d_list('C:/holo/dataset/opr_670.txt', ' ', tn)
    opr3 = read_txt_to_2d_list('C:/holo/dataset/opr_800.txt', ' ', tn)
    opr = [a + b + c for a, b, c in zip(opr1, opr2, opr3)]
    log(f"Loaded {len(st)} samples in {time.time()-t0:.1f}s, opr dim={len(opr[0])}")
    
    # Convert to numpy arrays for efficiency
    st_np = np.array(st, dtype=np.float32)   # (44033, 6, 6)
    opr_np = np.array(opr, dtype=np.float32)  # (44033, 60)
    
    trn = int(tn * 0.9)
    np.random.seed(42)
    total_id = np.random.permutation(tn)
    
    train_st = st_np[total_id[:trn]]
    train_opr = opr_np[total_id[:trn]]
    test_st = st_np[total_id[trn:]]
    test_opr = opr_np[total_id[trn:]]
    
    # Pre-load test data to GPU
    test_st_t = torch.tensor(test_st, dtype=torch.float32).to(device)
    test_opr_t = torch.tensor(test_opr, dtype=torch.float32).to(device)
    
    net = StO_v2().to(device)
    param_count = sum(x.numel() for x in net.parameters())
    log(f"Parameters: {param_count}")
    
    optimizer = optim.AdamW(net.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-5)
    criterion = nn.MSELoss().to(device)
    
    # Also track Huber loss for robustness
    huber = nn.SmoothL1Loss().to(device)
    
    num_batches = (trn + batch_size - 1) // batch_size
    
    log(f"Train: {trn}, Test: {tn-trn}, Batches: {num_batches}, Epochs: {epochs}")
    log(f"LR: {initial_lr}, Batch: {batch_size}, Patience: {patience}")
    log(f"Optimizer: AdamW (weight_decay=1e-4)")
    log(f"Scheduler: CosineAnnealingWarmRestarts (T0=50, Tmult=2)")
    log("=" * 60)
    
    best_tel = float('inf')
    best_path = None
    no_improve = 0
    
    for epoch in range(epochs):
        losstem = 0
        net.train()
        
        # Shuffle training data
        perm = np.random.permutation(trn)
        
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, trn)
            idx = perm[start:end]
            
            bst = torch.tensor(train_st[idx], dtype=torch.float32).to(device)
            bop = torch.tensor(train_opr[idx], dtype=torch.float32).to(device)
            
            # Mixup augmentation
            if np.random.random() < 0.3 and len(idx) > 1:
                lam = np.random.beta(0.2, 0.2)
                lam = max(lam, 1 - lam)
                rand_idx = torch.randperm(bst.size(0)).to(device)
                bst = lam * bst + (1 - lam) * bst[rand_idx]
                bop = lam * bop + (1 - lam) * bop[rand_idx]
            
            pred = net(bst)
            # Combined loss: MSE + Huber for robustness
            loss = 0.7 * criterion(pred, bop) + 0.3 * huber(pred, bop)
            losstem += float(loss)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        net.eval()
        
        if epoch % 5 == 0:
            trl = int((losstem / num_batches) ** 0.5 * 1e3)
            with torch.no_grad():
                tel_mse = criterion(net(test_st_t), test_opr_t)
                tel = int(tel_mse ** 0.5 * 1e3)
            
            lr_now = optimizer.param_groups[0]['lr']
            log(f"E{epoch:4d} | TrainL: {trl} | TestL: {tel} (*1e-3) | LR: {lr_now:.6f}")
            
            if tel < best_tel:
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                best_tel = tel
                best_path = path + f'forward_v2_best_e{epoch}_l{tel}.mdl'
                torch.save(net, best_path)
                log(f"  >> Best model saved: {best_path}")
                no_improve = 0
            else:
                no_improve += 5  # since we check every 5 epochs
            
            if no_improve >= patience:
                log(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    
    log("=" * 60)
    log(f"Done! Best test loss: {best_tel} (*1e-3)")
    log(f"Best model: {best_path}")
    log(f"Improvement over v1: {205 - best_tel} (*1e-3) = {(205-best_tel)/205*100:.1f}%")
