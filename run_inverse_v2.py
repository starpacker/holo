"""
Improved Inverse Model (OtS_v2) Training
Key improvements over v1:
  1. Much wider FC layers with skip connections (ResNet-style MLP)
  2. Deeper convolutional decoder with more channels
  3. Dropout regularization
  4. Cosine Annealing LR scheduler with warm restarts
  5. Larger batch size and more samples per epoch
  6. Gradient clipping
  7. AdamW optimizer with weight decay
  8. Combined MSE + Huber loss
  9. Binary regularization loss to encourage 0/1 outputs
  10. Uses the improved forward model v2

Run with: C:\ProgramData\anaconda3\envs\mappo\python.exe run_inverse_v2.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
import time
import sys
import glob

LOG_PATH = 'C:/holo/inverse_v2_log.txt'

def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')
    print(msg)
    sys.stdout.flush()


class StO_v2(nn.Module):
    """Improved forward model - must match saved architecture"""
    def __init__(self, num_output=60):
        super().__init__()
        self.conv1a = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1b = nn.BatchNorm2d(32)
        self.conv2a = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2b = nn.BatchNorm2d(64)
        self.conv3a = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3a = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.shortcut1 = nn.Conv2d(1, 32, 1, 1, 0)
        self.shortcut2 = nn.Conv2d(32, 64, 1, 1, 0)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_output)

    def forward(self, x):
        x = x.view(-1, 1, 6, 6)
        identity = self.shortcut1(x)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = self.bn1b(self.conv1b(x))
        x = F.relu(x + identity)
        identity = self.shortcut2(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + identity)
        x = self.pool(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


class ResBlock(nn.Module):
    """Residual block for MLP"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        identity = x
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.bn2(self.fc2(x))
        x = F.relu(x + identity)
        return x


class OtS_v2(nn.Module):
    """Improved inverse model with residual MLP + deeper conv decoder
    Input: 60-dim optical response
    Output: 6x6 structure matrix
    """
    def __init__(self, num_input=60):
        super().__init__()
        # Encoder MLP with residual blocks
        self.fc_in = nn.Linear(num_input, 512)
        self.bn_in = nn.BatchNorm1d(512)
        
        self.res1 = ResBlock(512, dropout=0.15)
        self.res2 = ResBlock(512, dropout=0.15)
        self.res3 = ResBlock(512, dropout=0.15)
        
        # Project to spatial feature map: 512 -> 32*6*6 = 1152
        self.fc_spatial = nn.Linear(512, 32 * 6 * 6)
        self.bn_spatial = nn.BatchNorm1d(32 * 6 * 6)
        
        # Conv decoder: 32 -> 16 -> 8 -> 1
        self.conv1 = nn.Conv2d(32, 16, 3, 1, 1)
        self.bn_c1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 3, 1, 1)
        self.bn_c2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 1, 3, 1, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # MLP encoder with residual blocks
        x = F.relu(self.bn_in(self.fc_in(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Project to spatial
        x = F.relu(self.bn_spatial(self.fc_spatial(x)))
        x = x.view(-1, 32, 6, 6)
        
        # Conv decoder
        x = F.relu(self.bn_c1(self.conv1(x)))
        x = F.relu(self.bn_c2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x


if __name__ == '__main__':
    with open(LOG_PATH, 'w') as f:
        f.write('')
    
    log("=" * 60)
    log("Inverse Model v2 (OtS_v2) Training - Improved")
    log("=" * 60)
    
    path = 'C:/holo/4/'
    ch = 30
    batch_size = 256
    initial_lr = 5e-4
    epochs = 500
    samples_per_epoch = 20000
    patience = 80
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    
    # Find the best forward model - prefer v2
    forward_model_path = None
    v2_candidates = sorted(glob.glob(path + 'forward_v2_best_*.mdl'))
    v1_candidates = sorted(glob.glob(path + 'forward_best_*.mdl'))
    original = path + '44033fe50l201.mdl'
    
    if v2_candidates:
        forward_model_path = v2_candidates[-1]
    elif v1_candidates:
        forward_model_path = v1_candidates[-1]
    elif os.path.exists(original):
        forward_model_path = original
    else:
        log("ERROR: No forward model found!")
        sys.exit(1)
    
    log(f"Loading forward model: {forward_model_path}")
    forward_model = torch.load(forward_model_path, map_location=device)
    forward_model.eval()
    for param in forward_model.parameters():
        param.requires_grad = False
    log(f"Forward model params: {sum(x.numel() for x in forward_model.parameters())}")
    
    # Create inverse model
    net = OtS_v2().to(device)
    param_count = sum(x.numel() for x in net.parameters())
    log(f"Inverse model params: {param_count}")
    
    optimizer = optim.AdamW(net.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-5)
    criterion = nn.MSELoss().to(device)
    huber = nn.SmoothL1Loss().to(device)
    
    num_batches = (samples_per_epoch + batch_size - 1) // batch_size
    
    log(f"Samples/epoch: {samples_per_epoch}, Batches: {num_batches}, Epochs: {epochs}")
    log(f"LR: {initial_lr}, Batch: {batch_size}, Patience: {patience}")
    log(f"Optimizer: AdamW (weight_decay=1e-4)")
    log(f"Scheduler: CosineAnnealingWarmRestarts (T0=50, Tmult=2)")
    log(f"Loss: 0.7*MSE + 0.3*Huber + 0.01*BinaryReg")
    log("=" * 60)
    
    best_tel = float('inf')
    best_path = None
    no_improve = 0
    
    # Fixed test set for consistent evaluation
    np.random.seed(999)
    a_test = np.random.rand(500, ch) * 2 * np.pi
    test_opr = np.zeros((500, ch * 2))
    for i in range(500):
        for j in range(ch):
            test_opr[i, 2*j] = np.sin(a_test[i, j]) / 2 + 0.5
            test_opr[i, 2*j+1] = np.cos(a_test[i, j]) / 2 + 0.5
    test_opr_t = torch.tensor(test_opr, dtype=torch.float32).to(device)
    
    for epoch in range(epochs):
        losstem = 0
        net.train()
        
        for i in range(num_batches):
            bs = min(batch_size, samples_per_epoch - i * batch_size)
            if bs <= 0:
                break
            
            # Generate random optical targets
            a = np.random.rand(bs, ch) * 2 * np.pi
            batchopr = np.zeros((bs, ch * 2))
            for ki in range(bs):
                for j in range(ch):
                    batchopr[ki, 2*j] = np.sin(a[ki, j]) / 2 + 0.5
                    batchopr[ki, 2*j+1] = np.cos(a[ki, j]) / 2 + 0.5
            batchopr = torch.tensor(batchopr, dtype=torch.float32).to(device)
            
            # Forward pass
            pred_struct = net(batchopr)
            pred_opr = forward_model(pred_struct)
            
            # Main loss: how well does the predicted structure reproduce the target optics
            loss_mse = criterion(pred_opr, batchopr)
            loss_huber = huber(pred_opr, batchopr)
            
            # Binary regularization: encourage output to be closer to 0 or 1
            # This helps because actual structures are binary
            binary_reg = torch.mean(pred_struct * (1 - pred_struct))
            
            loss = 0.7 * loss_mse + 0.3 * loss_huber + 0.01 * binary_reg
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            losstem += float(loss_mse)  # track MSE for comparison
            optimizer.step()
        
        scheduler.step()
        net.eval()
        
        if epoch % 5 == 0:
            trl = int((losstem / num_batches) ** 0.5 * 1e3)
            
            with torch.no_grad():
                tel_mse = criterion(forward_model(net(test_opr_t)), test_opr_t)
                tel = int(tel_mse ** 0.5 * 1e3)
            
            lr_now = optimizer.param_groups[0]['lr']
            log(f"E{epoch:4d} | TrainL: {trl} | TestL: {tel} (*1e-3) | LR: {lr_now:.6f}")
            
            if tel < best_tel:
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                best_tel = tel
                best_path = path + f'inverse_v2_best_e{epoch}_l{tel}.mdl'
                torch.save(net, best_path)
                log(f"  >> Best model saved: {best_path}")
                no_improve = 0
            else:
                no_improve += 5
            
            if no_improve >= patience:
                log(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    
    log("=" * 60)
    log(f"Done! Best test loss: {best_tel} (*1e-3)")
    log(f"Best model: {best_path}")
    log(f"Forward model used: {forward_model_path}")
    log(f"Improvement over v1: {317 - best_tel} (*1e-3) = {(317-best_tel)/317*100:.1f}%")
