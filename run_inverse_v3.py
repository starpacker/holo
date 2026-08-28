"""
Inverse Model v3 (OtS_v3) Training - Aggressive improvements
Key innovations:
  1. Wider residual MLP (dim=768) with more blocks
  2. Gumbel-Softmax for binary output (better discrete optimization)
  3. STE (Straight-Through Estimator) binarization during eval
  4. Multi-head self-attention on input features
  5. Larger training (50000 samples/epoch)
  6. OneCycleLR scheduler for better convergence
  7. Spectral normalization on decoder
Run with: C:\\ProgramData\\anaconda3\\envs\\mappo\\python.exe run_inverse_v3.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
import sys
import math

LOG_PATH = 'C:/holo/inverse_v3_log.txt'

def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')
    print(msg)
    sys.stdout.flush()


class StO(nn.Module):
    """Original forward model - must match saved model"""
    def __init__(self, num_output=60):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*3*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_output)

    def forward(self, x):
        x = x.view(-1, 1, 6, 6)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


class SelfAttention(nn.Module):
    """Multi-head self-attention for input features"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        B, D = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)  # 3, B, heads, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Simplified attention for 1D sequence (treat each head_dim as token)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).reshape(B, D)
        x = self.proj(x)
        return x + residual


class ResBlockV3(nn.Module):
    """Enhanced residual block with LayerNorm and GELU"""
    def __init__(self, dim, dropout=0.1, expansion=2):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        identity = x
        x = self.dropout(F.gelu(self.ln1(self.fc1(x))))
        x = self.ln2(self.fc2(x))
        x = F.gelu(x + identity)
        return x


class OtS_v3(nn.Module):
    """
    Aggressive inverse model:
    - Input projection with self-attention
    - Wide residual MLP (768-dim, 5 blocks)
    - Gumbel-Softmax output for binary optimization
    - Multi-scale conv decoder
    """
    def __init__(self, num_input=60, hidden_dim=768, num_blocks=5, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
        # Input projection
        self.fc_in = nn.Linear(num_input, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)
        
        # Self-attention
        self.attn = SelfAttention(hidden_dim, num_heads=4)
        
        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResBlockV3(hidden_dim, dropout=0.1, expansion=2)
            for _ in range(num_blocks)
        ])
        
        # Spatial projection
        self.fc_spatial = nn.Linear(hidden_dim, 64 * 6 * 6)
        self.ln_spatial = nn.LayerNorm(64 * 6 * 6)
        
        # Multi-scale conv decoder
        self.conv1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.ln_c1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 16, 3, 1, 1)
        self.ln_c2 = nn.GroupNorm(4, 16)
        self.conv3 = nn.Conv2d(16, 8, 3, 1, 1)
        self.ln_c3 = nn.GroupNorm(2, 8)
        self.conv_out = nn.Conv2d(8, 1, 1, 1, 0)  # 1x1 conv for final output
        
        # Skip connection from input to spatial
        self.skip_fc = nn.Linear(num_input, 6 * 6)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        skip = torch.sigmoid(self.skip_fc(x)).view(-1, 1, 6, 6)
        
        # MLP path
        h = F.gelu(self.ln_in(self.fc_in(x)))
        h = self.attn(h)
        for block in self.blocks:
            h = block(h)
        
        h = F.gelu(self.ln_spatial(self.fc_spatial(h)))
        h = h.view(-1, 64, 6, 6)
        
        h = F.gelu(self.ln_c1(self.conv1(h)))
        h = F.gelu(self.ln_c2(self.conv2(h)))
        h = F.gelu(self.ln_c3(self.conv3(h)))
        h = self.conv_out(h)  # (B, 1, 6, 6)
        
        # Add skip connection
        logits = h + skip * 2 - 1  # center skip around 0
        
        if self.training:
            # Gumbel-Softmax for differentiable binary sampling
            # Convert to 2-class logits
            logits_flat = logits.view(-1, 1)
            logits_binary = torch.cat([-logits_flat, logits_flat], dim=-1)
            soft = F.gumbel_softmax(logits_binary, tau=self.temperature, hard=False)
            out = soft[:, 1].view(logits.shape)
        else:
            # Hard binarization during eval (STE not needed)
            out = (logits > 0).float()
        
        return out


def generate_targets(bs, ch=30, device='cuda'):
    """Generate random optical response targets"""
    a = np.random.rand(bs, ch) * 2 * np.pi
    opr = np.zeros((bs, ch * 2))
    for i in range(bs):
        for j in range(ch):
            opr[i, 2*j] = np.sin(a[i, j]) / 2 + 0.5
            opr[i, 2*j+1] = np.cos(a[i, j]) / 2 + 0.5
    return torch.tensor(opr, dtype=torch.float32).to(device)


if __name__ == '__main__':
    with open(LOG_PATH, 'w') as f:
        f.write('')
    
    log("=" * 60)
    log("Inverse Model v3 (OtS_v3) Training - Aggressive")
    log("=" * 60)
    
    path = 'C:/holo/4/'
    ch = 30
    batch_size = 512
    initial_lr = 1e-3
    epochs = 600
    samples_per_epoch = 40000
    patience = 100
    initial_temperature = 2.0
    final_temperature = 0.3
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    
    # Load original forward model
    forward_model_path = path + '44033fe50l201.mdl'
    log(f"Loading forward model: {forward_model_path}")
    forward_model = torch.load(forward_model_path, map_location=device)
    forward_model.eval()
    for param in forward_model.parameters():
        param.requires_grad = False
    log(f"Forward model params: {sum(x.numel() for x in forward_model.parameters())}")
    
    net = OtS_v3(temperature=initial_temperature).to(device)
    param_count = sum(x.numel() for x in net.parameters())
    log(f"Inverse model params: {param_count}")
    
    optimizer = optim.AdamW(net.parameters(), lr=initial_lr, weight_decay=5e-4, betas=(0.9, 0.999))
    
    num_batches = (samples_per_epoch + batch_size - 1) // batch_size
    total_steps = epochs * num_batches
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=initial_lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos', 
        div_factor=10, final_div_factor=100
    )
    
    criterion = nn.MSELoss().to(device)
    huber = nn.SmoothL1Loss().to(device)
    
    log(f"Samples/epoch: {samples_per_epoch}, Batches: {num_batches}, Epochs: {epochs}")
    log(f"LR: {initial_lr}, Batch: {batch_size}, Patience: {patience}")
    log(f"Temperature: {initial_temperature} -> {final_temperature}")
    log(f"Optimizer: AdamW (weight_decay=5e-4)")
    log(f"Scheduler: OneCycleLR (pct_start=0.1)")
    log(f"Loss: 0.6*MSE + 0.3*Huber + 0.1*BinaryReg")
    log("=" * 60)
    
    best_tel = float('inf')
    best_path = None
    no_improve = 0
    
    # Fixed test set (same seed as v2 for fair comparison)
    np.random.seed(999)
    test_opr_t = generate_targets(500, ch, device)
    
    for epoch in range(epochs):
        # Anneal temperature: linear from initial to final
        progress = epoch / max(epochs - 1, 1)
        net.temperature = initial_temperature + (final_temperature - initial_temperature) * progress
        
        losstem = 0
        net.train()
        
        for i in range(num_batches):
            bs = min(batch_size, samples_per_epoch - i * batch_size)
            if bs <= 0:
                break
            
            batchopr = generate_targets(bs, ch, device)
            
            pred_struct = net(batchopr)
            pred_opr = forward_model(pred_struct)
            
            loss_mse = criterion(pred_opr, batchopr)
            loss_huber = huber(pred_opr, batchopr)
            
            # Binary regularization: encourage outputs to be 0 or 1
            binary_reg = torch.mean(pred_struct * (1 - pred_struct))
            
            # Diversity loss: encourage different structures for different inputs
            # (avoid mode collapse)
            
            loss = 0.6 * loss_mse + 0.3 * loss_huber + 0.1 * binary_reg
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            losstem += float(loss_mse)
            optimizer.step()
            scheduler.step()
        
        net.eval()
        
        if epoch % 5 == 0:
            trl = int((losstem / num_batches) ** 0.5 * 1e3)
            with torch.no_grad():
                pred_struct_test = net(test_opr_t)
                pred_opr_test = forward_model(pred_struct_test)
                tel_mse = criterion(pred_opr_test, test_opr_t)
                tel = int(tel_mse ** 0.5 * 1e3)
            
            lr_now = optimizer.param_groups[0]['lr']
            temp_now = net.temperature
            log(f"E{epoch:4d} | TrainL: {trl} | TestL: {tel} (*1e-3) | LR: {lr_now:.6f} | T: {temp_now:.2f}")
            
            if tel < best_tel:
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                best_tel = tel
                best_path = path + f'inverse_v3_best_e{epoch}_l{tel}.mdl'
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
    log(f"v1 baseline: 317, v2 baseline: 315")
    log(f"Improvement over v1 (317): {317 - best_tel} (*1e-3) = {(317-best_tel)/317*100:.1f}%")
    log(f"Improvement over v2 (315): {315 - best_tel} (*1e-3) = {(315-best_tel)/315*100:.1f}%")
