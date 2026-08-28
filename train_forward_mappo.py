"""
Forward Model Training Script (for mappo conda environment)
Trains StO (Structure-to-Optics) model: 6x6 binary structure -> 60-dim optical response
Uses the mappo conda env: C:\ProgramData\anaconda3\envs\mappo\python.exe

Usage:
    C:\ProgramData\anaconda3\envs\mappo\python.exe train_forward_mappo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import torch.nn.functional as F
import time

# ==================== Data Loading ====================
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

# ==================== Model ====================
class StO(nn.Module):
    """Structure-to-Optics Forward Model
    Input: 6x6 binary matrix (meta-surface unit cell structure)
    Output: 60-dim optical response (sin/cos of phase at 3 wavelengths x 10 polarizations)
    """
    def __init__(self, num_output=60):
        super().__init__()
        cv1_out_channels = 8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cv1_out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cv1_out_channels)

        cv2_out_channels = 16
        self.conv2 = nn.Conv2d(in_channels=cv1_out_channels, out_channels=cv2_out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(cv2_out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 3 * 3, 256)
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

# ==================== Training ====================
if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 500          # Reduced for preliminary results
    tn = 44033            # Total number of samples
    path = 'C:/holo/4/'
    
    # Tracking
    telmin = 500
    epochmin = 0
    lastbatch = 0
    
    print("=" * 60)
    print("Forward Model (StO) Training")
    print("=" * 60)
    
    # Load dataset
    print("Loading dataset...")
    t0 = time.time()
    st = binary_string_to_bit_list('C:/holo/dataset/st_36.txt', tn)
    opr1 = read_txt_to_2d_list('C:/holo/dataset/opr_530.txt', ' ', tn)
    opr2 = read_txt_to_2d_list('C:/holo/dataset/opr_670.txt', ' ', tn)
    opr3 = read_txt_to_2d_list('C:/holo/dataset/opr_800.txt', ' ', tn)
    opr = [item1 + item2 + item3 for item1, item2, item3 in zip(opr1, opr2, opr3)]
    print(f"Dataset loaded in {time.time()-t0:.1f}s")
    print(f"  Structure samples: {len(st)}, shape per sample: 6x6 binary")
    print(f"  Optical response dim: {len(opr[0])} (3 wavelengths x 10 polarizations x 2 sin/cos)")
    
    # Train/test split
    trn = int(tn * 0.9)
    total_id = list(range(tn))
    tdi = list(range(trn))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(total_id)
    
    # Setup log file
    log_file = path + "forward_loss_mappo.txt"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,test_loss\n")
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    net = StO().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    total_params = sum(x.numel() for x in net.parameters())
    print(f"Total parameters: {total_params}")
    
    # Prepare data
    trainst = copy.deepcopy([st[i] for i in total_id[:trn]])
    trainopr = copy.deepcopy([opr[i] for i in total_id[:trn]])
    testst = copy.deepcopy([st[i] for i in total_id[trn:]])
    testopr = copy.deepcopy([opr[i] for i in total_id[trn:]])
    
    if trn % batch_size != 0:
        lastbatch = 1
    k = trn // batch_size + lastbatch
    
    print(f"\nTraining samples: {trn}, Test samples: {tn - trn}")
    print(f"Batch size: {batch_size}, Batches per epoch: {k}")
    print(f"Epochs: {epochs}")
    print("=" * 60)
    
    best_test_loss = float('inf')
    best_model_path = None
    
    for epoch in range(epochs):
        losstem = 0
        np.random.shuffle(tdi)
        net.train()
        
        batchst, batchopr = [], []
        for i in range(k):
            if lastbatch == 1 and i == k - 1:
                bs = trn % batch_size
            else:
                bs = batch_size
            
            for j in range(bs):
                idx = i * batch_size + j if not (lastbatch == 1 and i == k - 1) else (k - 1) * batch_size + j
                if idx < trn:
                    batchst.append(trainst[tdi[idx]])
                    batchopr.append(trainopr[tdi[idx]])
            
            batchopr_t = torch.tensor(batchopr, dtype=torch.float32).to(device)
            batchst_t = torch.tensor(batchst, dtype=torch.float32).to(device)
            
            loss = criterion(net(batchst_t), batchopr_t)
            losstem += float(loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batchst, batchopr = [], []
        
        net.eval()
        
        if epoch % 10 == 0:
            trl = int((losstem / k) ** 0.5 * 1e3)
            with torch.no_grad():
                plt_st = torch.tensor(testst, dtype=torch.float32).to(device)
                plt_opr = torch.tensor(testopr, dtype=torch.float32).to(device)
                tel = int(criterion(net(plt_st), plt_opr) ** 0.5 * 1e3)
            
            with open(log_file, "a") as f:
                f.write(f"{epoch},{trl},{tel}\n")
            
            print(f"Epoch {epoch:4d} | Train Loss: {trl} | Test Loss: {tel} (*1e-3)")
            
            if tel < best_test_loss:
                # Remove old best model
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_test_loss = tel
                best_model_path = path + f'forward_best_e{epoch}_l{tel}.mdl'
                torch.save(net, best_model_path)
                print(f"  >> New best model saved: {best_model_path}")
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best test loss: {best_test_loss} (*1e-3)")
    print(f"Best model: {best_model_path}")
    print("=" * 60)
    
    # Save final summary
    with open(path + "forward_training_summary.txt", "w") as f:
        f.write(f"Forward Model Training Summary\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Best test loss: {best_test_loss} (*1e-3)\n")
        f.write(f"Best model: {best_model_path}\n")
        f.write(f"Total parameters: {total_params}\n")
