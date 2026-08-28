"""
Quick inverse model training - writes results to file directly.
Uses the existing pre-trained forward model for evaluation.
Run with: C:\ProgramData\anaconda3\envs\mappo\python.exe run_inverse.py
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

LOG_PATH = 'C:/holo/inverse_run_log.txt'

def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')
    print(msg)
    sys.stdout.flush()

class StO(nn.Module):
    """Forward model (Structure-to-Optics) - must match saved model"""
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
        self.sig = nn.Sigmoid()

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

class OtS(nn.Module):
    """Inverse model (Optics-to-Structure)
    Input: 60-dim optical response
    Output: 6x6 structure matrix (via sigmoid)
    """
    def __init__(self, num_input=60):
        super().__init__()
        self.fc1 = nn.Linear(num_input, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 36 * 16)
        self.sig = nn.Sigmoid()
        self.conv1 = nn.Conv2d(16, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 1, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 6, 6)
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.sigmoid(self.bn2(self.conv2(x)))
        return x

if __name__ == '__main__':
    # Clear log
    with open(LOG_PATH, 'w') as f:
        f.write('')
    
    log("=" * 60)
    log("Inverse Model (OtS) Training - MAPPO env")
    log("=" * 60)
    
    path = 'C:/holo/4/'
    ch = 30
    batch_size = 64
    learning_rate = 1e-4
    epochs = 200
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    
    # Load the existing pre-trained forward model
    forward_model_path = path + '44033fe50l201.mdl'
    if not os.path.exists(forward_model_path):
        # Try to find newly trained model
        candidates = glob.glob(path + 'forward_best_*.mdl')
        if candidates:
            forward_model_path = candidates[0]
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
    net = OtS().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    log(f"Inverse model params: {sum(x.numel() for x in net.parameters())}")
    
    # Training config
    samples_per_epoch = 10000  # Generate this many random targets per epoch
    lastbatch = 1 if samples_per_epoch % batch_size != 0 else 0
    k = samples_per_epoch // batch_size + lastbatch
    
    log(f"Samples/epoch: {samples_per_epoch}, Batches: {k}, Epochs: {epochs}")
    log(f"Channels: {ch}, Optical dim: {ch*2}={ch*2}")
    log("=" * 60)
    
    best_tel = float('inf')
    best_path = None
    
    for epoch in range(epochs):
        losstem = 0
        net.train()
        
        for i in range(k):
            if lastbatch == 1 and i == k - 1:
                bs = samples_per_epoch % batch_size
            else:
                bs = batch_size
            
            # Generate random optical targets (sin/cos encoding of random phases)
            a = np.random.rand(bs, ch) * 2 * np.pi
            batchopr = np.zeros((bs, ch * 2))
            for ki in range(bs):
                for j in range(ch):
                    batchopr[ki, 2*j] = np.sin(a[ki, j]) / 2 + 0.5
                    batchopr[ki, 2*j+1] = np.cos(a[ki, j]) / 2 + 0.5
            batchopr = torch.tensor(batchopr, dtype=torch.float32).to(device)
            
            # Inverse model predicts structure, forward model evaluates
            pred_struct = net(batchopr)
            pred_opr = forward_model(pred_struct)
            loss = criterion(pred_opr, batchopr)
            
            optimizer.zero_grad()
            loss.backward()
            losstem += float(loss)
            optimizer.step()
        
        net.eval()
        
        if epoch % 10 == 0:
            trl = int((losstem / k) ** 0.5 * 1e3)
            
            with torch.no_grad():
                a = np.random.rand(200, ch) * 2 * np.pi
                testopr = np.zeros((200, ch * 2))
                for i in range(200):
                    for j in range(ch):
                        testopr[i, 2*j] = np.sin(a[i, j]) / 2 + 0.5
                        testopr[i, 2*j+1] = np.cos(a[i, j]) / 2 + 0.5
                plt_opr = torch.tensor(testopr, dtype=torch.float32).to(device)
                tel = criterion(forward_model(net(plt_opr)), plt_opr)
                tel = int(tel ** 0.5 * 1e3)
            
            log(f"E{epoch:4d} | TrainL: {trl} | TestL: {tel} (*1e-3)")
            
            if tel < best_tel:
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                best_tel = tel
                best_path = path + f'inverse_best_e{epoch}_l{tel}.mdl'
                torch.save(net, best_path)
                log(f"  >> Best model saved: {best_path}")
    
    log("=" * 60)
    log(f"Done! Best test loss: {best_tel} (*1e-3)")
    log(f"Best model: {best_path}")
    
    with open(path + "inverse_summary.txt", "w") as f:
        f.write(f"Best test loss: {best_tel} (*1e-3)\n")
        f.write(f"Best model: {best_path}\n")
        f.write(f"Forward model used: {forward_model_path}\n")
        f.write(f"Epochs: {epochs}\n")
