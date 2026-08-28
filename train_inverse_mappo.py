"""
Inverse Model Training Script (for mappo conda environment)
Trains OtS (Optics-to-Structure) model: 60-dim optical response -> 6x6 binary structure
Uses a pre-trained forward model (StO) as a differentiable evaluator.
Uses the mappo conda env: C:\ProgramData\anaconda3\envs\mappo\python.exe

Usage:
    C:\ProgramData\anaconda3\envs\mappo\python.exe train_inverse_mappo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
import time
import glob

# ==================== Models ====================
class StO(nn.Module):
    """Structure-to-Optics Forward Model (must match the trained forward model)"""
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
    """Optics-to-Structure Inverse Model
    Input: 60-dim optical response
    Output: 6x6 binary structure (via sigmoid)
    """
    def __init__(self, num_input=60):
        super().__init__()
        self.fc1 = nn.Linear(num_input, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 36 * 16)
        self.sig = nn.Sigmoid()

        cv1_out_channels = 8
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=cv1_out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cv1_out_channels)

        cv2_out_channels = 1
        self.conv2 = nn.Conv2d(in_channels=cv1_out_channels, out_channels=cv2_out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(cv2_out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 6, 6)
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.sigmoid(self.bn2(self.conv2(x)))
        return x

# ==================== Training ====================
if __name__ == '__main__':
    path = 'C:/holo/4/'
    
    # Hyperparameters
    ch = 30                # Number of optical channels (phases)
    batch_size = 64
    learning_rate = 1e-4
    epochs = 500           # Reduced for preliminary results
    tn = 100000            # Number of random training samples per epoch cycle
    
    # Tracking
    telmin = 500
    epochmin = 0
    lastbatch = 0
    
    print("=" * 60)
    print("Inverse Model (OtS) Training")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load pre-trained forward model
    # Try to find the best forward model
    forward_model_path = None
    
    # First try our newly trained model
    candidates = glob.glob(path + 'forward_best_*.mdl')
    if candidates:
        forward_model_path = candidates[0]
    
    # Fall back to original model
    if forward_model_path is None:
        original_model = path + '44033fe50l201.mdl'
        if os.path.exists(original_model):
            forward_model_path = original_model
    
    if forward_model_path is None:
        print("ERROR: No forward model found! Train forward model first.")
        exit(1)
    
    print(f"Loading forward model: {forward_model_path}")
    forward_model = torch.load(forward_model_path, map_location=device)
    forward_model.eval()
    # Freeze forward model
    for param in forward_model.parameters():
        param.requires_grad = False
    
    # Create inverse model
    net = OtS().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    
    total_params = sum(x.numel() for x in net.parameters())
    print(f"Inverse model parameters: {total_params}")
    print(f"Forward model parameters: {sum(x.numel() for x in forward_model.parameters())}")
    
    # Setup log file
    log_file = path + "inverse_loss_mappo.txt"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,test_loss\n")
    
    trainnum = int(0.9 * tn)
    tdi = list(range(trainnum))
    
    if trainnum % batch_size != 0:
        lastbatch = 1
    k = trainnum // batch_size + lastbatch
    
    print(f"\nTraining with {trainnum} random optical targets per epoch")
    print(f"Batch size: {batch_size}, Batches per epoch: {k}")
    print(f"Epochs: {epochs}")
    print(f"Optical channels (phases): {ch}")
    print("=" * 60)
    
    best_test_loss = float('inf')
    best_model_path = None
    
    for epoch in range(epochs):
        losstem = 0
        np.random.shuffle(tdi)
        net.train()
        
        for i in range(k):
            if lastbatch == 1 and i == k - 1:
                bs = trainnum % batch_size
            else:
                bs = batch_size
            
            # Generate random optical targets (sin/cos encoding of random phases)
            a = np.random.rand(bs, ch) * 2 * np.pi
            batchopr = np.zeros((bs, ch * 2))
            for ki in range(bs):
                for j in range(ch):
                    batchopr[ki, 2 * j] = np.sin(a[ki, j]) / 2 + 0.5
                    batchopr[ki, 2 * j + 1] = np.cos(a[ki, j]) / 2 + 0.5
            batchopr = torch.tensor(batchopr, dtype=torch.float32).to(device)
            
            # Forward pass: target -> inverse_model -> structure -> forward_model -> predicted_target
            predicted_structure = net(batchopr)
            predicted_target = forward_model(predicted_structure)
            loss = criterion(predicted_target, batchopr)
            
            optimizer.zero_grad()
            loss.backward()
            losstem += float(loss)
            optimizer.step()
        
        net.eval()
        
        if epoch % 10 == 0:
            trl = int((losstem / k) ** 0.5 * 1e3)
            
            # Test with random samples
            with torch.no_grad():
                a = np.random.rand(100, ch) * 2 * np.pi
                testopr = np.zeros((100, ch * 2))
                for i in range(100):
                    for j in range(ch):
                        testopr[i, 2 * j] = np.sin(a[i, j]) / 2 + 0.5
                        testopr[i, 2 * j + 1] = np.cos(a[i, j]) / 2 + 0.5
                plt_opr = torch.tensor(testopr, dtype=torch.float32).to(device)
                tel = criterion(forward_model(net(plt_opr)), plt_opr)
                tel = int(tel ** 0.5 * 1e3)
            
            with open(log_file, "a") as f:
                f.write(f"{epoch},{trl},{tel}\n")
            
            print(f"Epoch {epoch:4d} | Train Loss: {trl} | Test Loss: {tel} (*1e-3)")
            
            if tel < best_test_loss:
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_test_loss = tel
                best_model_path = path + f'inverse_best_e{epoch}_l{tel}.mdl'
                torch.save(net, best_model_path)
                print(f"  >> New best model saved: {best_model_path}")
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best test loss: {best_test_loss} (*1e-3)")
    print(f"Best model: {best_model_path}")
    print("=" * 60)
    
    # Save final summary
    with open(path + "inverse_training_summary.txt", "w") as f:
        f.write(f"Inverse Model Training Summary\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Best test loss: {best_test_loss} (*1e-3)\n")
        f.write(f"Best model: {best_model_path}\n")
        f.write(f"Forward model used: {forward_model_path}\n")
        f.write(f"Total parameters: {total_params}\n")
