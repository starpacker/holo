"""
Quick forward model training - writes results to file directly.
Run with: C:\ProgramData\anaconda3\envs\mappo\python.exe run_forward.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import torch.nn.functional as F
import time
import sys

LOG_PATH = 'C:/holo/forward_run_log.txt'

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

class StO(nn.Module):
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

if __name__ == '__main__':
    # Clear log
    with open(LOG_PATH, 'w') as f:
        f.write('')
    
    log("=" * 60)
    log("Forward Model (StO) Training - MAPPO env")
    log("=" * 60)
    
    batch_size = 64
    learning_rate = 1e-3
    epochs = 200
    tn = 44033
    path = 'C:/holo/4/'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    
    log("Loading dataset...")
    t0 = time.time()
    st = binary_string_to_bit_list('C:/holo/dataset/st_36.txt', tn)
    opr1 = read_txt_to_2d_list('C:/holo/dataset/opr_530.txt', ' ', tn)
    opr2 = read_txt_to_2d_list('C:/holo/dataset/opr_670.txt', ' ', tn)
    opr3 = read_txt_to_2d_list('C:/holo/dataset/opr_800.txt', ' ', tn)
    opr = [a+b+c for a,b,c in zip(opr1, opr2, opr3)]
    log(f"Loaded {len(st)} samples in {time.time()-t0:.1f}s, opr dim={len(opr[0])}")
    
    trn = int(tn * 0.9)
    total_id = list(range(tn))
    np.random.seed(42)
    np.random.shuffle(total_id)
    
    net = StO().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    log(f"Parameters: {sum(x.numel() for x in net.parameters())}")
    
    trainst = [st[i] for i in total_id[:trn]]
    trainopr = [opr[i] for i in total_id[:trn]]
    testst = [st[i] for i in total_id[trn:]]
    testopr = [opr[i] for i in total_id[trn:]]
    
    tdi = list(range(trn))
    lastbatch = 1 if trn % batch_size != 0 else 0
    k = trn // batch_size + lastbatch
    
    log(f"Train: {trn}, Test: {tn-trn}, Batches: {k}, Epochs: {epochs}")
    log("=" * 60)
    
    best_tel = float('inf')
    best_path = None
    
    for epoch in range(epochs):
        losstem = 0
        np.random.shuffle(tdi)
        net.train()
        
        for i in range(k):
            if lastbatch == 1 and i == k - 1:
                bs = trn % batch_size
            else:
                bs = batch_size
            
            bst, bop = [], []
            for j in range(bs):
                idx = tdi[i * batch_size + j] if not (lastbatch == 1 and i == k-1) else tdi[(k-1)*batch_size + j]
                if i * batch_size + j < trn:
                    bst.append(trainst[tdi[i * batch_size + j] if i * batch_size + j < trn else 0])
                    bop.append(trainopr[tdi[i * batch_size + j] if i * batch_size + j < trn else 0])
            
            bop_t = torch.tensor(bop, dtype=torch.float32).to(device)
            bst_t = torch.tensor(bst, dtype=torch.float32).to(device)
            
            loss = criterion(net(bst_t), bop_t)
            losstem += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        net.eval()
        
        if epoch % 10 == 0:
            trl = int((losstem / k) ** 0.5 * 1e3)
            with torch.no_grad():
                test_st_t = torch.tensor(testst, dtype=torch.float32).to(device)
                test_opr_t = torch.tensor(testopr, dtype=torch.float32).to(device)
                tel = int(criterion(net(test_st_t), test_opr_t) ** 0.5 * 1e3)
            
            log(f"E{epoch:4d} | TrainL: {trl} | TestL: {tel} (*1e-3)")
            
            if tel < best_tel:
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                best_tel = tel
                best_path = path + f'forward_best_e{epoch}_l{tel}.mdl'
                torch.save(net, best_path)
                log(f"  >> Best model saved: {best_path}")
    
    log("=" * 60)
    log(f"Done! Best test loss: {best_tel} (*1e-3)")
    log(f"Best model: {best_path}")
    
    with open(path + "forward_summary.txt", "w") as f:
        f.write(f"Best test loss: {best_tel} (*1e-3)\n")
        f.write(f"Best model: {best_path}\n")
        f.write(f"Epochs: {epochs}\n")
