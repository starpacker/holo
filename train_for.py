import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import torch.nn.functional as F

def read_txt_to_2d_list(file_path, symbol,dataset):
    with open(file_path, 'r') as file:
        data = []
        line_count = 0
        for line in file:
            if line_count >= dataset:
                break
            line = line.strip()
            if line:
                # 假设数值由逗号分隔
                row = [float(num) for num in line.split(symbol)]
                data.append(row)
            line_count += 1
        return data
def binary_string_to_bit_list(file_path,dataset):
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
        cv1_out_channals=8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cv1_out_channals, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cv1_out_channals)

        cv2_out_channals = 16
        self.conv2 = nn.Conv2d(in_channels=cv1_out_channals, out_channels=cv2_out_channals, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(cv2_out_channals)
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

if __name__ == '__main__':
    batch_size, learning_rate, epochs, tn, path, telmin, epochmin, lastbatch = 64, 1e-3, 2000, 44033, 'C:/holo/4/', 500, 0, 0
    st = binary_string_to_bit_list('C:/holo/dataset/st_36.txt',tn)
    opr1 = read_txt_to_2d_list('C:/holo/dataset/opr_530.txt',' ',tn)
    opr3 = read_txt_to_2d_list('C:/holo/dataset/opr_800.txt', ' ', tn)
    opr2 = read_txt_to_2d_list('C:/holo/dataset/opr_670.txt', ' ', tn)
    opr = [item1 + item2+item3 for item1, item2,item3 in zip(opr1, opr2,opr3)]
    trn = int(tn * 0.9)
    total_id, tdi, batchst, batchopr = [j for j in range(tn)], [j for j in range(trn)], [], []
    np.random.shuffle(total_id)
    # 清空记录loss的文档
    with open(path+"forward loss.txt", "a") as f:
        f.truncate(0)
    f.close()
    device = torch.device('cuda:0')
    net2 = StO().to(device)
    optimizer = optim.Adam(net2.parameters(), lr=learning_rate)
    criteon = nn.MSELoss().to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net2.parameters())))
    # 获取训练集和测试集
    trainst = copy.deepcopy([st[i] for i in total_id[:trn]])
    trainopr = copy.deepcopy([opr[i] for i in total_id[:trn]])
    testst = copy.deepcopy([st[i] for i in total_id[trn:]])
    testopr = copy.deepcopy([opr[i] for i in total_id[trn:]])
    if trn % batch_size != 0:
        lastbatch = 1
    k = trn // batch_size + lastbatch
    for epoch in range(epochs):
        losstem = 0
        np.random.shuffle(tdi)
        net2.train()
        for i in range(k):
            if lastbatch == 1 and i == k - 1:
                bs = trn % batch_size
            else:
                bs = batch_size
            for j in range(bs):
                batchst.append(trainst[tdi[i * bs + j]])
                batchopr.append(trainopr[tdi[i * bs + j]])
            batchopr, batchst = torch.tensor(batchopr).cuda(), torch.tensor(batchst).cuda()
            loss = criteon(net2(batchst), batchopr)
            losstem += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchst, batchopr = [], []
        net2.eval()

        if epoch % 10 == 0:
            trl = int((losstem / k) ** 0.5 * 1e3)
            plt_st, plt_opr = torch.tensor(testst).cuda(), torch.tensor(testopr).cuda()
            tel = int(criteon(net2(plt_st), plt_opr) ** 0.5 * 1e3)
            with open(path + "forward loss.txt", "a") as f:
                f.write(str(trl) + "," + str(tel) + "\n")
            f.close()
            print("epoch ", epoch, " averange training loss is ", trl, ", averange testing loss is ", tel, " *10^(-3)")
            if epoch == 0:
                torch.save(net2, path + str(tn) + 'fe' + str(epochmin) + 'l' + str(telmin) + '.mdl')
            if telmin > tel:
                os.remove(path + str(tn) + 'fe' + str(epochmin) + 'l' + str(telmin) + '.mdl')
                epochmin, telmin = epoch, tel
                torch.save(net2, path + str(tn) + 'fe' + str(epochmin) + 'l' + str(telmin) + '.mdl')