import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
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
class OtS(nn.Module):
    def __init__(self, num_input=60):
        super().__init__()
        self.fc1 = nn.Linear(num_input, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 36*16)
        self.sig = nn.Sigmoid()

        cv1_out_channals = 8
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=cv1_out_channals, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cv1_out_channals)

        cv2_out_channals = 1
        self.conv2 = nn.Conv2d(in_channels=cv1_out_channals, out_channels=cv2_out_channals, kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(cv2_out_channals)
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
    path= 'C:\\holo\\4\\'
    ch, batch_size, learning_rate, epochs, tn, telmin, epochmin, lastbatch = 30, 64, 1e-4, 2000, int(1e6), 500, 0, 0
    trainnum = int(0.9 * tn)
    with open(path + str(tn) + " i loss.txt", "a") as f:
        f.truncate(0)
    f.close()
    model = torch.load(path + '44033fe50l201.mdl')
    device = torch.device('cuda:0')
    net = OtS().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criteon = nn.MSELoss().to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
    tdi = [j for j in range(trainnum)]
    if trainnum % batch_size != 0:
        lastbatch = 1
    k = trainnum // batch_size + lastbatch
    for epoch in range(epochs):
        losstem = 0
        np.random.shuffle(tdi)
        net.train()
        for i in range(k):
            if lastbatch == 1 and i == k - 1:
                bs = trainnum % batch_size
            else:
                bs = batch_size
            a = np.random.rand(bs, ch) * 2 * np.pi
            batchopr = np.zeros((bs, ch * 2))
            for k in range(bs):
                for j in range(ch):
                    batchopr[k, 2 * j], batchopr[k, 2 * j + 1] = np.sin(a[k, j]) / 2 + 0.5, np.cos(a[k, j]) / 2 + 0.5
            batchopr = torch.tensor(batchopr, dtype=torch.float32).cuda()

            loss = criteon(model(net(batchopr)), batchopr)
            
            # print("MSE",loss)
            optimizer.zero_grad()
            loss.backward()
            losstem += float(loss)
            optimizer.step()
        net.eval()
        if epoch % 10 == 0:
            trl = int((losstem / k) ** 0.5 * 1e3)
            a = np.random.rand(100, ch) * 2 * np.pi
            testopr = np.zeros((100, ch * 2))
            for i in range(100):
                for j in range(ch):
                    testopr[i, 2 * j] = np.sin(a[i, j]) / 2 + 0.5
                    testopr[i, 2 * j + 1] = np.cos(a[i, j]) / 2 + 0.5
            plt_opr = torch.tensor(testopr, dtype=torch.float32).cuda()
            tel = criteon(model(net(plt_opr)), plt_opr)
            tel = int(tel ** 0.5 * 1e3)
            with open(path + str(tn) + " i loss.txt", "a") as f:
                f.write(str(trl) + "," + str(tel) + "\n")
            f.close()
            print("epoch ", epoch, " averange training loss is ", trl, ", averange testing loss is ", tel, " *10^(-3)")
            if epoch == 0:
                torch.save(net, path + str(tn) + 'ie' + str(epochmin) + 'l' + str(telmin) + '.mdl')
            if telmin > tel:
                os.remove(path + str(tn) + 'ie' + str(epochmin) + 'l' + str(telmin) + '.mdl')
                telmin, epochmin = tel, epoch
                torch.save(net, path + str(tn) + 'ie' + str(epochmin) + 'l' + str(telmin) + '.mdl')
