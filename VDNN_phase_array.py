import torch, cv2, os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure  # 需要安装 torchmetrics
from torch.cuda.amp import autocast, GradScaler
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingLR

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
from train_back import OtS
from back_new import HybridDecoder,CBAM,ChannelAttention,SpatialAttention,ResidualBlock

def loss_ensemble(pred, target, epoch, num_epochs):
    ch = pred.shape[0]  # pred shape: [ch, H, W]

    # 1. 加权MSE损失（逐通道加权）
    mse_per_channel = torch.mean((pred - target) ** 2, dim=(1, 2))  # [ch]
    weights_per_channel = mse_per_channel / (mse_per_channel.sum() + 1e-8)
    loss_mse_weighted = torch.sum(mse_per_channel * weights_per_channel) * 10
    # if epoch % 100 == 0:
    #     print(f"Epoch {epoch} "
    #         f"MSE: {loss_mse_weighted.item():.6f} | ")
    return loss_mse_weighted
    # 2. SSIM损失
    # 初始化SSIM损失
    # ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    # ssim_val = ssim_loss(pred.unsqueeze(1), target.unsqueeze(1))
    # loss_ssim = 1 - ssim_val

    # 3. 正交损失
    channel_vectors = pred.view(ch, -1)
    norm = torch.norm(channel_vectors, dim=1, keepdim=True)
    normalized = channel_vectors / torch.clamp(norm, min=1e-8)
    gram_matrix = normalized @ normalized.T
    identity_mask = ~torch.eye(ch, dtype=torch.bool, device='cuda')
    loss_orth = torch.mean(gram_matrix[identity_mask] ** 2) 

    # 前期：强调MSE
    w_mse = 1.0
    # w_ssim = 0.0
    w_orth = 0.2 + 0.1 *  epoch / 2000
    if w_orth > 0.6:
        w_orth = 0.6

    if epoch % 10 == 0:
        print(f"Epoch {epoch} "
            f"MSE: {loss_mse_weighted.item():.6f} | "
            # f"SSIM Loss: {loss_ssim.item():.6f} | "
            f"Orth Loss: {loss_orth.item():.6f} | ")
    # 5. 总损失
    total_loss = w_mse * loss_mse_weighted +   w_orth * loss_orth

    return total_loss   

if __name__ == '__main__':
    lossm, loss_n, epmin, epstart = 55000, 56000, 0, 0
    # lossn记录，lossm画图
    epend = 1000000
    ch, d = 30, 2
    img_size = 200
    save_index = 10
    learning_rate = 5e-6
    path= 'C:\\holo\\4\\'
    back_model = torch.load(path + '1000000ie510l318.mdl')
    # back_model = torch.load(path + '44033ie1180l273.mdl')
    tensor_name = f'VDNN_phase_array/tensor_output_{save_index}.txt'
    forward_model = torch.load(path + '44033fe50l201.mdl')
    criteon = nn.MSELoss().to(torch.device('cuda:0'))
    use_scaler = False
    B, D0 = [], []
    for i in range(ch):
        if i<10:
            read_path='number/'
        elif i<36:
            read_path='Capital letters/'
            i-=10
        else:
            read_path = 'Lowercase letters/'
            i -= 36
        t = cv2.imread('C:/holo/pictures/' +read_path+ str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
        t = cv2.resize(t, (img_size, img_size))
        t = t.astype(np.float64) / 255.0
        B.append(t.tolist())
    B = torch.tensor(B)
    D_flag = 1
    if D_flag == 0:
        with open(path + 'VDNN_phase_array/tensor_output.txt', 'r') as file:
            for line in file:
                D0.append(list(map(float, line.strip().split())))
        D0 = torch.tensor(D0, dtype=torch.float32)
        D0 = D0.reshape(ch, img_size, img_size).cuda()
        D = D0
    if D_flag == 1:
        with open(path + f'VDNN_phase_array/tensor_output_{save_index-1}.txt', 'r') as file:
            for line in file:
                D0.append(list(map(float, line.strip().split())))
        D0 = torch.tensor(D0, dtype=torch.float32)
        D0 = D0.reshape(ch, img_size, img_size).cuda()
        D = D0
    if D_flag == 2: 
        # training from empty!
        D = torch.zeros(ch, img_size, img_size, device='cuda').uniform_(-1, 1)  # 均匀分布在 [-0.1, 0.1]
    D.requires_grad = True
    
    
    optimizer = optim.Adam([D], lr=learning_rate)
    # 余弦退火调度器
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=40,               # 半周期长度（epoch数）
    #     eta_min=1e-6,            # 最小学习率下限
    #     last_epoch=-1
    # )
    
    for epoch in range(epstart, epend):
        if use_scaler:
            with autocast():
                # D 就是结构的相位
                E = D.view(ch, -1).transpose(0, 1)  # shape: (40000, ch)
                G = torch.cat([
                    (torch.sin(E) / 2 + 0.5),
                    (torch.cos(E) / 2 + 0.5)
                ], dim=1)   
                H = back_model(G.float())
                I = forward_model(H) - 0.5
                J = torch.exp(1j * torch.atan2(I[:, ::2], I[:, 1::2]))
                # 优化后的版本
                J_reshaped = J.view(img_size, img_size, ch)  # shape: (200, 200, ch)
                K = J_reshaped.permute(2, 0, 1)    # shape: (ch, 200, 200)

                L = torch.abs(torch.fft.fftshift(torch.fft.fft2(K)))
                L[:, img_size // 2 - d:img_size //2 + d, img_size //2 - d:img_size //2 + d] = 0
                L0 = torch.zeros((ch, img_size, img_size), dtype=torch.float32).cuda()
                for i in range(ch):
                    L0[i] = L[i] / L[i].max()

                mse_loss = criteon(L0, B.cuda())
                loss = loss_ensemble(L0, B.cuda(), epoch - epstart, epend - epstart)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([D], max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            # scheduler.step(loss)
            # scheduler.step() # 余弦退火
        
        else:
            E = D.view(ch, -1).transpose(0, 1)  # shape: (40000, ch)
            G = torch.cat([
                (torch.sin(E) / 2 + 0.5),
                (torch.cos(E) / 2 + 0.5)
            ], dim=1)  
            H = back_model(G.float())
        
            I = forward_model(H) - 0.5   # I 理应和 G - 0.5 相同
            
            J = torch.exp(1j * torch.atan2(I[:, ::2], I[:, 1::2]))
            # 优化后的版本
            J_reshaped = J.view(img_size, img_size, ch)  # shape: (200, 200, ch)
            K = J_reshaped.permute(2, 0, 1)    # shape: (ch, 200, 200)
            
            L = torch.abs(torch.fft.fftshift(torch.fft.fft2(K)))
            # L = torch.abs(torch.fft.fftshift(torch.fft.fft2(D)))

            L[:, img_size // 2 - d:img_size //2 + d, img_size //2 - d:img_size //2 + d] = 0
            L0 = torch.zeros((ch, img_size, img_size), dtype=torch.float32).cuda()
            for i in range(ch):
                L0[i] = L[i] / L[i].max()

            mse_loss = criteon(L0, B.cuda())
            loss = loss_ensemble(L0, B.cuda(), epoch - epstart, epend - epstart)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([D], max_norm=1.0)
            optimizer.step()
            # scheduler.step(loss)
            # scheduler.step() # 余弦退火算法

        ii = int(mse_loss.cpu().detach().numpy() * ch * img_size * img_size)
        # with open(path + 'VDNN_phase_array/loss.txt', 'a') as f:
        #     f.write(str(epoch) + ' ' + str(ii) + '\n')
        if epoch % 40 == 0:
            with open(path + 'VDNN_phase_array/loss.txt', 'a') as f:
                f.write(str(epoch) + ' ' + str(ii) + '\n')
            print("epoch:",epoch)
            print("total loss:",ii)
        if ii < loss_n:
            loss_n = ii
            D0 = D.cpu().detach().numpy()
            with open(path + tensor_name, 'w') as f:
                for i in range(D0.shape[0]):
                    for j in range(D0.shape[1]):
                        for k in range(D0.shape[2]):
                            f.write(f"{D0[i, j, k]:.6f} ")
                        f.write('\n')
        if ii < lossm:
            # if epoch != 0:
                # os.remove(path + 'VDNN_phase_array/' + str(epmin) + '.png')
            N = np.array(L0.cpu().detach().numpy())
            plt.subplots(figsize=(140, 42))
            for i in range(ch):
                plt.subplot(3, 10, i + 1)
                plt.imshow(N[i], cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(path + 'VDNN_phase_array/' + str(epoch) + '.png')
            plt.close()
            # D0 = D.cpu().detach().numpy()
            # with open(path + tensor_name, 'w') as f:
            #     for i in range(D0.shape[0]):
            #         for j in range(D0.shape[1]):
            #             for k in range(D0.shape[2]):
            #                 f.write(f"{D0[i, j, k]:.6f} ")
            #             f.write('\n')
            epmin, lossm = epoch, ii
