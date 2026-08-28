import torch, cv2, os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    lossm, loss_n, epmin, epstart = 10000, 10000, 0, 0
    # lossn记录，lossm画图
    epend = 1000000
    ch, d = 30, 2
    img_size = 200
    save_index = 12
    learning_rate = 1e-3
    path= 'C:\\holo\\4\\'
    tensor_name = f'VDNN_phase_array/tensor_output_{save_index}.txt'
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
    
    # 定义 AdamW 优化器
    # optimizer = optim.AdamW([D], lr=1e-6, weight_decay=1e-4)
    optimizer = optim.Adam([D], lr=learning_rate)
    # 余弦退火调度器
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=40,               # 半周期长度（epoch数）
    #     eta_min=1e-6,            # 最小学习率下限
    #     last_epoch=-1
    # )
    
    for epoch in range(epstart, epend):
        J = torch.exp(1j * D)
        L = torch.abs(torch.fft.fftshift(torch.fft.fft2(J)))

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
            N = np.array(L0.cpu().detach().numpy())
            plt.subplots(figsize=(140, 42))
            for i in range(ch):
                plt.subplot(3, 10, i + 1)
                plt.imshow(N[i], cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(path + 'VDNN_phase_array/' + str(epoch) + '.png')
            plt.close()
            epmin, lossm = epoch, ii
