import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import NPZDataset
from HyerMamba import HyperMamba
from utils import *

test_dataset_path = './dataset/PaviaU_test.npz'
train_dataset_path = './dataset/PaviaU_train.npz'
hsi_bands = 103
msi_bands = 4
lr = 4e-4


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = HyperMamba(hsi_bands).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,step=100,decay_rate=0.1)
max_epochs = 500
test_dataset = NPZDataset(test_dataset_path)
train_dataset = NPZDataset(train_dataset_path)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

for e in range(max_epochs):
    for idx,batch_data in enumerate(train_dataloader):
        gt,lrhsi,msi = batch_data
        gt,lrhsi,msi = gt.to(device),lrhsi.to(device),msi.to(device)
        optimizer.zero_grad()
        pre_= model(lrhsi)
        loss_val = F.l1_loss(pre_,gt)
        loss_val.backward()
        optimizer.step()
    scheduler.step()
    mean_loss = []
    mean_psnr = []
    with torch.no_grad():
        model.eval()
        for idx,batch_data in enumerate(test_dataloader):
            gt,lrhsi,msi = batch_data
            gt,lrhsi,msi = gt.to(device),lrhsi.to(device),msi.to(device)
            pre_= model(lrhsi,msi)
            mean_psnr.append(Metrics.cal_psnr(pre_,gt).item())
            mean_loss.append(F.l1_loss(pre_,gt).item()) 
            model.train()

    print(f"Epoch {e+1}/{max_epochs}, Loss: {sum(mean_loss)/len(mean_loss):.4f}, PSNR:{sum(mean_psnr)/len(mean_psnr):.4f}")
    
