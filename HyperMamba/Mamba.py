
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat


import torch


def channel_shuffle(x, groups):
    """
    手动实现Channel Shuffle操作
    :param x: 输入张量，形状通常为 (batch_size, channels, height, width)
    :param groups: 分组数量
    :return: 经过Channel Shuffle后的输出张量
    """
    batch_size, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # 先将形状调整为 (batch_size, groups, channels_per_group, height, width)
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # 进行维度变换，实现通道打乱，这里是将维度1（groups）和维度2（channels_per_group）进行交换
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    # 再将形状恢复为 (batch_size, channels, height, width)
    x = x.view(batch_size, num_channels, height, width)
    return x

class MambaBlock(nn.Module):
    def __init__(self, in_dim, n, d_conv=4,expand=2):
        super(MambaBlock, self).__init__()
        self.n = n
        self.in_dim = in_dim
        self.d = int(in_dim * expand)
        self.in_proj = nn.Linear(in_dim, self.d * 2, bias=False)
        self.delta_rank = math.ceil(in_dim / 16)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d,
            out_channels=self.d,
            kernel_size=d_conv,
            groups=self.d,
            padding=d_conv-1,
        )
        self.out_norm = nn.LayerNorm(self.d)
        self.out_proj = nn.Linear(self.d, in_dim, bias=False)
        self.x_proj = nn.Linear(self.d, self.delta_rank + n * 2, bias=False)
        self.delta_proj = nn.Linear(self.delta_rank, self.d, bias=True)
        
        A = repeat(torch.arange(1, n+1), 'n -> d n', d=self.d) 
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d))
        

    def selective_scan(self, x, delta, A, B, C, D):
        x = rearrange(x, 'b l d -> b d l')
        delta = rearrange(delta, 'b l d -> b d l')
        B = rearrange(B, 'b l n -> b n l')
        C = rearrange(C, 'b l n -> b n l')
        res = selective_scan_fn(x, delta, A, B, C, D)
        res = rearrange(res, 'b d l -> b l d')
        return res
    
    def ssm(self, x):
        b,l,d = x.shape
        (d, n) = self.A_log.shape
        assert d == self.d
        assert n == self.n
        A = -torch.exp(self.A_log.float())  # (d, n)
        D = self.D.float()
        x_dbl = self.x_proj(x)  # (b, l, d) -> (b, l, delta_rank + 2 * n)
        delta, B, C = torch.split(x_dbl, [self.delta_rank, self.n, self.n], dim=-1)
        delta = F.softplus(self.delta_proj(delta))  # (b, l, d)
        x = self.selective_scan(x, delta, A, B, C, D)
        return x
    
    def forward(self, x):
        (b,l,in_dim) = x.shape
        assert in_dim == self.in_dim
        (x,res) = torch.split(self.in_proj(x), [self.d, self.d], dim=-1) # (b l in_dim) -> b l d, b l d
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :l] 
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)
        x = self.ssm(x)
        x =  self.out_norm(x)
        x = x * F.silu(res)
        x = self.out_proj(x) #  (b l d) -> (b l in_dim)
        return x
    

class VMamba(nn.Module):
    def __init__(self,in_channels,n=16,d_conv=3):
        super(VMamba, self).__init__()
        self.d = d = int(2 * in_channels)
        self.conv2d = nn.Conv2d(d, d, groups=d, kernel_size=d_conv,padding=(d_conv - 1) // 2)
        self.n = n
        self.in_proj = nn.Linear(in_channels, self.d * 2)
        self.delta_rank = math.ceil(in_channels / 16)
        self.delta_proj = nn.Linear(self.delta_rank, self.d * 4, bias=True)
        
        A = repeat(torch.arange(1, n+1), 'n -> d n', d=self.d * 4) 
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d * 4))
        self.x_proj = nn.Linear(self.d * 4, self.delta_rank + n * 2, bias=False)

        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, in_channels, bias=False)
        
        
        self.training = False
        
        
    def selective_scan(self, x, delta, A, B, C, D):
        x = rearrange(x, 'b l d -> b d l')
        delta = rearrange(delta, 'b l d -> b d l')
        B = rearrange(B, 'b l n -> b n l')
        C = rearrange(C, 'b l n -> b n l')
        res = selective_scan_fn(x, delta, A, B, C, D)
        res = rearrange(res, 'b d l -> b l d')
        return res
    
    def ssm(self, x):
        b,l,d = x.shape
        (d, n) = self.A_log.shape
        assert d == self.d * 4
        assert n == self.n
        A = -torch.exp(self.A_log.float())  # (d, n)
        D = self.D.float()
        x_dbl = self.x_proj(x)  # (b, l, d) -> (b, l, delta_rank + 2 * n)
        delta, B, C = torch.split(x_dbl, [self.delta_rank, self.n, self.n], dim=-1)
        delta = F.softplus(self.delta_proj(delta))  # (b, l, d)
        x = self.selective_scan(x, delta, A, B, C, D)
        return x
    
    def forward(self, x):
        b,c,h,w = x.shape
        x,z = self.in_proj(x.permute(0,2,3,1)).chunk(2, dim=-1)
        x = x.permute(0,3,1,2).contiguous()
        x = F.silu(self.conv2d(x)) # (b, d, h, w)
        # x = channel_shuffle(x,4)
        x1 = rearrange(x, 'b d h w -> b d (h w)')
        x2 = rearrange(x, 'b d h w -> b d (w h)')
        x3 = rearrange(x.flip(-1).flip(-2), 'b d h w -> b d (h w)')
        x4 = rearrange(x.flip(-1).flip(-2), 'b d h w -> b d (w h)')

        x = torch.cat([x1,x2,x3,x4],dim=1)
        x = rearrange(x, 'b d l -> b l d') # d -> 4d, l == (h w)
        x = self.ssm(x)
        x = rearrange(x, 'b (h w) (k d) -> b k d (h w)',k=4, h=h,w=w)
        assert x.dtype == torch.float
        x1,x2,x3,x4 = x[:,0],x[:,1],x[:,2],x[:,3]
        x1 = rearrange(x1, 'b d (h w)-> b d h w', h=h,w=w)
        x2 = rearrange(x2, 'b d (w h)-> b d h w', h=h,w=w)
        x3 = rearrange(x3, 'b d (h w)-> b d h w', h=h,w=w).flip(-2).flip(-1)
        x4 = rearrange(x4, 'b d (w h)-> b d h w', h=h,w=w).flip(-2).flip(-1)
        x = x1 + x2 + x3 + x4
        x = rearrange(x, 'b d h w -> b h w d')
        x = self.out_norm(x)
        x = x * F.silu(z)
        x = self.out_proj(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x
    
    
    
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
class SpectralMamba(nn.Module):
    def __init__(self,in_channels,n=16,d_conv=3):
        super(SpectralMamba, self).__init__()
        self.d = d = int(2 * in_channels)
        self.conv2d = nn.Conv2d(d, d, groups=d, kernel_size=d_conv,padding=(d_conv - 1) // 2)
        self.n = n
        self.in_proj = nn.Linear(in_channels, self.d * 2)
        self.delta_rank = math.ceil(in_channels / 16)
        self.delta_proj = nn.Linear(self.delta_rank, self.d * 4, bias=True)
        
        A = repeat(torch.arange(1, n+1), 'n -> d n', d=self.d * 4) 
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d * 4))
        self.x_proj = nn.Linear(self.d * 4, self.delta_rank + n * 2, bias=False)

        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, in_channels, bias=False)
        
        
        self.conv1 = nn.Conv2d(d,d,3,1,1)
        self.conv2 = nn.Conv2d(d,d,1,1,0)
        self.conv3 = nn.Conv2d(d,d,5,1,2)
        self.conv4 = nn.Conv2d(d,d,3,1,1)
        
        
    def selective_scan(self, x, delta, A, B, C, D):
        x = rearrange(x, 'b l d -> b d l')
        delta = rearrange(delta, 'b l d -> b d l')
        B = rearrange(B, 'b l n -> b n l')
        C = rearrange(C, 'b l n -> b n l')
        res = selective_scan_fn(x, delta, A, B, C, D)
        res = rearrange(res, 'b d l -> b l d')
        return res
    
    def ssm(self, x):
        b,l,d = x.shape
        (d, n) = self.A_log.shape
        assert d == self.d * 4
        assert n == self.n
        A = -torch.exp(self.A_log.float())  # (d, n)
        D = self.D.float()
        x_dbl = self.x_proj(x)  # (b, l, d) -> (b, l, delta_rank + 2 * n)
        delta, B, C = torch.split(x_dbl, [self.delta_rank, self.n, self.n], dim=-1)
        delta = F.softplus(self.delta_proj(delta))  # (b, l, d)
        x = self.selective_scan(x, delta, A, B, C, D)
        return x
    
    def forward(self, x):
        b,c,h,w = x.shape
        s = 1
        # print(s)
        if h == 64:s = 8
        if h == 32:s = 8
        if h == 16:s = 8
        if h == 8:s = 8
        x,z = self.in_proj(x.permute(0,2,3,1)).chunk(2, dim=-1)
        x = x.permute(0,3,1,2).contiguous()
        x = F.silu(self.conv2d(x)) # (b, d, h, w)
    
        x1 = rearrange(x, 'b d h (s w) -> b d (s h w)',s=s)
        x2 = rearrange(x, 'b d (s h) w -> b d (s h w)',s=s)
        x3 = rearrange(x.flip(-1).flip(-2), 'b d h (s w) -> b d (s h w)',s=s)
        x4 = rearrange(x.flip(-1).flip(-2), 'b d (s h) w  -> b d (s h w)',s=s)
        
        x = torch.cat([x1,x2,x3,x4],dim=1)
        x = rearrange(x, 'b d l -> b l d') # d -> 4d, l == (h w)
        x = self.ssm(x)
        x = rearrange(x, 'b (h w) (k d) -> b k d (h w)',k=4, h=h,w=w)
        assert x.dtype == torch.float
        x1,x2,x3,x4 = x[:,0],x[:,1],x[:,2],x[:,3]

        
        x1 = rearrange(x1, 'b d (s h w)-> b d h (s w)', h=h,w=w)
        x2 = rearrange(x2, 'b d (s h w)-> b d (s h) w', h=h,w=w)
        x3 = rearrange(x3, 'b d (s h w)-> b d h (s w)', h=h,w=w).flip(-2).flip(-1)
        x4 = rearrange(x4, 'b d (s h w)-> b d (s h) w', h=h,w=w).flip(-2).flip(-1)

        x = x1 + x2 + x3 + x4
        x = rearrange(x, 'b d h w -> b h w d')
        x = self.out_norm(x)
        x = x * F.silu(z)
        x = self.out_proj(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x
    
    

class SS2DBlock(nn.Module):
    
    def __init__(self,dim):
        super(SS2DBlock, self).__init__()
        self.ss2d = SpectralMamba(dim)

    def forward(self, x):
        x = self.ss2d(x)
        return x
