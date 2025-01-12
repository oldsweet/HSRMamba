import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mmcv
from utils import img_resize, crop_patchs



class NPZDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,get_graph=False):
        super(NPZDataset, self).__init__()
        loaded_data = np.load(data_path)
        self.LRHSI_list = loaded_data['LRHSI']
        self.RGB_list = loaded_data['RGB']
        self.GT_list = loaded_data['GT']

    def __getitem__(self, index):
        return torch.from_numpy(self.GT_list[index]).float(), \
                torch.from_numpy(self.LRHSI_list[index]).float(),\
                torch.from_numpy(self.RGB_list[index]).float()
    
    def __len__(self):
        return len(self.GT_list)


