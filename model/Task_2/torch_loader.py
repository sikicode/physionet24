import torch
from torch import nn
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2        
import os

class DenoiseImageDataset(Dataset):
    def __init__(self, label_df, transform=None, ecg_frequency = "100hz", 
                 parent_path='/data/padmalab/ecg/data/external/PhysioNet2024_Data/physionet.org/files/ptb-xl/1.0.3/',
                 out_channel = 1):
        """
        Args:
            dataframe (pd.DataFrame): A pandas dataframe containing 'file_path' and 'label' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.label_df = label_df
        self.transform = transform
        self.ecg_frequency = ecg_frequency
        self.parent_path = parent_path
        self.out_channel = out_channel

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        # try:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # 100 hz
        if self.ecg_frequency == "100hz":
            in_img_path = self.label_df.loc[idx, 'input_filename_lr'] 
            out_img_path = self.label_df.loc[idx,'output_filename_lr']
        else:        # 500 hz
            in_img_path = self.label_df.loc[idx, 'input_filename_hr'] 
            out_img_path = self.label_df.loc[idx,'output_filename_hr']

        # print (img_path)
        # print (out_img_path)
        input_image = cv2.imread(self.parent_path + in_img_path+'-0.png', cv2.IMREAD_COLOR)

        if self.out_channel == 1:
            output_image = cv2.imread(self.parent_path + out_img_path+'-0.png', cv2.IMREAD_GRAYSCALE)
            # print (
            # output_image[output_image < 255] = 0
            # print (output_image)
            output_image = output_image.reshape(1, output_image.shape[0], output_image.shape[1])
        else:
            output_image = cv2.imread(self.parent_path + out_img_path+'-0.png', cv2.IMREAD_COLOR)

        # if input_image == None:
        #     print (idx)
        if self.transform is not None:
            input_image = self.transform(input_image)
            if self.out_channel != 1:
                output_image = self.transform(output_image)        
            else:
                output_image = torch.from_numpy(output_image)

        input_image = TF.resize(input_image, (896, 1152))
        # print (output_image.shape)
        output_image = TF.resize(output_image, (896, 1152))

        input_image = torch.sigmoid(input_image)
        if self.out_channel == 1:
            output_image[output_image < 255] = 0
            output_image[output_image == 255] = 1            
        return input_image, output_image.float()

class ECGNoisedImageDataset(Dataset):
    def __init__(self, label_df, transform=None, ecg_frequency = "100hz", 
                 parent_path='/data/padmalab/ecg/data/external/PhysioNet2024_Data/physionet.org/files/ptb-xl/1.0.3/',
                 denoise_model = None,
                 out_channel = 1,
                 noise_transform = None,
                 device = None):
        """
        Args:
            dataframe (pd.DataFrame): A pandas dataframe containing 'file_path' and 'label' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.label_df = label_df
        self.noise_transform = noise_transform
        self.transform = transform
        self.ecg_frequency = ecg_frequency
        self.parent_path = parent_path
        self.out_channel = out_channel
        self.denoise_model = denoise_model
        self.device = device

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        # try:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # 100 hz
        if self.ecg_frequency == "100hz":
            in_img_path = self.label_df.loc[idx,'input_filename_lr']
        else:        # 500 hz
            in_img_path = self.label_df.loc[idx,'input_filename_hr']

        input_image = cv2.imread(self.parent_path + in_img_path+'-0.png', cv2.IMREAD_COLOR)
        input_image = self.noise_transform(input_image)        
        input_image = TF.resize(input_image, (896, 1152))
        input_image = torch.sigmoid(input_image).to(self.device)
        # print (input_image.shape)
        input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
        output_image = self.denoise_model(input_image)

        if self.transform:
            output_image = self.transform(output_image)

        label = self.label_df.loc[idx, 'Normal_ECG']
        return output_image, torch.tensor([label, ~label]).float()