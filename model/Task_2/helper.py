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
# import wandb

# get_file_names
def get_file_names(input_files: list) -> list:
    input_file_names = []
    for input_file in input_files:
        temp_file_name = input_file.split('/')[-1].strip('.png')
        input_file_names.append(temp_file_name)
    print ('Total', len(input_file_names), 'files')
    return input_file_names


# data loader
class ImageDenoisingDataset(Dataset):
    def __init__(self, input_file_names:list, transform=None, 
                 input_path= '/data/padmalab/ecg/data/external/PhysioNet2024_Data/physionet.org/files/ptb-xl/1.0.3/records100_generate_output/records100_generate_output_03000/%s.png',
                 target_path = '/data/padmalab/ecg/data/external/PhysioNet2024_Data/physionet.org/files/ptb-xl/1.0.3/records100_ground_truth/records100_ground_truth_03000/%s.png'):
        self.input_dir = input_path
        self.target_dir = target_path
        self.transform = transform
        self.input_file_names = input_file_names

    def __len__(self):
        return len(self.input_file_names)

    def __getitem__(self, idx):
        base_name = self.input_file_names[idx]
        input_path = self.input_dir%base_name
        target_path = self.target_dir%base_name

        input_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        output_image = cv2.imread(target_path, cv2.IMREAD_COLOR)
        
        if self.transform is not None:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)        
            
        input_image = TF.resize(input_image, (896, 1152))
        output_image = TF.resize(output_image, (896, 1152))

        input_image = torch.sigmoid(input_image)
        output_image = torch.sigmoid(output_image)
        
        return input_image, output_image

# Conv backbone
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#UNet model
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)
        self.double_conv1 = DoubleConv(64, 128)
        self.double_conv2 = DoubleConv(128, 256)
        self.double_conv3 = DoubleConv(256, 512)
        self.double_conv4 = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_conv_up1 = DoubleConv(1024, 512)
        self.double_conv_up2 = DoubleConv(512, 256)
        self.double_conv_up3 = DoubleConv(256, 128)
        self.double_conv_up4 = DoubleConv(128, 64)
        self.sigmoid = nn.Sigmoid()
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.double_conv1(x2)
        x3 = self.down2(x2)
        x3 = self.double_conv2(x3)
        x4 = self.down3(x3)
        x4 = self.double_conv3(x4)
        x5 = self.down4(x4)
        x5 = self.double_conv4(x5)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.double_conv_up1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.double_conv_up2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.double_conv_up3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.double_conv_up4(x)
        logits = self.outc(x)
        logits = self.sigmoid(logits)
        return logits

#Dice loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

#Focal Loss https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Focal-Loss
class FocalLoss(nn.Module): 
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha, gamma)

    def forward(self, inputs, targets):
        return self.dice_loss(inputs, targets) + self.focal_loss(inputs, targets)


def calculate_dice(target, prediction):
    """
    Calculate Dice Coefficient for a single image.
    
    Args:
        target (np.array): The ground truth binary segmentation image.
        prediction (np.array): The predicted binary segmentation image.
    
    Returns:
        float: The Dice Coefficient.
    """
    intersection = np.logical_and(target, prediction)
    dice_score = 2 * np.sum(intersection) / (np.sum(target) + np.sum(prediction))
    return dice_score

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
        # output_image = torch.sigmoid(output_image)
        # print (output_image)
        return input_image, output_image.float()