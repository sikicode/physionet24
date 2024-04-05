import pandas as pd
from tqdm import tqdm, trange
import json
import cv2        
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt 

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from helper import get_file_names, ImageDenoisingDataset, DoubleConv, UNet, DiceLoss, FocalLoss, CombinedLoss, DenoiseImageDataset
import numpy as np

PATH = './model/UNet_torch.model'
model = torch.load(PATH)
model.eval()

label_df = pd.read_csv('/data/padmalab/ecg/data/external/PhysioNet2024_Data/physionet.org/files/ptb-xl/1.0.3/ptbxl_label_df.csv')
label_df['input_filename_lr'] = label_df['filename_lr'].str.replace('records100/', 'records100_generate_output/records100_generate_output_')
label_df['output_filename_lr'] = label_df['filename_lr'].str.replace('records100/', 'records100_ground_truth/records100_ground_truth_')
label_df['input_filename_hr'] = label_df['filename_hr'].str.replace('records500/', 'records500_generate_output/records500_generate_output_')
label_df['output_filename_hr'] = label_df['filename_hr'].str.replace('records500/', 'records500_ground_truth/records500_ground_truth_')

batch_size = 1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization values
])

test_label_df.reset_index(inplace=True)

test_dataset = DenoiseImageDataset(label_df=test_label_df, transform=transform, ecg_frequency='500hz')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

dice_list = []
with tqdm(total=len(test_dataloader)) as pbar:
    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        targets_temp = targets.detach().cpu().squeeze().numpy()
        outputs_temp = outputs.detach().cpu().squeeze().numpy()
        pbar.update(1)

# outputs_temp is the dnoised ECG image