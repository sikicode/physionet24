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

import pandas as pd
from sklearn.model_selection import train_test_split

from helper import get_file_names, DoubleConv, UNet, DiceLoss, FocalLoss, CombinedLoss
from torch_loader import DenoiseImageDataset, ECGNoisedImageDataset


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# CV_num = 4

label_df = pd.read_csv('/data/padmalab/ecg/data/external/PhysioNet2024_Data/physionet.org/files/ptb-xl/1.0.3/ptbxl_label_df.csv')
label_df['input_filename_lr'] = label_df['filename_lr'].str.replace('records100/', 'records100_generate_output/records100_generate_output_')
label_df['output_filename_lr'] = label_df['filename_lr'].str.replace('records100/', 'records100_ground_truth/records100_ground_truth_')
label_df['input_filename_hr'] = label_df['filename_hr'].str.replace('records500/', 'records500_generate_output/records500_generate_output_')
label_df['output_filename_hr'] = label_df['filename_hr'].str.replace('records500/', 'records500_ground_truth/records500_ground_truth_')

batch_size = 1
num_epochs = 50
patience = 10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization values
])

train_label_df = label_df[(label_df['ecg_id'] <= 5000)]

train_label_df, val_label_df = train_test_split(train_label_df, test_size=0.2, random_state=42)  # 80% training, 20% validation
train_label_df.reset_index(inplace=True)
val_label_df.reset_index(inplace=True)

train_dataset = DenoiseImageDataset(label_df=train_label_df, transform=transform, noise_transform=noise_transform, ecg_frequency='500hz')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = DenoiseImageDataset(label_df=val_label_df, transform=transform, noise_transform=noise_transform, ecg_frequency='500hz')
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = UNet(n_channels=3, n_classes=1).to(device) # Adjust as necessary
criterion = CombinedLoss(alpha=1, gamma=2, logits=True, reduce=True).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.000001, patience=7)

best_loss = 10
record = {
    'epoch': [],
    'train_loss': [],
    'val_loss': []
}
for epoch in range(num_epochs):
    model.train()
    record['epoch'].append(epoch)
    train_loss = 0    
    with tqdm(total=len(train_dataloader)) as pbar:
        for inputs, targets in train_dataloader:    
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # Optionally log training loss here        
            pbar.set_description(f"[epoch {epoch+1}/{num_epochs} ]")
            pbar.set_postfix_str(f"loss = {loss.item():.4f}")
            pbar.update(1)
    record['train_loss'].append(train_loss/len(val_dataloader))

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_dataloader)
        # Optionally log validation loss here
        print(f'Validation - Epoch {epoch+1}, Loss: {val_loss}')
        record['val_loss'].append(val_loss)
        scheduler.step(val_loss) 
        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, './model/UNet_torch_all.model')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Stopping early due to no improvement in validation loss.")
            break

pd.DataFrame.from_dict(record, orient='index').to_pickle('./model/UNet_torch_all.pickle')