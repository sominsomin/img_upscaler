import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from image_upscaler_cnn import ImageUpscalerCNN
from Dataset import ImageDataset, transform_hr, transform_lr

test_data_path = "data/DIV2K_valid_HR"
model_path = "models/upscaler_model_2x_w_500_h_500_epoch_2_loss_0.0014.pth"

test_dataset = ImageDataset(test_data_path, transform_lr, transform_hr)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = ImageUpscalerCNN()
criterion = nn.MSELoss()
model.load_state_dict(torch.load(model_path))

test_loss = 0
n_batch = 0
n_batches = len(test_dataloader)

with torch.no_grad():
    for lr_imgs, hr_imgs in test_dataloader:
        n_batch += 1
        print(f"current batch: {n_batch}/{n_batches}", end='\r')
        outputs = model(lr_imgs)
        loss = criterion(outputs, hr_imgs)
        test_loss += loss.item()

test_loss /= len(test_dataloader)
print(f"Test Loss: {test_loss:.4f}")
