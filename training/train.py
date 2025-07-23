import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from image_upscaler_cnn import ImageUpscalerCNN
from Dataset import ImageDataset, transform_hr, transform_lr
from settings import SCALING_FACTOR, INPUT_HEIGHT, INPUT_WIDTH

data_path = "data/DIV2K_train_HR"
dataset = ImageDataset(data_path, transform_lr, transform_hr)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = ImageUpscalerCNN()

# load previously trained model
file = "training/models/upscaler_model_2x_w_500_h_500_epoch_0_loss_0.0032.pth"
model.load_state_dict(torch.load(file))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
n_batch = 0
n_batches = len(dataloader)
best_loss = float('inf')

for epoch in range(epochs):
    epoch_loss = 0
    for lr_imgs, hr_imgs in dataloader:
        n_batch += 1
        print(f"current batch: {n_batch}/{n_batches}", end='\r')
        optimizer.zero_grad()
        outputs = model(lr_imgs)
        loss = criterion(outputs, hr_imgs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    n_batch = 0
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    val_loss = epoch_loss/len(dataloader)
    model_target_path = f"training/models/upscaler_model_{SCALING_FACTOR}x_w_{INPUT_WIDTH}_h_{INPUT_HEIGHT}_epoch_{epoch}_loss_{val_loss:.4f}.pth"
    torch.save(model.state_dict(), model_target_path)

    if val_loss > best_loss:
        best_loss = val_loss
        print("early stopping reached.")
        break
    else:
        best_loss = val_loss
