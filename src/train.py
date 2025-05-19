import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from ImageUpscaler import ImageUpscalerCNN
from Dataset import ImageDataset, transform_hr, transform_lr


dataset = ImageDataset("data/DIV2K_train_HR", transform_lr, transform_hr)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, loss, and optimizer
model = ImageUpscalerCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    epoch_loss = 0
    for lr_imgs, hr_imgs in dataloader:
        optimizer.zero_grad()
        outputs = model(lr_imgs)
        loss = criterion(outputs, hr_imgs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), "upscaler_model.pth")
