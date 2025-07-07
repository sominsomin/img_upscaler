import torch
import torchvision.transforms as T
import torch.nn as nn
from torchvision import transforms
from image_upscaler_cnn import ImageUpscalerCNN
from Dataset import ImageDataset, transform_lr, transform_hr
from settings import INPUT_WIDTH, INPUT_HEIGHT, SCALING_FACTOR

model = ImageUpscalerCNN()

model.load_state_dict(torch.load("training/models/upscaler_model_2x_w_500_h_500_epoch_2_loss_0.0014.pth"))
model.eval() 
dataset = ImageDataset("data/DIV2K_valid_HR", transform_lr, transform_hr)
criterion = nn.MSELoss()

i = 20

resize_transform = transforms.Resize((INPUT_WIDTH*SCALING_FACTOR, INPUT_HEIGHT*SCALING_FACTOR))

test_lr = dataset[i][0].unsqueeze(0)
test_hr = dataset[i][1].unsqueeze(0)

test_output = model(test_lr)
test_lr_resized_bicubic_interpolation = resize_transform(test_lr)

print(f"loss ml model: {criterion(test_output, test_hr)}")
print(f"loss bicubic interpolation: {criterion(test_lr_resized_bicubic_interpolation, test_hr)}")

pil_image_0 = T.ToPILImage()(test_lr.squeeze(0).cpu().clip(0, 1))
pil_image_1 = T.ToPILImage()(test_hr.squeeze(0).cpu().clip(0, 1))
pil_image_2 = T.ToPILImage()(test_output.squeeze(0).cpu().clip(0, 1))
pil_image_3 = T.ToPILImage()(test_lr_resized_bicubic_interpolation.squeeze(0).cpu().clip(0, 1))

pil_image_0.save("input_image.png")
pil_image_1.save("target_image.png")
pil_image_2.save("upscaled_image.png")
pil_image_3.save("test_lr_resized.png")
