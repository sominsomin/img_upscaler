import torch
import torchvision.transforms as T



from ImageUpscaler import ImageUpscalerCNN
from Dataset import ImageDataset, transform_lr, transform_hr

model = ImageUpscalerCNN()

model.load_state_dict(torch.load("upscaler_model.pth"))
model.eval()  # Set to evaluation mode

dataset = ImageDataset("data/DIV2K_train_HR", transform_lr, transform_hr)

test_lr = dataset[2][0].unsqueeze(0)
test_hr = dataset[2][1].unsqueeze(0)

test_output = model(test_lr)

pil_image_0 = T.ToPILImage()(test_lr.squeeze(0).cpu().clip(0, 1))
pil_image_1 = T.ToPILImage()(test_hr.squeeze(0).cpu().clip(0, 1))
pil_image_2 = T.ToPILImage()(test_output.squeeze(0).cpu().clip(0, 1))

pil_image_0.save("output_image_0.png")
pil_image_1.save("output_image_1.png")
pil_image_2.save("output_image_2.png")
