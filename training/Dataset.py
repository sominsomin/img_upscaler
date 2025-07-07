from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os

from settings import INPUT_WIDTH, INPUT_HEIGHT, SCALING_FACTOR


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform_lr, transform_hr):
        self.data_dir = data_dir
        self.images = os.listdir(data_dir)
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        hr_img = self.transform_hr(img)
        lr_img = self.transform_lr(img)
        return lr_img, hr_img

transform_lr = T.Compose([
    T.Resize((INPUT_WIDTH, INPUT_HEIGHT)),
    T.ToTensor()
])

transform_hr = T.Compose([
    T.Resize((INPUT_WIDTH * SCALING_FACTOR, INPUT_HEIGHT * SCALING_FACTOR)),
    T.ToTensor()
])