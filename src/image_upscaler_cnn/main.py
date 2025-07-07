import argparse
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import importlib.resources
from image_upscaler_cnn.model import ImageUpscalerCNN

BUNDLED_MODEL_FILENAME = 'upscaler_model_2x.pth'
BUNDLED_MODEL_PATH = importlib.resources.files('image_upscaler_cnn') / 'model_file' / BUNDLED_MODEL_FILENAME

def main():
    parser = argparse.ArgumentParser(description='Image upscaler using PyTorch')
    parser.add_argument('input_file', help='Input image file')
    parser.add_argument('output_file', help='Output image file')
    args = parser.parse_args()

    input_image = Image.open(args.input_file).convert('RGB')

    preprocess = T.Compose([
        T.ToTensor(),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)
    upscaled_image = upscale_image(input_tensor)
    pil_image = T.ToPILImage()(upscaled_image.squeeze(0).cpu().clip(0, 1))
    pil_image.save(args.output_file)


def upscale_image(input_tensor):
    model = ImageUpscalerCNN()
    model.load_state_dict(torch.load(BUNDLED_MODEL_PATH))
    upscaled_image = model(input_tensor)
    return upscaled_image

if __name__ == '__main__':
    main()