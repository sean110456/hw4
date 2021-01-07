import torch
from PIL import Image
import os
import numpy as np
from torch.autograd import Variable
import time
import math
import matplotlib.pyplot as plt
from torchvision import transforms

# functions


def colorize(y, ycbcr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:, :, 0] = y
    img[:, :, 1] = ycbcr[:, :, 1]
    img[:, :, 2] = ycbcr[:, :, 2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img


# For GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pretrained model
model = torch.load('trained_model/model_epoch_50.pth')
model = model.to(device)
model.eval()

TEST_PATH = 'testing_lr_images'
OUTPUT_PATH = 'output/'
if not os.path.exists('output/'):
    os.makedirs('output')

for filename in os.listdir(TEST_PATH):
    # Convert the images into YCbCr mode and extraction the Y channel
    img = Image.open(os.path.join(TEST_PATH, filename)).convert('YCbCr')
    hr_length = img.size[0] * 3
    hr_width = img.size[1] * 3
    resize_img = transforms.Resize((hr_width, hr_length))(img)
    resize_img_arr = np.array(resize_img)
    resize_img_Y = resize_img_arr[:, :, 0].astype(float)/255

    # Prepare for the input, a pytorch tensor
    input_img = Variable(torch.from_numpy(resize_img_Y).float()).view(
        1, -1, resize_img_Y.shape[0], resize_img_Y.shape[1])
    input_img = input_img.to(device)

    # Get output
    out = model(input_img).cpu()
    hr_img_Y = out.data[0].numpy().astype(np.float32) * 255
    hr_img_Y[hr_img_Y < 0] = 0
    hr_img_Y[hr_img_Y > 255] = 255
    hr_img_Y = hr_img_Y[0, :, :]

    # Generate the hr image with the new Y channel and CbCr from the resized YCbCr image
    hr_img = colorize(hr_img_Y, resize_img_arr)
    hr_img.save(OUTPUT_PATH + filename)
