import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time

start_time = time.time()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# UNet Model Definition
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoding path
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)

        # Decoding path with upsampling
        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)  # Match channels after concatenation
        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)  # Match channels after concatenation
        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)  # Match channels after concatenation
        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)  # Match channels after concatenation

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.functional.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(nn.functional.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(nn.functional.max_pool2d(enc3, kernel_size=2))
        bottleneck = self.bottleneck(nn.functional.max_pool2d(enc4, kernel_size=2))

        # Decoding path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


# Load model and weights from initial training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=2).to(device)
model.load_state_dict(torch.load('../Model weights/MUIA_model_weights_unet.pth', map_location=torch.device('cpu')))
model.eval()


# Define pseudolabelling dataset
class UnlabeledUltrasoundDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, img_name


# Pseudolabelling settings
unlabeled_image_dir = '../Datasets/TM_Split/test/cropped_all'
pseudolabel_dir = '../Datasets/TM_Split/test/croppedAll_unet'
os.makedirs(pseudolabel_dir, exist_ok=True)

# Pseudolabelling process
unlabeled_dataset = UnlabeledUltrasoundDataset(unlabeled_image_dir, transform)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=False)

with torch.no_grad():
    for images, image_names in unlabeled_dataloader:
        images = images.to(device)
        outputs = model(images)
        pseudolabels = torch.argmax(outputs, dim=1)

        for pseudolabel, image_name in zip(pseudolabels, image_names):
            pseudolabel_np = (pseudolabel.cpu().numpy() * 255).astype(np.uint8)
            pseudolabel_path = os.path.join(pseudolabel_dir, image_name.replace('.bmp', '_pseudolabel.png'))
            cv2.imwrite(pseudolabel_path, pseudolabel_np)


print("--- Pseudolabeling complete: %s seconds ---" % (time.time() - start_time))