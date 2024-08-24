import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filepaths = [os.path.join(root_dir, img_name) for img_name in os.listdir(root_dir)]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

from torch.utils.data import DataLoader

# Instantiate the dataset
dataset = CustomImageDataset(root_dir='D:\landscape_data',transform=transform)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# images = next(iter(data_loader))
# save_image(images,'images\h1.png')
    

