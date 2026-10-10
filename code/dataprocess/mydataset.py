import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(name for name in os.listdir(root)
                              if not name.startswith(".") and os.path.isdir(os.path.join(root, name)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self._load_images()
        print(len(self.classes))

    def _load_images(self):
        images = []
        for i, cls in enumerate(self.classes):
            class_path = os.path.join(self.root, cls)
            if not os.path.isdir(class_path):
                continue

            if i >= len(self.classes):
                raise ValueError("More classes found than expected.")

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if (os.path.isfile(img_path) and not img_name.startswith(".")
                        and img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))):
                    images.append((img_path, i))

        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, target = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
