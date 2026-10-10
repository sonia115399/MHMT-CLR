import torch
import torchvision.transforms as transforms
from PIL import Image

class LeafDiseaseDataAugmentation:
    def __init__(self):
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Lambda(lambda img: self.resize_image(img)),
                transforms.Resize(256),
                transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Lambda(lambda img: self.resize_image(img)),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

    def resize_image(self, img):

        width, height = img.size


        if width < height:
            scale = 256 / width
        else:
            scale = 256 / height


        new_width = int(width * scale)
        new_height = int(height * scale)


        return img.resize((new_width, new_height), Image.LANCZOS)

    def get_transforms(self, phase):
        return self.data_transforms[phase]
