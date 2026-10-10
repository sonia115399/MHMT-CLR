from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from PIL import Image

def resize_image(img):

    width, height = img.size


    if width < height:
        scale = 256 / width
    else:
        scale = 256 / height


    new_width = int(width * scale)
    new_height = int(height * scale)


    return img.resize((new_width, new_height), Image.LANCZOS)


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):

        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.Lambda(lambda img: resize_image(img)),
                                              transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'tomato': lambda: datasets.ImageFolder(self.root_folder,
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(256),
                                                              n_views)),
                          'corn': lambda: datasets.ImageFolder(self.root_folder,
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(256),
                                                              n_views)),
                          'datepalm': lambda: datasets.ImageFolder(self.root_folder,
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(256),
                                                              n_views)),
                          'tea': lambda: datasets.ImageFolder(self.root_folder,
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(256),
                                                              n_views))}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
