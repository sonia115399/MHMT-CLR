from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def downstream_transform(training=False):
    steps = [T.Resize(256, interpolation=InterpolationMode.LANCZOS)]
    if training:
        steps += [
            T.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 4 / 3)),
            T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5),
            T.RandomRotation(15), T.RandomAffine(15, translate=(0.1, 0.1)),
            T.GaussianBlur(3, sigma=(0.1, 2.0)), T.ColorJitter(0.1, 0.1, 0.1, 0.1),
        ]
    else:
        steps += [T.CenterCrop(256)]
    return T.Compose(steps + [
        T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


class ContrastiveViews:


    def __init__(self):
        self.transform = T.Compose([
            T.Resize(256, interpolation=InterpolationMode.LANCZOS),
            T.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 4 / 3)),
            T.RandomHorizontalFlip(0.5),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2), T.RandomRotation(15),
            T.GaussianBlur(25, sigma=(0.1, 2.0)), T.ToTensor(),
        ])

    def __call__(self, image):
        return self.transform(image), self.transform(image)
