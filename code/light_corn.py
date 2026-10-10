import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mhmt.original_runtime import parse_original_args


def main():
    args = parse_original_args(__file__, "corn", "head_tail", 21, 50)
    import torch
    import torch.nn as nn
    from model.resnet import ResNet18
    import os
    from dataprocess.mydataset import MyDataset
    from dataprocess.augmentation import LeafDiseaseDataAugmentation
    from torch.utils.data import DataLoader
    import datetime
    from utils.train import ModelTrainer
    from utils.test import ModelEvaluator
    from torch.optim import lr_scheduler
    from torch.optim.lr_scheduler import StepLR, _LRScheduler
    import torch.optim as optim
    import random
    import numpy as np
    from torchvision import models

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = str(args.checkpoint)
    backbone_model = ResNet18(10)

    backbone_model.fc = nn.Linear(512, 512)


    class LightConvTail(nn.Module):
        def __init__(self, in_channels, num_classes):
            super(LightConvTail, self).__init__()

            self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(512)
            self.relu1 = nn.ReLU(inplace=True)


            self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(256)
            self.relu2 = nn.ReLU(inplace=True)


            self.conv3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu3 = nn.ReLU(inplace=True)


            self.dropout = nn.Dropout(p=0.3)


            self.gap = nn.AdaptiveAvgPool2d(1)


            self.fc = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))

            x = self.gap(x)
            x = x.view(x.size(0), -1)

            x = self.dropout(x)
            x = self.fc(x)

            return x


    num_new_classes = 4
    tailmlp = LightConvTail(512, num_new_classes)


    class CustomModel(nn.Module):
        def __init__(self, resnet_model, mlp):
            from collections import OrderedDict
            super(CustomModel, self).__init__()

            self.resnet = nn.Sequential(OrderedDict([
                ("0", resnet_model.conv1),
                ("1", resnet_model.bn1),
                ("stem_relu", nn.ReLU(inplace=False)),
                ("stem_pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ("2", resnet_model.layer1),
                ("3", resnet_model.layer2),
                ("4", resnet_model.layer3),
                ("5", resnet_model.layer4),
            ]))
            self.resnet.requires_grad_(False)
            self.resnet.eval()
            self.mlp = mlp

        def train(self, mode=True):

            super().train(mode)
            self.resnet.eval()
            return self

        def forward(self, x):
            x = self.resnet(x)
            x = self.mlp(x)
            return x


    custom_model = CustomModel(backbone_model.model, tailmlp)

    saved_state_dict = torch.load(model_path, map_location=device)

    custom_model.load_state_dict(saved_state_dict, strict=False)


    custom_model.to(device)

    for param in backbone_model.model.parameters():
        param.requires_grad = False

    class SqueezeExcitation(nn.Module):
        def __init__(self, in_channels, reduction):
            super(SqueezeExcitation, self).__init__()
            self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
            self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
            self.relu = nn.ReLU(inplace=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):

            batch_size, channels, _, _ = x.size()
            y = x.view(batch_size, channels, -1).mean(dim=2)


            y = self.fc1(y)
            y = self.relu(y)
            y = self.fc2(y)
            y = self.sigmoid(y).view(batch_size, channels, 1, 1)


            return x * y

    class HeadMLP(nn.Module):
        def __init__(self, in_channels=3, out_channels=3):
            super(HeadMLP, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU(inplace=False)


            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.relu2 = nn.ReLU(inplace=False)

            self.se2 = SqueezeExcitation(32, 2)

            self.conv3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(3)
            self.relu3 = nn.ReLU(inplace=False)

        def forward(self, x):
            x = self.relu1(self.bn1(self.conv1(x)))

            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.se2(x)
            x = self.relu3(self.bn3(self.conv3(x)))
            return x

    in_channels = 3

    out_channels = 3
    headmlp = HeadMLP(in_channels, out_channels)

    headmlp.to(device)

    class MultiheadMultitail(nn.Module):
        def __init__(self, headmlp, custom_model):
            super(MultiheadMultitail, self).__init__()
            self.headmlp = headmlp
            self.custom_model = custom_model

        def forward(self, x):
            x = self.headmlp(x)
            x = self.custom_model(x)
            return x

    multi_model = MultiheadMultitail(headmlp, custom_model).to(device)


    print(multi_model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, multi_model.parameters()), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.85)


    criterion = nn.CrossEntropyLoss()

    data_augmentor = LeafDiseaseDataAugmentation()
    train_dataset = MyDataset(root=str(args.data_root / "train"), transform=data_augmentor.get_transforms('train'))
    test_dataset = MyDataset(root=str(args.data_root / "test"), transform=data_augmentor.get_transforms('test'))


    class_names = train_dataset.classes


    class_info = {i: class_names[i] for i in range(len(class_names))}


    class_file_path = str(args.output_dir / "corn_info.txt")
    with open(class_file_path, 'w') as class_file:
        for idx, name in class_info.items():
            class_file.write(f'Class Index: {idx}, Class Name: {name}\n')

    print('Class information saved to corn_info.txt file.')

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    log_parent_dir = str(args.output_dir)
    log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(log_parent_dir, log_dir), exist_ok=True)

    num_epochs = args.epochs
    max_iters = len(train_loader) * num_epochs


    trainer = ModelTrainer(multi_model, train_loader, val_loader, optimizer, criterion, device, scheduler=scheduler)
    trainer.train_model(num_epochs=num_epochs, log_parent_dir=log_parent_dir, log_dir=log_dir)


    evaluator = ModelEvaluator(multi_model, test_loader, device, log_parent_dir=log_parent_dir, log_dir=log_dir)
    evaluator.evaluate_model()


if __name__ == "__main__":
    main()
