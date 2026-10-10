import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mhmt.original_runtime import parse_original_args


def main():
    args = parse_original_args(__file__, "datepalm", "tail", 42, 30)
    import torch
    import torch.nn as nn
    from models.resnet_simclr import ResNetSimCLR
    import os
    from torch.utils.data import DataLoader
    import datetime
    from util.train import ModelTrainer
    from util.test import ModelEvaluator
    from torch.optim import lr_scheduler
    from torch.optim.lr_scheduler import StepLR
    import torch.optim as optim
    import random
    import numpy as np
    from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
    from dataprocess.mydataset import MyDataset
    from dataprocess.augmentation import LeafDiseaseDataAugmentation


    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_path = str(args.checkpoint)


    model = ResNetSimCLR(base_model='resnet18', out_dim=128)


    checkpoint = torch.load(model_path)
    pretrained_dict = checkpoint['state_dict']


    model.load_state_dict(pretrained_dict)


    for param in model.backbone.parameters():
        param.requires_grad = False


    class LightConvTail(nn.Module):
        def __init__(self, in_channels, num_classes):
            super(LightConvTail, self).__init__()

            self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(128)
            self.relu1 = nn.ReLU(inplace=True)


            self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU(inplace=True)


            self.dropout = nn.Dropout(p=0.5)


            self.gap = nn.AdaptiveAvgPool2d(1)


            self.fc = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))


            x = self.gap(x)
            x = x.view(x.size(0), -1)

            x = self.dropout(x)
            x = self.fc(x)

            return x


    num_new_classes = 3
    tail = LightConvTail(512, num_new_classes)


    class TransferLearningModel(nn.Module):
        def __init__(self, resnet_model, tail):
            super(TransferLearningModel, self).__init__()
            self.resnet = nn.Sequential(*list(resnet_model.backbone.children())[:-2])
            self.tail = tail

        def forward(self, x):

            x = self.resnet(x)

            x = self.tail(x)
            return x


    transfer_model = TransferLearningModel(model, tail).to(device)


    print(transfer_model)


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.85)


    criterion = nn.CrossEntropyLoss()
    data_augmentor = LeafDiseaseDataAugmentation()
    train_dataset = MyDataset(root=str(args.data_root / "train"), transform=data_augmentor.get_transforms('train'))
    test_dataset = MyDataset(root=str(args.data_root / "test"), transform=data_augmentor.get_transforms('test'))


    class_names = train_dataset.classes


    class_info = {i: class_names[i] for i in range(len(class_names))}


    class_file_path = str(args.output_dir / "datepalm_info.txt")
    with open(class_file_path, 'w') as class_file:
        for idx, name in class_info.items():
            class_file.write(f'Class Index: {idx}, Class Name: {name}\n')

    print('Class information saved to datepalm_info.txt file.')

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    log_parent_dir = str(args.output_dir)
    log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(log_parent_dir, 'datepalm', log_dir), exist_ok=True)

    num_epochs = args.epochs
    max_iters = len(train_loader) * num_epochs


    trainer = ModelTrainer(transfer_model, train_loader, val_loader, optimizer, criterion, device, scheduler=None)
    trainer.train_model(num_epochs=num_epochs, log_parent_dir=log_parent_dir, log_dir=log_dir)


    evaluator = ModelEvaluator(transfer_model, test_loader, device, log_parent_dir=log_parent_dir, log_dir=log_dir)
    evaluator.evaluate_model()


if __name__ == "__main__":
    main()
