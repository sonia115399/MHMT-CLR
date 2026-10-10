import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mhmt.original_runtime import parse_original_args


def main():
    args = parse_original_args(__file__, "tea", "head_tail", 42, 45)
    import torch
    import torch.nn as nn
    from models.resnet_simclr import ResNetSimCLR
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
    import os
    import torch
    from dataprocess.mydataset import MyDataset
    from dataprocess.augmentation import LeafDiseaseDataAugmentation
    from torchinfo import summary
    from thop import profile
    import time
    from memory_profiler import memory_usage

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = str(args.checkpoint)
    backbone_model = ResNetSimCLR(base_model='resnet18', out_dim=128)
    backbone_model.backbone.fc = nn.Linear(512, 512)

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


    num_new_classes = 8
    dropout_rate = 0.5
    tailmlp = LightConvTail(in_channels=512, num_classes=num_new_classes)

    input = torch.randn(1, 512, 4, 4)
    flops, params = profile(tailmlp, inputs=(input,))
    print(f"FLOPs: {flops/1e9:.2f}G")


    class CustomModel(nn.Module):
        def __init__(self, backbone_model, mlp):
            super(CustomModel, self).__init__()
            self.resnet = nn.Sequential(*list(backbone_model.backbone.children())[:-2])
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


    custom_model = CustomModel(backbone_model, tailmlp)
    saved_state_dict = torch.load(model_path, map_location=device)
    keys_to_remove = ['backbone_model.backbone.conv1.weight', 'backbone_model.backbone.conv1.bias',
                        'backbone_model.backbone.bn1.weight', 'backbone_model.backbone.bn1.bias', 'backbone_model.backbone.bn1.running_mean',
                        'backbone_model.backbone.bn1.running_var', 'backbone_model.backbone.bn1.num_batches_tracked']
    for key in keys_to_remove:
        saved_state_dict.pop(key, None)

    custom_model.load_state_dict(saved_state_dict, strict=False)


    custom_model.to(device)

    for param in backbone_model.parameters():
        param.requires_grad = False

    class SqueezeExcitation(nn.Module):
        def __init__(self, in_channels, reduction):
            super(SqueezeExcitation, self).__init__()
            self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
            self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
            self.relu = nn.ReLU(inplace=True)
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
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu1 = nn.ReLU(inplace=False)


            self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(8)
            self.relu2 = nn.ReLU(inplace=False)

            self.se2 = SqueezeExcitation(8, 2)

            self.conv3 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)
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

    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(headmlp, inputs=(input.to(device),))
    print(f"FLOPs: {flops/1e9:.2f}G")

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

    def measure_inference_time(model, input_tensor):

        model.eval()


        start_time = time.time()


        with torch.no_grad():
            output = model(input_tensor)


        end_time = time.time()


        inference_time = end_time - start_time
        return inference_time

    input_tensor = torch.randn(1, 512, 4, 4).to(device)
    inference_time = measure_inference_time(multi_model.custom_model.mlp, input_tensor)
    print(f"Inference Time: {inference_time:.6f} seconds")


    print(multi_model)

    lr = 0.0005
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, multi_model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.85)
    summary(multi_model.custom_model.mlp, input_size=(1, 512, 4, 4))


    criterion = nn.CrossEntropyLoss()

    data_augmentor = LeafDiseaseDataAugmentation()
    train_dataset = MyDataset(root=str(args.data_root / "train"), transform=data_augmentor.get_transforms('train'))
    test_dataset = MyDataset(root=str(args.data_root / "test"), transform=data_augmentor.get_transforms('test'))


    class_names = train_dataset.classes


    class_info = {i: class_names[i] for i in range(len(class_names))}


    class_file_path = str(args.output_dir / "tea_info.txt")
    with open(class_file_path, 'w') as class_file:
        for idx, name in class_info.items():
            class_file.write(f'Class Index: {idx}, Class Name: {name}\n')

    print('Class information saved to tea_info.txt file.')

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
