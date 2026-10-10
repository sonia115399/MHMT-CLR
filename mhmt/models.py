import inspect
import torch
from torch import nn
from torchvision.models import resnet18

SPECIES = {"corn": 4, "datepalm": 3, "tea": 8}


def make_resnet18():

    if "weights" in inspect.signature(resnet18).parameters:
        return resnet18(weights=None)
    return resnet18(pretrained=False)


def conv_block(in_channels, out_channels, kernel=3, stride=1, padding=1):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False),
    )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        gate = self.gate(x.mean(dim=(2, 3)))[:, :, None, None]
        return x * gate


class SpeciesHead(nn.Module):
    def __init__(self, species):
        super().__init__()
        first, second, reduction = {
            "corn": (32, 32, 2), "datepalm": (32, 32, 4), "tea": (16, 8, 2)
        }[species]
        self.layers = nn.Sequential(
            conv_block(3, first, stride=2), conv_block(first, second),
            SqueezeExcitation(second, reduction), conv_block(second, 3),
        )

    def forward(self, x):
        return self.layers(x)


class SpeciesTail(nn.Module):
    def __init__(self, species, use_tail=True):
        super().__init__()
        if not use_tail:
            self.layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 512),
                nn.ReLU(), nn.Linear(512, SPECIES[species]),
            )
        elif species == "corn":
            self.layers = nn.Sequential(
                conv_block(512, 512, stride=2), conv_block(512, 256, stride=2),
                conv_block(256, 128, kernel=1, padding=0),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                nn.Linear(128, SPECIES[species]),
            )
        else:
            self.layers = nn.Sequential(
                conv_block(512, 128, stride=2), conv_block(128, 64, stride=2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5),
                nn.Linear(64, SPECIES[species]),
            )

    def forward(self, x):
        return self.layers(x)


class SourceModel(nn.Module):


    def __init__(self, method):
        super().__init__()
        self.backbone = make_resnet18()
        if method == "clr":
            self.backbone.fc = nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)
            )
        else:
            self.backbone.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.backbone(x)


class MHMT(nn.Module):


    def __init__(self, species, use_head=True, use_tail=True):
        super().__init__()
        backbone = make_resnet18()
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.head = SpeciesHead(species) if use_head else nn.Identity()
        self.tail = SpeciesTail(species, use_tail)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)

        self.encoder.eval()
        return self

    def forward(self, x):

        return self.tail(self.encoder(self.head(x)))

    def load_source(self, path, method):
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("method") != method or checkpoint.get("stage") != "pretrain":
            raise ValueError("请提供对应方法的源域预训练 checkpoint（本版本格式）。")
        source = SourceModel(method)
        source.load_state_dict(checkpoint["state_dict"], strict=True)
        encoder = nn.Sequential(*list(source.backbone.children())[:-2])
        self.encoder.load_state_dict(encoder.state_dict(), strict=True)
