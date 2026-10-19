"""Only the selected constructor is embedded in a training notebook."""
import torch
from torch import nn


def resnet18_encoder(pretrained=True):
    from torchvision.models import ResNet18_Weights, resnet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Identity()
    return model


def dinov2_encoder(pretrained=True):
    import os
    os.environ["XFORMERS_DISABLED"] = "1"
    return torch.hub.load("facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8",
                          "dinov2_vits14", pretrained=pretrained, trust_repo=True, verbose=False)


def swin_t_encoder(pretrained=True):
    from torchvision.models import Swin_T_Weights, swin_t
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
    model.head = nn.Identity()
    return model


def convnext_tiny_encoder(pretrained=True):
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[2] = nn.Identity()
    return model


def vit_b16_encoder(pretrained=True):
    from torchvision.models import ViT_B_16_Weights, vit_b_16
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    model.heads = nn.Identity()
    return model
