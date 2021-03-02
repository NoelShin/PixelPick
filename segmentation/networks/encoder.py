import torch.nn as nn
from networks.backbones.resnet_backbone import ResNetBackbone

resnet50 = {
    "supervised": "../networks/backbones/pretrained/resnet50-pytorch.pth",
    # "supervised": "/home/gishin-temp/projects/open_set/segmentation/networks/backbones/pretrained/resnet50-pytorch.pth",
    "deepcluster_v2": "../networks/backbones/pretrained/deepclusterv2_400ep_pretrain.pth.tar",
    "moco_v2": "../networks/backbones/pretrained/moco_v2_800ep_pretrain.pth.tar",
    "swav": "../networks/backbones/pretrained/swav_800ep_pretrain.pth.tar"
}

resnet = {
    18: "../networks/backbones/pretrained/resnet18-pytorch.pth",
    34: "../networks/backbones/pretrained/resnet34-pytorch.pth",
    # 50: "../networks/backbones/pretrained/resnet50-pytorch.pth",
    50: "/home/gishin-temp/projects/open_set/segmentation/networks/backbones/pretrained/resnet50-pytorch.pth",
    101: "../networks/backbones/pretrained/resnet101-pytorch.pth",
    "moco_v2": "../networks/backbones/pretrained/moco_v2_800ep_pretrain.pth.tar"
}


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        weight_type = args.weight_type
        use_dilated_resnet = args.use_dilated_resnet
        n_layers = args.n_layers

        if weight_type == "supervised":
            self.base = ResNetBackbone(backbone=f'resnet{n_layers}_dilated8', pretrained=resnet[n_layers])
            # pretrained=resnet50[weight_type])

        else:
            self.base = ResNetBackbone(backbone='resnet50_dilated8', pretrained=None)
            # if use_dilated_resnet:
            #     self.base = ResNetBackbone(backbone='resnet50_dilated8', pretrained=None)

            # else:
            #     model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained=None)
            #     self.base = nn.Sequential(nn.Sequential(model.prefix, model.maxpool),
            #                               model.layer1,
            #                               model.layer2,
            #                               model.layer3,
            #                               model.layer4)

        print(f"Encoder initialised with {weight_type} weights.")

        self.weight_type = weight_type
        self.use_fpn = use_dilated_resnet

    def forward_backbone(self, x):
        return self.model.module.forward_backbone(x)

    def forward(self, x):
        if self.weight_type in ["random", "supervised", "moco_v2"]:
            x = self.base(x)

        else:
            x = self.model.module.forward_backbone(x, return_inter_features=self.use_fpn)
        return x

    def get_backbone_params(self):
        if self.weight_type == "self-supervised":
            return self.model.parameters()

        else:
            return self.base.parameters()

