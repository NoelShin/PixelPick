import torch.nn as nn
from networks.backbones.resnet_backbone import ResNetBackbone

resnet50 = {
    "supervised": "../networks/backbones/pretrained/resnet50-pytorch.pth",
    "moco_v2": "../networks/backbones/pretrained/moco_v2_800ep_pretrain.pth.tar",
}

resnet = {
    18: "../networks/backbones/pretrained/resnet18-pytorch.pth",
    34: "../networks/backbones/pretrained/resnet34-pytorch.pth",
    50: "../networks/backbones/pretrained/resnet50-pytorch.pth",
    101: "../networks/backbones/pretrained/resnet101-pytorch.pth",
    "moco_v2": "../networks/backbones/pretrained/moco_v2_800ep_pretrain.pth.tar"
}


class Encoder(nn.Module):
    def __init__(self, args, load_pretrained):
        super(Encoder, self).__init__()
        weight_type = args.weight_type
        use_dilated_resnet = args.use_dilated_resnet
        n_layers = args.n_layers

        if load_pretrained:
            if weight_type == "supervised":
                if load_pretrained:
                    self.base = ResNetBackbone(backbone=f'resnet{n_layers}_dilated8', pretrained=resnet[n_layers])
                    print(f"Encoder initialised with supervised weights.")
            else:
                self.base = ResNetBackbone(backbone='resnet50_dilated8', pretrained=None)

        else:
            self.base = ResNetBackbone(backbone=f'resnet{n_layers}_dilated8', pretrained=None,
                                       width_multiplier=args.width_multiplier)

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

