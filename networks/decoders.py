import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        n_classes = args.n_classes
        width_multiplier = args.width_multiplier
        if args.n_layers in [18, 34]:
            n_ch_0 = int(512 * width_multiplier)
            n_ch_1 = int(256 * width_multiplier)
            n_ch_2 = int(128 * width_multiplier)
            n_ch_3 = int(64 * width_multiplier)

        elif args.n_layers in [50, 101]:
            n_ch_0 = int(2048 * width_multiplier)
            n_ch_1 = int(1024 * width_multiplier)
            n_ch_2 = int(512 * width_multiplier)
            n_ch_3 = int(256 * width_multiplier)
        else:
            raise ValueError(args.n_layers)

        self.lat_layer_0 = nn.Conv2d(n_ch_0, 256, 1)
        self.lat_layer_1 = nn.Conv2d(n_ch_1, 256, 1)
        self.lat_layer_2 = nn.Conv2d(n_ch_2, 256, 1)
        self.lat_layer_3 = nn.Conv2d(n_ch_3, 256, 1)

        self.upsample_blocks_0 = nn.Sequential(
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 128),
            UpsampleBlock(128, 128)
        )

        self.upsample_blocks_1 = nn.Sequential(
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 128),
            UpsampleBlock(128, 128)
        )

        self.upsample_blocks_2 = nn.Sequential(
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 128),
            UpsampleBlock(128, 128)
        )

        self.upsample_blocks_3 = nn.Sequential(
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 128)
        )

        self.classifier = nn.Conv2d(128, n_classes, 1)

        self.apply(self._init)

    def forward(self, list_inter_features,  return_output_emb=False):
        c2, c3, c4, c5 = list_inter_features

        c5 = self.lat_layer_0(c5)
        c4 = self.lat_layer_1(c4)
        c3 = self.lat_layer_2(c3)
        c2 = self.lat_layer_3(c2)

        p5 = c5
        p4 = self._upsample_add(p5, c4)
        p3 = self._upsample_add(p4, c3)
        p2 = self._upsample_add(p3, c2)

        p5 = self.upsample_blocks_0(p5)
        p4 = self.upsample_blocks_1(p4)
        p3 = self.upsample_blocks_2(p3)
        p2 = self.upsample_blocks_3(p2)

        emb = p2 + p3 + p4 + p5

        return {"emb": emb, "pred": self.classifier(emb)}

    @staticmethod
    def _upsample_add(x, y):
        _, _, h, w = y.shape
        return F.interpolate(x, size=(h, w), mode="bilinear") + y

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels=128, kernel_size=3, padding=1, n_groups=32, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.ReLU(inplace=True)
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(self.block(x), scale_factor=self.scale_factor, mode="bilinear")


class SegmentHead(nn.Module):
    def __init__(self, args, openset=False):
        super(SegmentHead, self).__init__()
        self.segment_head = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(256),  # 304=256+48
                                          nn.ReLU(),
                                          nn.Dropout(0.5),  # decoder dropout 1
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          nn.Dropout(args.mc_dropout_p))

        self.classifier = nn.Conv2d(256, args.n_classes, 1)
        self._init_weight()
        self.n_classes = args.n_classes

    def forward(self, x):
        emb = self.segment_head(x)
        pred = self.classifier(emb)
        return {"emb": emb, "pred": pred}

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
