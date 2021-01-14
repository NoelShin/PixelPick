import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        n_classes = args.n_classes
        use_softmax = args.use_softmax
        use_img_inp = args.use_img_inp

        self.lat_layer_0 = nn.Conv2d(2048, 256, 1)
        self.lat_layer_1 = nn.Conv2d(1024, 256, 1)
        self.lat_layer_2 = nn.Conv2d(512, 256, 1)
        self.lat_layer_3 = nn.Conv2d(256, 256, 1)

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

        if use_softmax:
            self.classifier = nn.Conv2d(128, n_classes, 1)

        else:
            self.fc = nn.Conv2d(128, args.n_emb_dims, 1)

        if use_img_inp:
            self.fc_img_inp = nn.Conv2d(128, 3, 1)

        self.apply(self._init)
        self.use_softmax = use_softmax
        self.use_img_inp = use_img_inp

    def forward(self, list_inter_features,  return_output_emb=False):
        dict_outputs = dict()

        c2, c3, c4, c5 = list_inter_features
        p5 = self.lat_layer_0(c5)
        p4 = self._upsample_add(p5, self.lat_layer_1(c4))
        p3 = self._upsample_add(p4, self.lat_layer_2(c3))
        p2 = self._upsample_add(p3, self.lat_layer_3(c2))

        p5 = self.upsample_blocks_0(p5)
        p4 = self.upsample_blocks_1(p4)
        p3 = self.upsample_blocks_2(p3)
        p2 = self.upsample_blocks_3(p2)

        emb = p2 + p3 + p4 + p5

        if self.use_softmax:
            dict_outputs.update({"pred": self.classifier(emb)})
        else:
            dict_outputs.update({"emb": self.fc(emb)})

        if self.use_img_inp:
            dict_outputs.update({"img_inp": self.fc_img_inp(emb)})
        return dict_outputs

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
        # self.size = size

    def forward(self, x):
        return F.interpolate(self.block(x), scale_factor=self.scale_factor, mode="bilinear")


class SegmentHead(nn.Module):
    def __init__(self, args):
        super(SegmentHead, self).__init__()
        self.segment_head = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(256),  # 304=256+48
                                          nn.ReLU(),
                                          nn.Dropout(0.5),  # decoder dropout 1
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          nn.Dropout(args.mc_dropout_p))  # , constants.MC_DROPOUT_RATE),  # MC dropout
                                          # nn.Conv2d(256, args.n_classes, kernel_size=1, stride=1))

        if args.use_softmax:
            self.classifier = nn.Conv2d(256, args.n_classes, 1)

        else:
            self.fc = nn.Conv2d(256, args.n_emb_dims, 1)

        if args.use_img_inp or args.use_visual_acuity:
            self.fc_img_inp = nn.Conv2d(256, 3, 1)

        self._init_weight()

        self.n_classes = args.n_classes
        self.use_img_inp = args.use_img_inp
        self.use_visual_acuity = args.use_visual_acuity

        self.use_softmax = args.use_softmax

    def forward(self, x):
        dict_outputs = {}
        emb = self.segment_head(x)
        if self.use_img_inp or self.use_visual_acuity:
            img_inp = self.fc_img_inp(emb)
            dict_outputs.update({"img_inp": img_inp})

        if self.use_softmax:
            pred = self.classifier(emb)
            dict_outputs.update({"pred": pred})

        else:
            emb = self.fc(emb)
            dict_outputs.update({"emb": emb})
        return dict_outputs

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()