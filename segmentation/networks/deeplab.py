import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aspp import ASPP
from .decoders import SegmentHead
from .mobilenet_v2 import MobileNetV2


class DeepLab(nn.Module):
    def __init__(self,
                 args,
                 backbone='mobilenet',
                 output_stride=16,
                 with_mask=False,
                 with_pam=False,
                 branch_early=False):
        super(DeepLab, self).__init__()

        self.backbone = MobileNetV2(output_stride, nn.BatchNorm2d, mc_dropout=args.use_mc_dropout)
        # self.backbone = build_backbone(backbone, output_stride, BatchNorm, mc_dropout)
        self.aspp = ASPP(backbone, output_stride, nn.BatchNorm2d)
        # self.aspp = build_aspp(backbone, output_stride, BatchNorm)

        # low level features
        low_level_inplanes = 24
        self.low_level_conv = nn.Sequential(nn.Conv2d(low_level_inplanes, 48, 1, bias=False),
                                            nn.BatchNorm2d(48),
                                            nn.ReLU())
        # segment
        self.seg_head = SegmentHead(args)

        self.use_contrastive_loss = args.use_contrastive_loss
        if self.use_contrastive_loss:
            self.projection_head = nn.Sequential(
                nn.Conv2d(304, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 256, 1, bias=False),
                nn.BatchNorm2d(256)
            )

        self.use_openset = args.use_openset
        # if self.use_openset:
        #     self.fc = nn.Conv2d(256, args.n_emb_dims, 1)
        if self.use_openset:
            self.aspp_ = ASPP(backbone, output_stride, nn.BatchNorm2d)

            # low level features
            low_level_inplanes = 24
            self.low_level_conv_ = nn.Sequential(nn.Conv2d(low_level_inplanes, 48, 1, bias=False),
                                                nn.BatchNorm2d(48),
                                                nn.ReLU())
            # segment
            self.seg_head_ = SegmentHead(args, openset=True)

        # error mask -> difficulty branch
        self.with_mask = with_mask
        self.branch_early = branch_early
        if with_mask:
            if branch_early:
                self.mask_head = MaskHead_branch(304, args.n_classes, nn.BatchNorm2d, with_pam)
            else:
                self.mask_head = MaskHead(args.n_classes, with_pam)

        self.return_features = False
        self.return_attention = False

        self.use_img_inp = args.use_img_inp
        self.use_visual_acuity = args.use_visual_acuity
        self.use_softmax = args.use_softmax

    def turn_on_dropout(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def turn_off_dropout(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()

    def forward(self, inputs):
        backbone_feat, low_level_feat = self.backbone(inputs)  # 1/16, 1/4;
        x = self.aspp(backbone_feat)  # 1/16 -> aspp -> 1/16

        # low + high features
        low_level_feat_ = self.low_level_conv(low_level_feat)  # 256->48
        x = F.interpolate(x, size=low_level_feat_.size()[2:], mode='bilinear', align_corners=True)  # 1/4
        second_to_last_features = torch.cat((x, low_level_feat_), dim=1)  # 304 = 256 + 48

        # segment
        dict_outputs = self.seg_head(second_to_last_features)
        if self.use_contrastive_loss:
            projection = self.projection_head(second_to_last_features)
            projection = F.interpolate(projection, size=inputs.size()[2:], mode='bilinear', align_corners=True)
            dict_outputs.update({"projection": projection})

        if self.with_mask:
            if self.branch_early:
                mask, attention = self.mask_head(second_to_last_features)  # 1/4 features same to seg_head
            else:
                mask, attention = self.mask_head(pred)  # segment output

            pred = F.interpolate(pred, size=inputs.size()[2:], mode='bilinear', align_corners=True)
            mask = F.interpolate(mask, size=inputs.size()[2:], mode='bilinear', align_corners=True)  # nearest can't use align_corners
            mask = torch.sigmoid(mask)
            if self.return_attention:
                return pred, mask, attention
            else:
                return pred, mask
        else:
            pred = dict_outputs['pred']
            pred = F.interpolate(pred, size=inputs.size()[2:], mode='bilinear', align_corners=True)
            dict_outputs['pred'] = pred

            emb = dict_outputs['emb']
            emb = F.interpolate(emb, size=inputs.size()[2:], mode='bilinear', align_corners=True)
            dict_outputs['emb'] = emb

            # if self.use_openset:
            #     emb_ = self.fc(dict_outputs["emb"])
            #     emb_ = F.interpolate(emb_, size=inputs.size()[2:], mode='bilinear', align_corners=True)
            #     dict_outputs['emb_'] = emb_
            if self.use_openset:
                x = self.aspp_(backbone_feat)  # 1/16 -> aspp -> 1/16

                # low + high features
                low_level_feat_ = self.low_level_conv_(low_level_feat)  # 256->48
                x = F.interpolate(x, size=low_level_feat_.size()[2:], mode='bilinear', align_corners=True)  # 1/4
                second_to_last_features = torch.cat((x, low_level_feat_), dim=1)  # 304 = 256 + 48

                dict_outputs_ = self.seg_head_(second_to_last_features)
                emb_ = dict_outputs_["emb"]
                dict_outputs.update({"emb_": F.interpolate(emb_, size=inputs.size()[2:],
                                                           mode='bilinear', align_corners=True)})

            if self.return_features:
                return dict_outputs
                # return pred, second_to_last_features  # for coreset
            else:
                return dict_outputs
                # return pred

    def set_return_features(self, return_features):  # True or False
        self.return_features = return_features

    def set_return_attention(self, return_attention):  # True or False
        self.return_attention = return_attention

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.low_level_conv, self.seg_head]
        if self.with_mask:
            modules.append(self.mask_head)
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def load_pretrain(self, pretrained):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')['state_dict']
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}  # 不加载最后的 head 参数
            # for k, v in pretrained_dict.items():
            #     print('=> loading {} | {}'.format(k, v.size()))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            print('No such file {}'.format(pretrained))