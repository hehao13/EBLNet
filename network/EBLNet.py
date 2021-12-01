import torch
import torch.nn as nn
import torch.nn.functional as F

from network import resnet_d as Resnet_Deep
from network.resnext import resnext101_32x8
from network.nn.mynn import Norm2d
from network.nn.contour_point_gcn import ContourPointGCN
from network.nn.operators import _AtrousSpatialPyramidPoolingModule


class Edge_extractorWofirstext(nn.Module):
    def __init__(self, inplane, skip_num, norm_layer):
        super(Edge_extractorWofirstext, self).__init__()
        self.skip_mum = skip_num
        self.pre_extractor = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3,
                      padding=1, groups=1, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.extractor = nn.Sequential(
            nn.Conv2d(inplane + skip_num, inplane, kernel_size=3,
                      padding=1, groups=8, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )

    def forward(self, aspp, layer1):  # 200        # 100
        seg_edge = torch.cat([F.interpolate(aspp, size=layer1.size()[2:], mode='bilinear',
                                            align_corners=True), layer1], dim=1)  # 200
        seg_edge = self.extractor(seg_edge)  # 200
        seg_body = F.interpolate(aspp, layer1.size()[2:], mode='bilinear', align_corners=True) - seg_edge

        return seg_edge, seg_body


class EBLNet(nn.Module):
    """
    Implement deeplabv3 plus module without depthwise conv
    A: stride=8
    B: stride=16
    with skip connection
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48, num_cascade=4,
                 num_points=96, threshold=0.8):
        super(EBLNet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_mum = skip_num
        self.num_cascade = num_cascade
        self.num_points = num_points
        self.threshold = threshold

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-101-32x8':
            resnet = resnext101_32x8()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError('Only support resnet50 and resnet101 for now')

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            print('Not using dilation')

        self.aspp = _AtrousSpatialPyramidPoolingModule(in_dim=2048, reduction_dim=256,
                                                       output_stride=8 if self.variant == 'D' else 16)

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_mum, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_mum, kernel_size=1, bias=False)
        else:
            raise ValueError('Not a valid skip')
        self.body_fines = nn.ModuleList()
        for i in range(self.num_cascade):
            inchannels = 2 ** (11 - i)
            self.body_fines.append(nn.Conv2d(inchannels, 48, kernel_size=1, bias=False))
        self.body_fuse = [nn.Conv2d(256 + 48, 256, kernel_size=1, bias=False) for _ in range(self.num_cascade)]
        self.body_fuse = nn.ModuleList(self.body_fuse)

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.edge_extractors = [Edge_extractorWofirstext(256, norm_layer=Norm2d, skip_num=48)
                                for _ in range(self.num_cascade)]
        self.edge_extractors = nn.ModuleList(self.edge_extractors)

        self.refines = [ContourPointGCN(256, self.num_points, self.threshold) for _ in range(self.num_cascade)]
        self.refines = nn.ModuleList(self.refines)

        self.edge_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.edge_out_pre = nn.ModuleList(self.edge_out_pre)
        self.edge_out = nn.ModuleList([nn.Conv2d(256, 1, kernel_size=1, bias=False)
                                       for _ in range(self.num_cascade)])

        self.body_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.body_out_pre = nn.ModuleList(self.body_out_pre)
        self.body_out = nn.ModuleList([nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
                                       for _ in range(self.num_cascade)])

        self.final_seg_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade - 1)]
        self.final_seg_out_pre.append(nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)))
        self.final_seg_out_pre = nn.ModuleList(self.final_seg_out_pre)
        self.final_seg_out = nn.ModuleList([nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
                                            for _ in range(self.num_cascade)])

    def forward(self, x, gts=None):

        x_size = x.size()  # 800
        feats = []
        feats.append(self.layer0(x))  # 200
        feats.append(self.layer1(feats[0]))  # 200, 200
        feats.append(self.layer2(feats[1]))  # 200, 200, 100
        feats.append(self.layer3(feats[2]))  # 200, 200, 100, 100
        feats.append(self.layer4(feats[3]))  # 200, 200, 100, 100, 100
        aspp = self.aspp(feats[-1])  # 100
        fine_size = feats[1].size()  # 200

        seg_edges = []
        seg_edge_outs = []
        seg_bodys = []
        seg_body_outs = []
        seg_finals = []
        seg_final_outs = []
        aspp = self.bot_aspp(aspp)  # 100
        final_fuse_feat = F.interpolate(aspp, size=fine_size[2:], mode='bilinear', align_corners=True)  # 200

        low_feat = self.bot_fine(feats[1])  # 200

        for i in range(self.num_cascade):
            if i == 0:
                last_seg_feat = aspp  # 100
            else:
                last_seg_feat = seg_finals[-1]  # 200
                last_seg_feat = F.interpolate(last_seg_feat, size=aspp.size()[2:],
                                              mode='bilinear', align_corners=True)  # 100

            seg_edge, seg_body = self.edge_extractors[i](last_seg_feat, low_feat)  # 200

            high_fine = F.interpolate(self.body_fines[i](feats[-(i + 1)]), size=fine_size[2:], mode='bilinear',
                                      align_corners=True)  # 200
            seg_body = self.body_fuse[i](torch.cat([seg_body, high_fine], dim=1))  # 200
            seg_body_pre = self.body_out_pre[i](seg_body)
            seg_body_out = F.interpolate(self.body_out[i](seg_body_pre), size=x_size[2:],
                                         mode='bilinear', align_corners=True)  # 800
            seg_bodys.append(seg_body_pre)
            seg_body_outs.append(seg_body_out)

            seg_edge_pre = self.edge_out_pre[i](seg_edge)  # 200
            seg_edge_out_pre = self.edge_out[i](seg_edge_pre)
            seg_edge_out = F.interpolate(seg_edge_out_pre, size=x_size[2:],
                                         mode='bilinear', align_corners=True)  # 800
            seg_edges.append(seg_edge_pre)
            seg_edge_outs.append(seg_edge_out)

            seg_out = seg_body + seg_edge  # 200
            seg_out = self.refines[i](seg_out, torch.sigmoid(seg_edge_out_pre.clone().detach()))

            if i >= self.num_cascade - 1:
                seg_final_pre = self.final_seg_out_pre[i](torch.cat([final_fuse_feat, seg_out], dim=1))
            else:
                seg_final_pre = self.final_seg_out_pre[i](seg_out)
            seg_final_out = F.interpolate(self.final_seg_out[i](seg_final_pre), size=x_size[2:],
                                          mode='bilinear', align_corners=True)
            seg_finals.append(seg_final_pre)
            seg_final_outs.append(seg_final_out)

        if self.training:
            return self.criterion((seg_final_outs, seg_body_outs, seg_edge_outs), gts)

        return seg_final_outs[-1]


def EBLNet_resnet50_os8(num_classes, criterion, num_cascade=1, num_points=96, threshold=0.85):
    """
    ResNet-50 Based Network with stride=8 and cascade model
    """
    return EBLNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1',
                  num_cascade=num_cascade, num_points=num_points, threshold=threshold)


def EBLNet_resnet101_os8(num_classes, criterion, num_cascade=1, num_points=96, threshold=0.85):
    """
    ResNet-101 Based Network with stride=8 and cascade model
    """
    return EBLNet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1',
                  num_cascade=num_cascade, num_points=num_points, threshold=threshold)


def EBLNet_resnet50_os16(num_classes, criterion, num_cascade=1, num_points=96, threshold=0.85):
    """
    ResNet-50 Based Network with stride=16 and cascade model
    """
    return EBLNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D16', skip='m1',
                  num_cascade=num_cascade, num_points=num_points, threshold=threshold)


def EBLNet_resnext101_os8(num_classes, criterion, num_cascade=1, num_points=96, threshold=0.85):
    """
    ResNeXt-101 Based Network with stride=8 and cascade model
    """
    return EBLNet(num_classes, trunk='resnext-101-32x8', criterion=criterion, variant='D', skip='m1',
                  num_cascade=num_cascade, num_points=num_points, threshold=threshold)
