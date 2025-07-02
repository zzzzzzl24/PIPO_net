# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np


from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import seg_configs as configs
from .seg_modeling_resnet_skip import ResNetV2
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import time
logger = logging.getLogger(__name__)

gaussian_att = False
visualization = False
tsne_points = []
tsne_labels = []
tsne_colors = []

def plot_embedding_2D(data, label, colors, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(colors[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

class CompositionalLayer(nn.Module):
    def __init__(self, normalization_sign=False):
        super().__init__()
        self.normalization = normalization_sign

        # Compose two modality features using a 2D convolution
        # weight_std is not used here, kept for compatibility
        self.conv = nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1, bias=False)

    def forward(self, f1, f2):
        """
        :param f1: shared-modality features, shape (B, C, H, W)
        :param f2: specific-modality features, shape (B, C, H, W)
        :return: composed features
        """
        if self.normalization:
            f1_n = F.normalize(f1, dim=1)
            f2_n = F.normalize(f2, dim=1)
            residual = torch.cat((f1_n, f2_n), dim=1)
        else:
            residual = torch.cat((f1, f2), dim=1)

        residual = self.conv(residual)
        features = f1 + residual  # Combine shared and residual information

        return features


class ASPP2D(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ASPP2D, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(inplace=True),
        )

        # Branch 2: 3x3 convolution with dilation=2
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(inplace=True),
        )

        # Branch 3: 3x3 convolution with dilation=4
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(inplace=True),
        )

        # Branch 4: 3x3 convolution with dilation=8
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(inplace=True),
        )

        # Branch 5: global average pooling + 1x1 convolution + upsampling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)
        self.branch5_norm = nn.BatchNorm2d(dim_out)
        self.branch5_act = nn.LeakyReLU(inplace=True)

        # Final 1x1 convolution after concatenation
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        size = x.size()[2:]  # get height and width

        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)

        # Global feature branch
        global_feat = self.global_pool(x)
        global_f = global_feat  # optional: keep for external use
        global_feat = self.branch5_conv(global_feat)
        global_feat = self.branch5_norm(global_feat)
        global_feat = self.branch5_act(global_feat)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=True)

        # Concatenate features from all branches
        concat_feat = torch.cat([feat1, feat2, feat3, feat4, global_feat], dim=1)

        # Final projection
        output = self.conv_cat(concat_feat)
        return output, global_f#输出各分支的融合（24,256,8,8,）以及x在size维度上avgpooling后的（24,256,1,1）

class U_Res2D_enc(nn.Module):
    def __init__(self, config):
        super(U_Res2D_enc, self).__init__()
        self.config = config
        self.backbone = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)#resNet网络，包括瓶颈结构和GN层
        self.dropout = Dropout(config["dropout_rate"])
        self.asppreduce = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        self.aspp = ASPP2D(256, 256)

    def forward(self, image):
        x, features = self.backbone(image)
        x = self.asppreduce(x)  # Reduce the number of channels to 256
        global_f_avg, global_f = self.aspp(x)
        return global_f_avg, global_f, features

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            768,
            512,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])#[512, 256, 128, 64]
        out_channels = decoder_channels # (256, 128, 64, 16)

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels #[512, 256, 64, 16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]#配置跳跃连接，是否使用以及使用多少个

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]#构建解码
        self.blocks = nn.ModuleList(blocks)

    def forward(self, enc_out, enc_features):
        x = self.conv_more(enc_out).contiguous()#x:(24,512,8,8)
        for i, decoder_block in enumerate(self.blocks):
            if enc_features is not None:
                skip = enc_features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1, stride=1, bias=bias)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x_A, x_B, x_C):
        x = torch.concat([x_A, x_B, x_C], dim=1).contiguous()
        x = self.fusion_proj(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Fusion_Embed2(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed2, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1).contiguous()
        x = self.fusion_proj(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class PIPO_Model(nn.Module):
    def __init__(self, config, img_size=224, self_att=True, cross_att=True, num_classes=None, normalization_sign=True):
        super(PIPO_Model, self).__init__()
        #basic_config
        self.config = config
        self.self_att = self_att
        self.cross_att = cross_att
        if self_att or cross_att:
            h = w = int(img_size)
            embed_dim = 256
            self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim*3, num_heads=8, dropout=0.1).cuda()
            self.multihead_attn1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.1).cuda()
            self.multihead_attn2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.1).cuda()
            self.multihead_attn3 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.1).cuda()

        self.num_classes = num_classes  #类别数        
        self.classifier = config.classifier  #定义分类器
        
        #shared_encoder
        self.transformer = U_Res2D_enc(config)
        
        #specific_encoder
        self.transformer1 = U_Res2D_enc(config)
        self.transformer2 = U_Res2D_enc(config)
        self.transformer3 = U_Res2D_enc(config)
        
        #shared_specific_fusion
        self.compos_layer = CompositionalLayer(normalization_sign)

        #decoder
        self.decoder = DecoderCup(config)
        
        #dimensionality operation
        self.feature_fusion = nn.Sequential(OrderedDict([
            ('fusion1', Fusion_Embed(512)),
            ('fusion2', Fusion_Embed(256)),
            ('fusion3', Fusion_Embed(64))
        ]))
        
        self.feature_fusion2 = nn.Sequential(OrderedDict([
            ('fusion1', Fusion_Embed(512)),
            ('fusion2', Fusion_Embed(256)),
            ('fusion3', Fusion_Embed(64))
        ]))

        self.feature_fusion3 = nn.Sequential(OrderedDict([
            ('fusion1', Fusion_Embed2(512)),
            ('fusion2', Fusion_Embed2(256)),
            ('fusion3', Fusion_Embed2(64))
        ]))

        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
       
        self.fuse_mlp = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )

        self.dom_classifier = nn.Linear(in_features=256, out_features=4, bias=True)

    def compute_shared_features(self, mode_Type, X1_share_f_avg, X2_share_f_avg, X3_share_f_avg):
        share_list = []
        B = mode_Type.shape[0]
        for i in range(B):
            available_feats = []
            if mode_Type[i, 0] == 1:
                available_feats.append(X1_share_f_avg[i])
            if mode_Type[i, 1] == 1:
                available_feats.append(X2_share_f_avg[i])
            if mode_Type[i, 2] == 1:
                available_feats.append(X3_share_f_avg[i])
            if len(available_feats) > 0:
                avg_feat = torch.stack(available_feats, dim=0).mean(dim=0)
                share_list.append(avg_feat)
            else:
                raise ValueError(f"样本{i}没有可用模态")
        return torch.stack(share_list, dim=0)

    def fuse_modalities_per_sample(self,
        mode_Type,
        share_f_avg,
        X1_share_f_avg, X1_spec_f_avg, 
        X2_share_f_avg, X2_spec_f_avg, 
        X3_share_f_avg, X3_spec_f_avg
        ):
        """
        对每个样本的三模态特征进行融合处理，支持模态缺失。

        参数说明：
        - mode_Type: shape [B, 3],每行代表三个模态的可用性(1=可用,0=缺失)
        - share_f_avg: shape [B, C, H, W]，融合后的共享特征
        - Xn_share_f_avg, Xn_spec_f_avg: shape [B, C, H, W],模态n的共享/特有特征

        返回：
        - X1_fused_ft, X2_fused_ft, X3_fused_ft:融合后的模态特征,shape均为 [B, C, H, W]
        """
        X1_fused_list, X2_fused_list, X3_fused_list = [], [], []
        B = mode_Type.shape[0]

        for i in range(B):
            share_i = share_f_avg[i]

            # 模态1
            if mode_Type[i, 0] == 1:
                x1 = self.compos_layer(X1_share_f_avg[i].unsqueeze(0), X1_spec_f_avg[i].unsqueeze(0))[0]
            else:
                x1 = share_i
            X1_fused_list.append(x1)

            # 模态2
            if mode_Type[i, 1] == 1:
                x2 = self.compos_layer(X2_share_f_avg[i].unsqueeze(0), X2_spec_f_avg[i].unsqueeze(0))[0]
            else:
                x2 = share_i
            X2_fused_list.append(x2)

            # 模态3
            if mode_Type[i, 2] == 1:
                x3 = self.compos_layer(X3_share_f_avg[i].unsqueeze(0), X3_spec_f_avg[i].unsqueeze(0))[0]
            else:
                x3 = share_i
            X3_fused_list.append(x3)

        # 堆叠为 batch tensor
        X1_fused_ft = torch.stack(X1_fused_list, dim=0)
        X2_fused_ft = torch.stack(X2_fused_list, dim=0)
        X3_fused_ft = torch.stack(X3_fused_list, dim=0)

        return X1_fused_ft, X2_fused_ft, X3_fused_ft


    def forward(self, M1, M2, M3, mode_Type='[1,1,1]'):
        B, C, H, W = M1.shape
        if mode_Type.ndim == 1:
            mode_Type = mode_Type.unsqueeze(0)
        modalities = torch.cat([M1, M2, M3], dim=1)
        mode_mask = mode_Type.unsqueeze(-1).unsqueeze(-1).cuda()
        masked_modalities = modalities * mode_mask
        M1, M2, M3 = masked_modalities[:, :C, :, :], masked_modalities[:, C:2*C, :, :], masked_modalities[:, 2*C:, :, :]
        
        if M1.size()[1] == 1:
            M1 = M1.repeat(1,3,1,1)#bSSFP
            M2 = M2.repeat(1,3,1,1)#LGE
            M3 = M3.repeat(1,3,1,1)#T2w
        
        #shared_encoder
        X1_share_f_avg, _, X1_share_features = self.transformer(M1) #(24,256,8,8),(24,256,1,1),[(24,1024,8,8),...,(24,64,64,64)]
        X2_share_f_avg, _, X2_share_features = self.transformer(M2)
        X3_share_f_avg , _, X3_share_features = self.transformer(M3)

        share_f_avg = self.compute_shared_features(mode_Type, X1_share_f_avg, X2_share_f_avg, X3_share_f_avg)
 
        #specific_encoder
        X1_spec_f_avg, X1_spec_f, X1_spec_features = self.transformer1(M1) #x在不同空洞卷积后的平均输出以及最后输出x的avgpooling、以及各层的特征
        X2_spec_f_avg, X2_spec_f, X2_spec_features = self.transformer2(M2)
        X3_spec_f_avg, X3_spec_f, X3_spec_features = self.transformer3(M3)

        share_f = torch.stack([X1_share_f_avg, X2_share_f_avg, X3_share_f_avg], dim=0)  # (3, 24, 256, 8, 8)
        spec_f = torch.stack([X1_spec_f, X2_spec_f, X3_spec_f], dim=0) # (3, 24, 256, 1, 1)，这个只用做求解域分类
        
        # visualization of shared-specific fts here
        if visualization:
            # collecting enough data points
            # tsne_2D = TSNE(n_components=2, random_state=0)
            tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
            vis_ft = torch.cat([X1_share_f_avg, X2_share_f_avg, X3_share_f_avg,
                                X1_spec_f_avg, X2_spec_f_avg, X3_spec_f_avg])
            vis_ft = vis_ft.view(vis_ft.shape[0], -1)
            tsne_points.append(vis_ft.cpu().detach().numpy())
            tsne_colors.append([0, 1, 2, 3, 4, 5])
            # tsne_labels.append([0, 1, 2, 3, 4, 5, 6, 7])
            tsne_labels.append(['x', 'x', 'x', 'o', 'o', 'o'])
            if len(tsne_points) == 500:  # actual visualization
                vis_tsne_points = np.concatenate(tsne_points)
                vis_tsne_labels = np.concatenate(tsne_labels)
                vis_tsne_colors = np.concatenate(tsne_colors)
                vis_tsne_2D = tsne_2D.fit_transform(vis_tsne_points)
                tsne_fig = plot_embedding_2D(vis_tsne_2D, vis_tsne_labels, vis_tsne_colors, 't-SNE of Features')
                # plt.show()
                plt.savefig('tsne.png')
                quit()

        # fused features
        X1_fused_ft, X2_fused_ft, X3_fused_ft = self.fuse_modalities_per_sample(
                                            mode_Type,
                                            share_f_avg,
                                            X1_share_f_avg, X1_spec_f_avg,
                                            X2_share_f_avg, X2_spec_f_avg,
                                            X3_share_f_avg, X3_spec_f_avg
                                        )

         # attention注意力机制
        if self.cross_att:
            original_size = X1_fused_ft.size()
            X1_fused_input = X1_fused_ft.view(original_size[0], original_size[1],
                                              original_size[2] * original_size[3])
            X2_fused_input = X2_fused_ft.view(original_size[0], original_size[1],
                                                original_size[2] * original_size[3])
            X3_fused_input = X3_fused_ft.view(original_size[0], original_size[1],
                                                original_size[2] * original_size[3])
            x1_input = X1_fused_input.permute(0, 2, 1)
            x2_input = X2_fused_input.permute(0, 2, 1)
            x3_input = X3_fused_input.permute(0, 2, 1)
            x_others1 = (x2_input + x3_input) / 2
            x_others2 = (x1_input + x3_input) / 2
            x_others3 = (x1_input + x2_input) / 2

            # cross attention
            X1_fused_ft, _ = self.multihead_attn1(x1_input, x_others1, x_others1)
            X2_fused_ft, _ = self.multihead_attn2(x2_input, x_others2, x_others2)
            X3_fused_ft, _ = self.multihead_attn3(x3_input, x_others3, x_others3)
            X1_fused_ft = X1_fused_ft.permute(0, 1, 2).view(original_size)
            X2_fused_ft = X2_fused_ft.permute(0, 1, 2).view(original_size)
            X3_fused_ft = X3_fused_ft.permute(0, 1, 2).view(original_size)

        fused_ft = torch.stack([X1_fused_ft, X2_fused_ft, X3_fused_ft], dim=1)  # (24, 3, 256, 8, 8)

       
        if self.self_att:
            cat_attend = fused_ft.view(B, fused_ft.shape[1] * fused_ft.shape[2], fused_ft.shape[3], fused_ft.shape[4])
            original_size = cat_attend.size()
            flat_input = cat_attend.view(original_size[0], original_size[1],
                                         original_size[2] * original_size[3])
            perm_input = flat_input.permute(0, 2, 1)

            att_input, att_weights = self.multihead_attn(perm_input, perm_input, perm_input)
            flat_output = att_input.permute(0, 1, 2)
            out_ft = flat_output.view(original_size)
        else:
            out_ft = fused_ft.view(B, fused_ft.shape[1] * fused_ft.shape[2], fused_ft.shape[3], fused_ft.shape[4])

        # decoder部分
        features_fusion_share = []
        features_fusion_spec = []
        features_fusion = []
        
        for i in range(1, len(X1_share_features)):
            features_fusion_share.append(self.feature_fusion[i-1](X1_share_features[i], X2_share_features[i], X3_share_features[i]))
        
        for i in range(1, len(X1_spec_features)):
            features_fusion_spec.append(self.feature_fusion2[i-1](X1_spec_features[i], X2_spec_features[i], X3_spec_features[i]))

        for i in range(0, len(features_fusion_share)):
            features_fusion.append(self.feature_fusion3[i](features_fusion_share[i], features_fusion_spec[i]))

        output = self.decoder(out_ft, features_fusion)
        output = self.segmentation_head(output)
        _spec_f = spec_f.squeeze()
        spec_logits = self.dom_classifier(_spec_f)
        spec_logits = spec_logits.transpose(0, 1) 
        share_f = share_f.transpose(0, 1)
        if self.training:
            return output, share_f, spec_logits, mode_Type
        else:
            return output
    

CONFIGS = {
    'R50': configs.r50_config(),
    'testing': configs.get_testing(),
}
