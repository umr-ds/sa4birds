import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.block import MultiHeadSABlock



class DSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.in_chans = self.cfg.frontend.in_chans
        self.px_per_5s = 501 # can change with different fft params
        self.downsample_factor = 64 # can change with backbone change

        timm_params = {"model_name": cfg.network.model_name,
                       "pretrained": True,
                       "in_chans": self.in_chans,
                       "drop_path_rate": self.cfg.network.droppath_rate}

        self.backbone = timm.create_model(**timm_params)
        backbone_out = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0, '')
        self.attention = MultiHeadSABlock(backbone_out,
                                           self.cfg.train.num_classes,
                                            heads=1,
                                            activation='sigmoid',
                                            temperature=self.cfg.network.temperature)


        self.l_proj = nn.Conv2d(backbone_out, backbone_out, kernel_size=(1,1), bias=True)
        self.g_proj = nn.Conv2d(backbone_out, backbone_out, kernel_size=(1,1), bias=True)


        self.dropout = cfg.network.dropout_rate

        self.norm = nn.LayerNorm(normalized_shape=backbone_out)

        self.fuse_weight = nn.Parameter(torch.tensor(0.9))

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2))


    def forward(self, x, center_5s=False):
        """
           Forward pass of the model.

           Args:
               x (Tensor): Input tensor of shape (batch, channels, freq, time)
               center_5s (bool): predict center 5s of input if true

           Returns:
               - weak (Tensor): Output tensor of shape (batch, num_classes)
        """

        def normalize_and_project(feat, proj_layer):
            feat = self.norm(feat.transpose(1, -1)).transpose(1, -1)
            feat = F.relu(proj_layer(feat))
            return feat

        t = x.shape[-1]
        x = self.backbone(x)  # -> (b, feats, f, t)

        # Branch 1: Local features
        l_x = F.dropout(x, p=self.dropout, training=self.training)
        residual = l_x
        l_x = normalize_and_project(l_x, self.l_proj)
        l_x = residual + l_x
        l_x = F.dropout(l_x, p=self.dropout, training=self.training)

        # Branch 2: Global features
        g_x = self.avg_pool(x)
        g_x = F.dropout(g_x, p=self.dropout, training=self.training)
        g_x = normalize_and_project(g_x, self.g_proj)
        g_x = F.dropout(g_x, p=self.dropout, training=self.training)

        if center_5s:
            crop = int(round((t - self.px_per_5s) / self.downsample_factor))
        else:
            crop =  0

        l_weak, l_att, _ = self.attention(l_x, crop)
        g_weak, g_att, _ = self.attention(g_x, crop // 2)

        fuse_weight = torch.clamp(self.fuse_weight, min=0, max=1)
        weak = l_weak * fuse_weight + g_weak * (1 - fuse_weight)
        return weak
