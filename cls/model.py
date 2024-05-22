import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx


class EFR(nn.Module):
    def __init__(self, c_out):
        super(EFR, self).__init__()

        # self.conv = nn.Conv2d(c_out + 1, c_out, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(1, c_out, kernel_size=1, bias=False)

    def forward(self, x, k=20, idx=None, dim6=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if dim6 == False:
                idx = knn(x, k=k)  # (batch_size, num_points, k)
            else:
                idx = knn(x[:, 0:3], k=k)
            device = x.device

            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

            idx = idx + idx_base

            idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature_mean = feature.mean(dim=-1, keepdim=True)

        # feature = torch.cat((feature - x, feature_mean, x), dim=3).permute(0, 3, 1, 2).contiguous()
        feature = feature_mean.permute(0, 3, 1, 2).contiguous()
        # feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        feature = self.conv(feature)

        return feature, idx


class GCT(nn.Module):
    def __init__(self, in_channels, out_channels, nhiddens, feat_channels):
        super(GCT, self).__init__()
        self.in_channels = in_channels
        self.nhiddens = nhiddens
        self.out_channels = out_channels
        self.feat_channels = feat_channels

        self.conv0 = nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels*in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(nhiddens)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.convq = nn.Conv1d(feat_channels // 2, feat_channels, kernel_size=1, bias=False)
        self.convpos1 = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.convpos2 = nn.Conv2d(in_channels, feat_channels, kernel_size=1, bias=False)
        self.convattn = nn.Conv2d(feat_channels, nhiddens * in_channels, kernel_size=1, bias=False)

        self.bnq = nn.BatchNorm1d(feat_channels)
        # self.bnpos = nn.BatchNorm2d(in_channels)
        self.bnpos1 = nn.BatchNorm1d(in_channels)
        self.bnattn = nn.BatchNorm2d(nhiddens * in_channels)

        self.residual_layer = nn.Sequential(nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(out_channels),
                                            )

        self.linear = nn.Sequential(nn.Conv2d(nhiddens, out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_channels))

    def forward(self, x, y, points, feat):
        # x: (bs, in_channels, num_points, k), y: (bs, feat_channels, num_points, k)
        batch_size, n_dims, num_points, k = x.size()

        knn_points = x
        knn_feat = y

        x_q = self.convq(feat) # (b, c:12, n)
        x_q = self.leaky_relu(self.bnq(x_q))
        x_k = knn_feat # (b, c:12, n, k)
        x_v = knn_points

        pos = self.convpos1(points)
        pos = self.leaky_relu(self.bnpos1(pos))
        pos = pos[:, :, :, None] - knn_points
        pos = self.convpos2(pos)

        attn = x_q[:, :, :, None] - x_k + pos
        attn = F.softmax(attn / np.sqrt(x_k.size(1)), dim=-1)  # (b, c:12, n, k)
        attn = self.convattn(attn)
        attn = attn.permute(0, 2, 3, 1).view(batch_size, num_points, k, self.out_channels, self.in_channels)  # (b, n, k, nhiddens, in)

        x_v = x_v.permute(0, 2, 3, 1).unsqueeze(4) # (bs, num_points, k, in_channels, 1)
        x = torch.matmul(attn, x_v).squeeze(4) # (bs, num_points, k, out_channels)
        x = x.permute(0, 3, 1, 2).contiguous() # (bs, out_channels, num_points, k)

        y = self.leaky_relu(self.bn1(x))
        y = self.linear(y)
        residual = self.residual_layer(knn_feat)
        y += residual
        y = self.leaky_relu(y)

        return y

class Net(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Net, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.GCT1 = GCT(6, 64, 64, 6)
        self.GCT2 = GCT(6, 64, 64, 64*2)

        self.EFR1 = EFR(6)
        self.EFR2 = EFR(128)

    def forward(self, x):
        batch_size = x.size(0)
        feat = x
        points = x

        x, idx = self.EFR1(x, k=self.k)
        p, _ = self.EFR1(points, k=self.k, idx=idx)
        x = self.GCT1(p, x, points, feat)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x, idx = self.EFR2(x1, k=self.k)
        p, _ = self.EFR1(points, k=self.k, idx=idx)
        x = self.GCT2(p, x, points, x1)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, _ = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, _ = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x