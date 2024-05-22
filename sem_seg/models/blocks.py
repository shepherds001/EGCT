import time
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
import torch.nn.functional as F
import numpy as np

from utils.ply import write_ply


def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig ** 2 + eps))


def closest_pool(x, inds):
    """
    Pools features from the closest neighbors. WARNING: this function assumes the neighbors are ordered.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] Only the first column is used for pooling
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get features for each pooling location [n2, d]
    return gather(x, inds[:, 0])


def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features


def global_average(x, batch_lengths):
    """
    Block performing a global average over batch pooling
    :param x: [N, D] input features
    :param batch_lengths: [B] list of batch lengths
    :return: [B, D] averaged features
    """

    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        # Average features for each batch cloud
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))

        # Increment for next cloud
        i0 += length

    # Average features in each batch
    return torch.stack(averaged_features)


def get_graph_feature(x, idx, feat_mode='none'):
    # x: (n_supports, num_dims), idx: (n_queries, k)
    n_supports, num_dims = x.size()
    n_queries, k = idx.size()

    feature = x[idx, :]  # (n_queries, k, num_dims)
    if feat_mode == 'asym':
        x1 = x[idx[:, 0], :]  # (n_queries, num_dims)
        x1 = x1.view(n_queries, 1, num_dims).repeat(1, k, 1)

        feature = torch.cat((feature - x1, x1), dim=2).contiguous()  # (n_queries, k, num_dims*2)

    return feature


class EFR(nn.Module):
    def __init__(self, c_out):
        super(EFR, self).__init__()

        self.conv = nn.Conv1d(c_out + 1, c_out, kernel_size=1, bias=False)

    def forward(self, x, idx, feat_mode='none'):
        # x: (n_supports, num_dims), idx: (n_queries, k)
        n_supports, num_dims = x.size()
        n_queries, k = idx.size()

        feature = x[idx, :]
        if feat_mode == 'asym':
            x1 = x[idx[:, 0], :]  # (n_queries, num_dims)
            x1 = x1.view(n_queries, 1, num_dims).repeat(1, k, 1)

            feature_mean = feature.mean(dim=-1, keepdim=True)

            feature = torch.cat((feature - x1, feature_mean, x1), dim=2).permute(0, 2, 1).contiguous()

            feature = self.conv(feature)

            feature = feature.permute(0, 2, 1)

        return feature


class GCT(nn.Module):
    def __init__(self, in_channels, feat_channels, out_channels, bn_momentum, feat_mode, strided=False):
        super(GCT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.feat_mode = feat_mode
        self.strided = strided

        if self.feat_mode == 'asym':
            # self.conv_asym_p = nn.Conv1d(self.in_channels, self.in_channels * 2, kernel_size=1, bias=False)
            # self.conv_asym_f = nn.Conv1d(self.feat_channels, self.feat_channels * 2, kernel_size=1, bias=False)
            self.in_channels *= 2
            self.feat_channels *= 2

        if 'xyz' in self.feat_mode:
            self.conv_xyz = nn.Conv1d(self.in_channels, self.in_channels * 2, kernel_size=1, bias=False)
            if 'xyz2' in self.feat_mode and not self.strided:
                self.in_channels *= 2

        if 'feat2' in self.feat_mode and not self.strided:
            self.conv_feat2 = nn.Conv1d(self.feat_channels, self.feat_channels * 2, kernel_size=1, bias=False)
            self.feat_channels *= 2

        if 'joint' in self.feat_mode:
            self.conv_joint = nn.Conv1d(self.feat_channels, self.feat_channels + self.in_channels, kernel_size=1,
                                        bias=False)
            self.feat_channels += self.in_channels

        self.conv0 = nn.Conv2d(self.feat_channels, self.out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels * self.in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(self.out_channels, momentum=bn_momentum)
        self.bn1 = nn.BatchNorm2d(self.out_channels, momentum=bn_momentum)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        self.convq = nn.Conv1d(self.feat_channels // 2, self.feat_channels, kernel_size=1, bias=False)
        self.convpos1 = nn.Conv1d(self.in_channels // 2, self.in_channels, kernel_size=1, bias=False)
        self.convpos2 = nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=1, bias=False)
        self.convattn = nn.Conv2d(self.feat_channels, self.out_channels * in_channels, kernel_size=1, bias=False)
        self.convv1 = nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=1, bias=False)

        self.bnq = nn.BatchNorm1d(self.feat_channels, momentum=bn_momentum)
        # self.bnpos = nn.BatchNorm2d(in_channels)
        self.bnpos1 = nn.BatchNorm1d(self.in_channels, momentum=bn_momentum)
        self.bnattn = nn.BatchNorm2d(self.out_channels * self.in_channels, momentum=bn_momentum)

        self.residual_layer = nn.Sequential(nn.Conv2d(self.feat_channels, self.out_channels, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(self.out_channels),)

        self.EFR_P = EFR(self.in_channels * 2)
        self.EFR_F = EFR(self.feat_channels * 2)

    def forward(self, q_points, s_points, neighb_inds, feat):
        # s_points: (n_supports 88751, 3), q_points: (n_queries 25039, 3), feat: (n_supports 88751, feat_channels 32), neighb_inds: (n_queries 25039, k 29)
        n_supports, _ = s_points.size()
        n_queries, k = neighb_inds.size()

        # Add a fake point in the last row for shadow neighbors

        s_points_ori = s_points  # (n_supports 88750, 3)
        feat_ori = feat  # (n_supports 88750, feat_channels 32)

        s_points = torch.cat((s_points, torch.zeros_like(s_points[:1, :])), 0)  # (n_supports+1 13920, 3)
        # Add a zero feature for shadow neighbors
        feat = torch.cat((feat, torch.zeros_like(feat[:1, :])), 0)  # (n_supports+1 13920, feat_channels 5)

        x = self.EFR_P(s_points, idx=neighb_inds,
                       feat_mode=self.feat_mode)
        y = self.EFR_F(feat, idx=neighb_inds,
                       feat_mode=self.feat_mode)
        if n_supports != n_queries:
            s_points_ori = s_points_ori[:n_queries]
            feat_ori = feat_ori[:n_queries]

        s_points_ori = s_points_ori.unsqueeze(2)
        feat_ori = feat_ori.unsqueeze(2)

        if 'xyz' in self.feat_mode:
            # asymmetric feature
            if 'xyz2' in self.feat_mode and not self.strided:
                q_points = q_points.unsqueeze(1).repeat(1, k, 1)
                x = torch.cat((x - q_points, q_points), dim=2)  # (n_queries, k, in_channels*2)

                s_points_ori = self.conv_xyz(s_points_ori)

            else:
                x = x - q_points.unsqueeze(1)

            # zeros out-ranged points
            out_ranged = (neighb_inds.view(-1) == int(s_points.shape[0] - 1))
            last_dim = x.shape[-1]
            x = x.view(n_queries * k, last_dim)

            x[out_ranged, :] = torch.zeros_like(x[:1, :])
            x = x.view(n_queries, k, last_dim).contiguous()
        else:
            x = x - q_points.unsqueeze(1)

        if 'feat2' in self.feat_mode and not self.strided:
            # asymmetric feature
            y_center = feat[neighb_inds[:, 0], :]  # (n_queries, feat_channels)
            y_center = y_center.unsqueeze(1).repeat(1, k, 1)  # (n_queries, k, feat_channels)
            y = torch.cat((y - y_center, y_center), dim=2)  # (n_queries, k, feat_channels*2)

            feat_ori = self.conv_feat2(feat_ori)

        # feat+points
        if 'joint' in self.feat_mode:
            y = torch.cat((x, y), dim=2)

            feat_ori = self.conv_joint(feat_ori)

        s_points_ori = s_points_ori.permute(2, 1, 0)  # (1, in * 2, n_queres)
        feat_ori = feat_ori.permute(2, 1, 0)

        x_q = feat_ori  # (b, c, n)
        x_q = self.leaky_relu(self.bnq(x_q))
        y = y.permute(2, 0, 1).unsqueeze(0)  # (n, k, c) -> (c, n, k) -> (b, c, n, k)
        x_k = y  # (b, c_feat, n, k)
        x = x.permute(2, 0, 1).unsqueeze(0)
        x_v = x  # (b, c_in, n, k)

        pos = s_points_ori
        pos = self.leaky_relu(self.bnpos1(pos))
        pos = pos[:, :, :, None] - x # pos ()
        pos = self.convpos2(pos)

        attn = x_q[:, :, :, None] - x_k + pos

        attn = F.softmax(attn / np.sqrt(x_k.size(1)), dim=-1)  # (b, c:12, n, k)
        attn = self.convattn(attn)
        attn = attn.permute(0, 2, 3, 1).view(1, n_queries, k, self.out_channels,
                                             self.in_channels)  # (b, n, k, out, in)

        x_v = x_v.permute(0, 2, 3, 1).unsqueeze(4)  # (b, c_in, n, k) -> (bs, num_points, k, in_channels, 1)
        y_e = torch.matmul(attn, x_v).squeeze(4)  # (bs, num_points, k, out)
        y_e = y_e.permute(0, 3, 1, 2).contiguous()  # (bs, out, num_points, k)

        y_e = self.leaky_relu(self.bn1(y_e))
        residual = self.residual_layer(y)
        y_e += residual
        y_e = self.leaky_relu(y_e)

        y_e = y_e.max(dim=-1, keepdim=False)[0]  # (bs, out_channels, n_queries)
        y_e = y_e.permute(0, 2, 1).squeeze(0).contiguous()  # (n_queries, out_channels)

        return y_e


class GctResnetBottleneckBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(GctResnetBottleneckBlock, self).__init__()

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.hidden_dim = out_dim // 4  # if out_dim // 4 > 64 else 64
        # First downscaling mlp
        if in_dim != self.hidden_dim:
            self.unary1 = UnaryBlock(in_dim, self.hidden_dim, self.use_bn, self.bn_momentum)
        else:
            self.unary1 = nn.Identity()

        # Conv block
        self.GCTrans = GCT(
            in_channels=3,
            feat_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            bn_momentum=self.bn_momentum,
            feat_mode=config.Gct_feature,
            strided='strided' in self.block_name)

        self.batch_norm_conv = BatchNormBlock(self.hidden_dim, self.use_bn, self.bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(self.hidden_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)

        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, features, batch):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        # First downscaling mlp
        x = self.unary1(features)

        # Convolution
        x = self.GCTrans(q_pts, s_pts, neighb_inds, x)
        # x = self.leaky_relu(self.batch_norm_conv(x))

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)

        return self.leaky_relu(x + shortcut)


class GctSimpleBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(GctSimpleBlock, self).__init__()

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.actual_dim = out_dim // 2

        # Conv block
        self.GCTrans = GCT(
            in_channels=3,
            feat_channels=in_dim,
            out_channels=self.actual_dim,
            bn_momentum=self.bn_momentum,
            feat_mode=config.first_Gct_feature,
            strided='strided' in self.block_name)

        self.batch_norm_conv = BatchNormBlock(self.actual_dim, self.use_bn, self.bn_momentum)

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, x, batch):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        # Convolution
        x = self.GCTrans(q_pts, s_pts, neighb_inds, x)
        # x = self.leaky_relu(self.batch_norm_conv(x))

        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, feat_channels, out_channels, bn_momentum, feat_mode):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.feat_mode = feat_mode

        if self.feat_mode == 'asym':
            self.in_channels *= 2

        if 'xyz' in self.feat_mode:
            if 'xyz2' in self.feat_mode:
                self.in_channels *= 2

        if 'joint' in self.feat_mode:
            self.feat_channels += self.in_channels

        self.conv = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(negative_slope=0.1))

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))

        return

    def forward(self, q_points, s_points, neighb_inds, feat):
        # s_points: (n_supports, 3), q_points: (n_queries, 3), feat: (n_supports, in_channels), neighb_inds: (n_queries, k)
        n_supports, _ = s_points.size()
        n_queries, k = neighb_inds.size()

        # Add a fake point in the last row for shadow neighbors
        s_points = torch.cat((s_points, torch.zeros_like(s_points[:1, :])), 0)  # (n_supports+1, 3)
        # Add a zero feature for shadow neighbors
        feat = torch.cat((feat, torch.zeros_like(feat[:1, :])), 0)  # (n_supports+1, feat_channels)

        x = get_graph_feature(s_points, idx=neighb_inds, feat_mode=self.feat_mode)  # (n_queries, k, in_channels)
        y = get_graph_feature(feat, idx=neighb_inds, feat_mode=self.feat_mode)  # (n_queries, k, feat_channels)

        if 'xyz' in self.feat_mode and 'joint' in self.feat_mode:
            # asymmetric feature
            if 'xyz2' in self.feat_mode:
                q_points = q_points.unsqueeze(1).repeat(1, k, 1)
                x = torch.cat((x - q_points, q_points), dim=2)  # (n_queries, k, in_channels*2)
            else:
                x = x - q_points.unsqueeze(1)

            # zeros out-ranged points
            out_ranged = (neighb_inds.view(-1) == int(s_points.shape[0] - 1))
            last_dim = x.shape[-1]
            x = x.view(n_queries * k, last_dim)
            x[out_ranged, :] = torch.zeros_like(x[:1, :])
            x = x.view(n_queries, k, last_dim).contiguous()
        else:
            x = x - q_points.unsqueeze(1)

        # feat+points
        if 'joint' in self.feat_mode:
            y = torch.cat((x, y), dim=2)

        # conv
        y = y.permute(2, 0, 1).unsqueeze(0)  # (bs, feat_channels, n_queries, k)
        y = self.conv(y)  # (bs, out_channels, n_queries, k)
        y = y.max(dim=-1, keepdim=False)[0]  # (bs, out_channels, n_queries)
        y = y.permute(0, 2, 1).squeeze(0).contiguous()  # (n_queries, out_channels)

        return y


class GraphResnetBottleneckBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(GraphResnetBottleneckBlock, self).__init__()

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.hidden_dim = out_dim // 4  # if out_dim // 4 > 64 else 64
        # First downscaling mlp
        if in_dim != self.hidden_dim:
            self.unary1 = UnaryBlock(in_dim, self.hidden_dim, self.use_bn, self.bn_momentum)
        else:
            self.unary1 = nn.Identity()

        # Conv block
        self.Conv = GraphConv(
            in_channels=3,
            feat_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            bn_momentum=self.bn_momentum,
            feat_mode=config.Gct_feature)

        self.batch_norm_conv = BatchNormBlock(self.hidden_dim, self.use_bn, self.bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(self.hidden_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)

        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, features, batch):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        # First downscaling mlp
        x = self.unary1(features)

        # Convolution
        x = self.Conv(q_pts, s_pts, neighb_inds, x)
        # x = self.leaky_relu(self.batch_norm_conv(x))

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)

        return self.leaky_relu(x + shortcut)


class GraphSimpleBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(GraphSimpleBlock, self).__init__()

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.actual_dim = out_dim // 2

        # Conv block
        self.Conv = GraphConv(
            in_channels=3,
            feat_channels=in_dim,
            out_channels=self.actual_dim,
            bn_momentum=self.bn_momentum,
            feat_mode=config.first_Gct_feature)

        # Other opperations
        self.batch_norm = BatchNormBlock(self.actual_dim, self.use_bn, self.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, x, batch):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        # convolution
        x = self.Conv(q_pts, s_pts, neighb_inds, x)

        return x


def block_decider(block_name,
                  radius,
                  in_dim,
                  out_dim,
                  layer_ind,
                  config):
    if block_name == 'unary':
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)

    elif block_name in ['Gct_resnetb',
                        'Gct_resnetb_strided',
                        'Gct_resnetb_deformable',
                        'Gct_resnetb_deformable_strided']:
        return GctResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name in ['Gct_simple',
                        'Gct_simple_strided']:
        return GctSimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name in ['graph_resnetb',
                        'graph_resnetb_strided',
                        'graph_resnetb_deformable',
                        'graph_resnetb_deformable_strided']:
        return GraphResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name in ['graph_simple',
                        'graph_simple_strided']:
        return GraphSimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


class BatchNormBlock(nn.Module):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
            # self.batch_norm = nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:

            x = x.unsqueeze(2)
            x = x.transpose(0, 2)
            x = self.batch_norm(x)
            x = x.transpose(0, 2)
            return x.squeeze()
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(self.in_dim,
                                                                                         self.bn_momentum,
                                                                                         str(not self.use_bn))


class UnaryBlock(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim,
                                                                                        self.out_dim,
                                                                                        str(self.use_bn),
                                                                                        str(not self.no_relu))


class GlobalAverageBlock(nn.Module):

    def __init__(self):
        """
        Initialize a global average block with its ReLU and BatchNorm.
        """
        super(GlobalAverageBlock, self).__init__()
        return

    def forward(self, x, batch):
        return global_average(x, batch.lengths[-1])


class NearestUpsampleBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return closest_pool(x, batch.upsamples[self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(self.layer_ind,
                                                                  self.layer_ind - 1)


class MaxPoolBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a max pooling block with its ReLU and BatchNorm.
        """
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return max_pool(x, batch.pools[self.layer_ind + 1])


class S3DISConfig():
    conv_radius = 2.5

    KP_extent = 1.2

    use_batch_norm = True
    batch_norm_momentum = 0.02

    Gct_feature = 'none'
    first_Gct_feature = 'xyz_joint'


class TempBatch():
    points = None


if __name__ == '__main__':
    block_name = 'graph_simple'
    in_dim = 64
    out_dim = in_dim * 2 if 'strided' in block_name else in_dim
    radius = 0.03
    layer_ind = 0
    config = S3DISConfig()

    layer = block_decider(block_name, radius, in_dim, out_dim, layer_ind, config)
    print(layer)

    npoints1 = 10000
    npoints2 = 5000
    n1 = 30
    n2 = 40

    batch = TempBatch()
    batch.points = [torch.rand(npoints1, 3)]
    batch.points.append(torch.rand(npoints2, 3))
    batch.neighbors = [torch.randint(0, npoints1, (npoints1, n1))]
    batch.pools = [torch.randint(0, npoints1, (npoints2, n2))]
    feature = torch.rand(npoints1, in_dim)

    out = layer(feature, batch)
    print(out.size())