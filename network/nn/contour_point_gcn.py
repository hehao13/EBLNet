import torch
import torch.nn as nn


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = point_indices % W
    point_coords[:, :, 1] = point_indices // W
    return point_indices, point_coords


def point_sample(input, point_indices, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_indices (Tensor): A tensor of shape (N, P) or (N, Hgrid, Wgrid, 2) that contains sampled indices.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    N, C, H, W = input.shape
    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
    flatten_input = input.flatten(start_dim=2)
    sampled_feats = flatten_input.gather(dim=2, index=point_indices).view_as(point_indices)   # n c p
    return sampled_feats


class GCN(nn.Module):
    """
        Implementation of simple GCN operation in Glore Paper cvpr2019
    """
    def __init__(self, node_num, node_fea):
        super(GCN, self).__init__()
        self.node_num = node_num
        self.node_fea = node_fea
        self.conv_adj = nn.Conv1d(self.node_num, self.node_num, kernel_size=1, bias=False)
        self.bn_adj = nn.BatchNorm1d(self.node_num)

        self.conv_wg = nn.Conv1d(self.node_fea, self.node_fea, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(self.node_fea)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
            x: shape: (b, n, d)
        """
        z = self.conv_adj(x)
        z = self.bn_adj(z)
        z = self.relu(z)

        # Laplacian smoothing
        z += x
        z = z.transpose(1, 2).contiguous()  # (b, d, n)
        z = self.conv_wg(z)
        z = self.bn_wg(z)
        z = self.relu(z)
        z = z.transpose(1, 2).contiguous()  # (b, n, d)

        return z


class ContourPointGCN(nn.Module):
    def __init__(self, inplance, num_points, thresholds=0.8):
        super(ContourPointGCN, self).__init__()
        self.num_points = num_points
        self.thresholds = thresholds
        self.gcn = GCN(num_points, inplance)

    def forward(self, x, edge):
        """
            x is body feature (upsampled)
            edge is boundary feature map
            both features are the same size
        """
        B, C, H, W = x.size()
        edge[edge < self.thresholds] = 0
        edge_index, point_coords = get_uncertain_point_coords_on_grid(edge, self.num_points)

        gcn_features = point_sample(
            x, edge_index).permute(0, 2, 1)  # b, c, n - > b, n, c (n: points, c: features)

        gcn_features_reasoned = self.gcn(gcn_features)  # b, n, c

        gcn_features_reasoned = gcn_features_reasoned.permute(0, 2, 1)  # b, c, n

        edge_index = edge_index.unsqueeze(1).expand(-1, C, -1)

        final_features = x.reshape(B, C, H * W).scatter(2, edge_index, gcn_features_reasoned).view(B, C, H, W)

        return final_features
