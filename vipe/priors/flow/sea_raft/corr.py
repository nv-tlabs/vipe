import torch
import torch.nn as nn
import torch.nn.functional as F
from .raft_utils import coords_grid, bilinear_sampler

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

def coords_feature(fmap, b, x, y):
    H, W = fmap.shape[2:]
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    b = b.long()
    x = torch.clamp(x, 0, W-1).long()
    y = torch.clamp(y, 0, H-1).long()
    res = fmap[b, :, y, x] * mask.float().unsqueeze(1)
    return res

def bilinear_sampling(fmap, coords):
    """coords: (bhw)"""
    device = fmap.device
    offset = (coords - coords.floor()).to(device)
    dx, dy = offset[:, 1, None], offset[:, 2, None]
    b = coords[:, 0].long()
    x0, y0 = coords[:, 1].floor(), coords[:, 2].floor()
    x1, y1 = x0 + 1, y0 + 1
    f00 = (1 - dy) * (1 - dx) * coords_feature(fmap, b, x0, y0)
    f01 = (1 - dy) * dx * coords_feature(fmap, b, x0, y1)
    f10 = dy * (1 - dx) * coords_feature(fmap, b, x1, y0)
    f11 = dy * dx * coords_feature(fmap, b, x1, y1)
    return f00 + f01 + f10 + f11

def coords_corr(corr, idx, b, x, y):
    # corr: [N, H, W, H, W]
    H, W = corr.shape[-2:]
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    b = b.long()
    idx = idx.long()
    x = torch.clamp(x, 0, W-1).long()
    y = torch.clamp(y, 0, H-1).long()
    res = corr[b, idx[:, 2], idx[:, 1], y, x] * mask.float()
    print(mask.requires_grad, x.requires_grad, y.requires_grad, res.requires_grad)
    return res

def bilinear_sampling_corr(corr, idx1, idx2):
    """idx1: [M, (bhw)], idx2: [M, n_points, (bhw)]"""
    M, n_points = idx2.shape[:2]
    # reshape idx: [M * n_points, (bhw)]
    idx1 = idx1.unsqueeze(1).repeat(1, n_points, 1).view(-1, 3)
    idx2 = idx2.view(-1, 3)
    device = corr.device
    offset = idx2 - idx2.floor()
    dx, dy = offset[:, 1], offset[:, 2]
    b = idx2[:, 0].long()
    x0, y0 = idx2[:, 1].floor(), idx2[:, 2].floor()
    x1, y1 = x0 + 1, y0 + 1
    f00 = (1 - dy) * (1 - dx) * coords_corr(corr, idx1, b, x0, y0)
    # f01 = (1 - dy) * dx * coords_corr(corr, idx1, b, x0, y1)
    # f10 = dy * (1 - dx) * coords_corr(corr, idx1, b, x1, y0)
    # f11 = dy * dx * coords_corr(corr, idx1, b, x1, y1)
    # res = f00 + f01 + f10 + f11\
    res = f00
    return res.view(M, n_points)

class CorrBlock:
    def __init__(self, fmap1, fmap2, args):
        self.num_levels = args.corr_levels
        self.radius = args.corr_radius
        self.args = args
        self.corr_pyramid = []
        # all pairs correlation
        for i in range(self.num_levels):
            corr = CorrBlock.corr(fmap1, fmap2, 1)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch*h1*w1, dim, h2, w2)
            fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode='bilinear', align_corners=False)
            self.corr_pyramid.append(corr)

    def __call__(self, coords, dilation=None):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        if dilation is None:
            dilation = torch.ones(batch, 1, h1, w1, device=coords.device)

        # print(dilation.max(), dilation.mean(), dilation.min())
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            device = coords.device
            dx = torch.linspace(-r, r, 2*r+1, device=device)
            dy = torch.linspace(-r, r, 2*r+1, device=device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            delta_lvl = delta_lvl * dilation.view(batch * h1 * w1, 1, 1, 1)
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous().float()  
        return out

    @staticmethod
    def corr(fmap1, fmap2, num_head):
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]
        fmap1 = fmap1.view(batch, num_head, dim // num_head, h1*w1)
        fmap2 = fmap2.view(batch, num_head, dim // num_head, h2*w2) 
        corr = fmap1.transpose(2, 3) @ fmap2
        corr = corr.reshape(batch, num_head, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)
        return corr  / torch.sqrt(torch.tensor(dim).float())

# class CorrBlock:
#     def __init__(self, context, fmap1, fmap2, args):
#         self.num_levels = args.corr_levels
#         self.radius = args.corr_radius
#         self.args = args
#         self.context = context
#         self.corr_pyramid = []
#         # all pairs correlation
#         for i in range(self.num_levels):
#             corr = CorrBlock.corr(fmap1, fmap2)
#             batch, h1, w1, h2, w2 = corr.shape
#             fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode='bilinear', align_corners=False)
#             self.corr_pyramid.append(corr)

#     def __call__(self, idx1, idx2):
#         # idx: [M, (bhw)]
#         idx1 = idx1.detach()
#         idx2 = idx2.detach()
#         M = idx1.shape[0]
#         r = self.radius
#         n_points = (2 * self.radius + 1) ** 2
#         inp = coords_feature(self.context, idx1[:, 0], idx1[:, 1], idx1[:, 2])
#         device = idx1.device
#         corr_list = []
#         for i in range(self.num_levels):
#             dx = torch.linspace(-r, r, 2*r+1, device=device)
#             dy = torch.linspace(-r, r, 2*r+1, device=device)
#             delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
#             delta_lvl = delta.view(1, n_points, 2)
#             # delta_lvl: [..., (bhw)]
#             delta_lvl = torch.cat([torch.zeros([1, n_points, 1], device=device), delta_lvl], dim=-1)
#             centroid_lvl = idx2
#             centroid_lvl[:, 1:] /= 2**i
#             coords_lvl = centroid_lvl.unsqueeze(1) + delta_lvl
#             corr = bilinear_sampling_corr(self.corr_pyramid[i], idx1, coords_lvl)
#             corr_list.append(corr)

#         corr = torch.cat(corr_list, dim=1)
#         return inp, corr

#     def corr(fmap1, fmap2):
#         batch, dim, h1, w1 = fmap1.shape
#         h2, w2 = fmap2.shape[2:]
#         fmap1 = fmap1.view(batch, dim, h1*w1)
#         fmap2 = fmap2.view(batch, dim, h2*w2) 
#         corr = fmap1.transpose(1, 2) @ fmap2
#         corr = corr.reshape(batch, h1, w1, h2, w2)
#         return corr  / torch.sqrt(torch.tensor(dim).float())

# class FastCorr:
#     def __init__(self, context, fmap1, fmap2, args):
#         self.num_levels = args.corr_levels
#         self.radius = args.corr_radius
#         self.num_head = args.num_head
#         self.fmap1 = fmap1
#         self.context = context
#         self.fmap2_pyramid = [fmap2]
#         for i in range(self.num_levels - 1):
#             fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
#             self.fmap2_pyramid.append(fmap2)

#     def __call__(self, idx1, idx2):
#         # idx: [M, (bhw)]
#         M = idx1.shape[0]
#         r = self.radius
#         n_points = (2 * self.radius + 1) ** 2
#         inp = coords_feature(self.context, idx1[:, 0], idx1[:, 1], idx1[:, 2])
#         f1 = coords_feature(self.fmap1, idx1[:, 0], idx1[:, 1], idx1[:, 2])
#         corr_list = []
#         for i in range(self.num_levels):
#             device = idx1.device
#             dx = torch.linspace(-r, r, 2*r+1, device=device)
#             dy = torch.linspace(-r, r, 2*r+1, device=device)
#             delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
#             delta_lvl = delta.view(1, n_points, 2)
#             # delta_lvl: [..., (bhw)]
#             delta_lvl = torch.cat([torch.zeros([1, n_points, 1], device=device), delta_lvl], dim=-1)
#             centroid_lvl = idx2
#             centroid_lvl[:, 1:] /= 2**i
#             coords_lvl = centroid_lvl.unsqueeze(1) + delta_lvl
#             # f2: [M * n_points, C]
#             f2 = bilinear_sampling(self.fmap2_pyramid[i], coords_lvl)
#             f2 = f2.view(M, n_points, -1)
#             corr = torch.einsum('mnc,mc->mn', f2, f1) / torch.sqrt(torch.tensor(f1.shape[-1]).float())
#             corr_list.append(corr)

#         corr = torch.cat(corr_list, dim=1)
#         return inp, corr
