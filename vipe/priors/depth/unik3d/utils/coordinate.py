import torch


def coords_grid(b, h, w, homogeneous=False, device=None, noisy=False):
    pixel_coords_x = torch.linspace(0.5, w - 0.5, w, device=device)
    pixel_coords_y = torch.linspace(0.5, h - 0.5, h, device=device)
    if noisy:  # \pm 0.5px noise
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5

    stacks = [pixel_coords_x.repeat(h, 1), pixel_coords_y.repeat(w, 1).t()]
    if homogeneous:
        ones = torch.ones_like(stacks[0])  # [H, W]
        stacks.append(ones)
    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]
    if device is not None:
        grid = grid.to(device)

    return grid


def normalize_coords(coords, h, w):
    c = torch.tensor([(w - 1) / 2.0, (h - 1) / 2.0], device=coords.device).view(1, 2, 1, 1)
    return (coords - c) / c
