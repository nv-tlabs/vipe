from functools import wraps
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from scipy import interpolate


@torch.jit.script
def max_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).max(dim=-1).values


def last_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return tensors[-1]


def first_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return tensors[0]


@torch.jit.script
def softmax_stack(tensors: list[torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return F.softmax(torch.stack(tensors, dim=-1) / temperature, dim=-1).sum(dim=-1)


@torch.jit.script
def mean_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).mean(dim=-1)


@torch.jit.script
def sum_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).sum(dim=-1)


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def format_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def get_params(module, lr, wd):
    skip_list = {}
    skip_keywords = {}
    if hasattr(module, "no_weight_decay"):
        skip_list = module.no_weight_decay()
    if hasattr(module, "no_weight_decay_keywords"):
        skip_keywords = module.no_weight_decay_keywords()
    has_decay = []
    no_decay = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (name in skip_list)
            or any((kw in name for kw in skip_keywords))
            or len(param.shape) == 1
            or name.endswith(".gamma")
            or name.endswith(".beta")
            or name.endswith(".bias")
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)

    group1 = {
        "params": has_decay,
        "weight_decay": wd,
        "lr": lr,
        "weight_decay_init": wd,
        "weight_decay_base": wd,
        "lr_base": lr,
    }
    group2 = {
        "params": no_decay,
        "weight_decay": 0.0,
        "lr": lr,
        "weight_decay_init": 0.0,
        "weight_decay_base": 0.0,
        "weight_decay_final": 0.0,
        "lr_base": lr,
    }
    return [group1, group2], [lr, lr]


def get_num_layer_for_swin(var_name, num_max_layer, layers_per_stage):
    if var_name in ("cls_token", "mask_token", "pos_embed", "absolute_pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("layers"):
        if var_name.split(".")[2] == "blocks":
            stage_id = int(var_name.split(".")[1])
            layer_id = int(var_name.split(".")[3]) + sum(layers_per_stage[:stage_id])
            return layer_id + 1
        elif var_name.split(".")[2] == "downsample":
            stage_id = int(var_name.split(".")[1])
            layer_id = sum(layers_per_stage[: stage_id + 1])
            return layer_id
    else:
        return num_max_layer - 1


def get_params_layerdecayswin(module, lr, wd, ld):
    skip_list = {}
    skip_keywords = {}
    if hasattr(module, "no_weight_decay"):
        skip_list = module.no_weight_decay()
    if hasattr(module, "no_weight_decay_keywords"):
        skip_keywords = module.no_weight_decay_keywords()
    layers_per_stage = module.depths
    num_layers = sum(layers_per_stage) + 1
    lrs = []
    params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            print(f"{name} frozen")
            continue  # frozen weights
        layer_id = get_num_layer_for_swin(name, num_layers, layers_per_stage)
        lr_cur = lr * ld ** (num_layers - layer_id - 1)
        # if (name in skip_list) or any((kw in name for kw in skip_keywords)) or len(param.shape) == 1 or name.endswith(".bias"):
        if (name in skip_list) or any((kw in name for kw in skip_keywords)):
            wd_cur = 0.0
        else:
            wd_cur = wd
        params.append({"params": param, "weight_decay": wd_cur, "lr": lr_cur})
        lrs.append(lr_cur)
    return params, lrs


def log(t, eps: float = 1e-5):
    return torch.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, dim=-1)


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def divisible_by(numer, denom):
    return (numer % denom) == 0


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


rearrange_many = _many(rearrange)
repeat_many = _many(repeat)
reduce_many = _many(reduce)


def load_pretrained(state_dict, checkpoint):
    checkpoint_model = checkpoint["model"]
    if any([True if "encoder." in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {
            k.replace("encoder.", ""): v for k, v in checkpoint_model.items() if k.startswith("encoder.")
        }
        print("Detect pre-trained model, remove [encoder.] prefix.")
    else:
        print("Detect non-pre-trained model, pass without doing anything.")
    print(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
    checkpoint = load_checkpoint_swin(state_dict, checkpoint_model)


def load_checkpoint_swin(model, checkpoint_model):
    state_dict = model.state_dict()
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    print(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r**n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind="cubic")
                        all_rel_pos_bias.append(
                            torch.Tensor(f_cubic(dx, dy))
                            .contiguous()
                            .view(-1, 1)
                            .to(relative_position_bias_table_pretrained.device)
                        )

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # # re-map keys due to name change
    rpe_mlp_keys = [k for k in checkpoint_model.keys() if "cpb_mlp" in k]
    for k in rpe_mlp_keys:
        checkpoint_model[k.replace("cpb_mlp", "rpe_mlp")] = checkpoint_model.pop(k)

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    encoder_keys = [k for k in checkpoint_model.keys() if k.startswith("encoder.")]
    for k in encoder_keys:
        checkpoint_model[k.replace("encoder.", "")] = checkpoint_model.pop(k)

    return checkpoint_model


def add_padding_metas(out, image_metas):
    device = out.device
    # left, right, top, bottom
    paddings = [img_meta.get("paddings", [0] * 4) for img_meta in image_metas]
    paddings = torch.stack(paddings).to(device)
    outs = [F.pad(o, padding, value=0.0) for padding, o in zip(paddings, out)]
    return torch.stack(outs)


# left, right, top, bottom
def remove_padding(out, paddings):
    H, W = out.shape[-2:]
    outs = [o[..., padding[2] : H - padding[3], padding[0] : W - padding[1]] for padding, o in zip(paddings, out)]
    return torch.stack(outs)


def remove_padding_metas(out, image_metas):
    B, C, H, W = out.shape
    device = out.device
    # left, right, top, bottom
    paddings = [torch.tensor(img_meta.get("paddings", [0] * 4)) for img_meta in image_metas]
    return remove_padding(out, paddings)


def ssi_helper(tensor1, tensor2):
    stability_mat = 1e-4 * torch.eye(2, device=tensor1.device)
    tensor2_one = torch.stack([tensor2, torch.ones_like(tensor2)], dim=1)
    scale_shift = torch.inverse(tensor2_one.T @ tensor2_one + stability_mat) @ (tensor2_one.T @ tensor1.unsqueeze(1))
    scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    return scale, shift


def calculate_mean_values(names, values):
    # Create a defaultdict to store sum and count for each name
    name_values = {name: {} for name in names}

    # Iterate through the lists and accumulate values for each name
    for name, value in zip(names, values):
        name_values[name]["sum"] = name_values[name].get("sum", 0.0) + value
        name_values[name]["count"] = name_values[name].get("count", 0.0) + 1

    # Calculate mean values and create the output dictionary
    output_dict = {name: name_values[name]["sum"] / name_values[name]["count"] for name in name_values}

    return output_dict


def remove_leading_dim(infos):
    if isinstance(infos, dict):
        return {k: remove_leading_dim(v) for k, v in infos.items()}
    elif isinstance(infos, torch.Tensor):
        return infos.squeeze(0)
    else:
        return infos


def recursive_index(infos, index):
    if isinstance(infos, dict):
        return {k: recursive_index(v, index) for k, v in infos.items()}
    elif isinstance(infos, torch.Tensor):
        return infos[index]
    else:
        return infos


def to_cpu(infos):
    if isinstance(infos, dict):
        return {k: to_cpu(v) for k, v in infos.items()}
    elif isinstance(infos, torch.Tensor):
        return infos.detach()
    else:
        return infos


def masked_mean(
    data: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim: list[int] | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    dim = dim if dim is not None else list(range(data.dim()))
    if mask is None:
        return data.mean(dim=dim, keepdim=keepdim)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(mask_sum, min=1.0)
    return mask_mean.squeeze(dim) if not keepdim else mask_mean


class ProfileMethod:
    def __init__(self, model, func_name, track_statistics=True, verbose=False):
        self.model = model
        self.func_name = func_name
        self.verbose = verbose
        self.track_statistics = track_statistics
        self.timings = []

    def __enter__(self):
        # Start timing
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.end_time = time()

            elapsed_time = self.end_time - self.start_time

            self.timings.append(elapsed_time)
            if self.track_statistics and len(self.timings) > 25:
                # Compute statistics if tracking
                timings_array = np.array(self.timings)
                mean_time = np.mean(timings_array)
                std_time = np.std(timings_array)
                quantiles = np.percentile(timings_array, [0, 25, 50, 75, 100])
                print(f"{self.model.__class__.__name__}.{self.func_name} took {elapsed_time:.4f} seconds")
                print(f"Mean Time: {mean_time:.4f} seconds")
                print(f"Std Time: {std_time:.4f} seconds")
                print(
                    f"Quantiles: Min={quantiles[0]:.4f}, 25%={quantiles[1]:.4f}, Median={quantiles[2]:.4f}, 75%={quantiles[3]:.4f}, Max={quantiles[4]:.4f}"
                )

            else:
                print(f"{self.model.__class__.__name__}.{self.func_name} took {elapsed_time:.4f} seconds")


def profile_method(track_statistics=True, verbose=False):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with ProfileMethod(self, func.__name__, track_statistics, verbose):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


class ProfileFunction:
    def __init__(self, func_name, track_statistics=True, verbose=False):
        self.func_name = func_name
        self.verbose = verbose
        self.track_statistics = track_statistics
        self.timings = []

    def __enter__(self):
        # Start timing
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.end_time = time()

            elapsed_time = self.end_time - self.start_time

            self.timings.append(elapsed_time)
            if self.track_statistics and len(self.timings) > 25:
                # Compute statistics if tracking
                timings_array = np.array(self.timings)
                mean_time = np.mean(timings_array)
                std_time = np.std(timings_array)
                quantiles = np.percentile(timings_array, [0, 25, 50, 75, 100])
                print(f"{self.func_name} took {elapsed_time:.4f} seconds")
                print(f"Mean Time: {mean_time:.4f} seconds")
                print(f"Std Time: {std_time:.4f} seconds")
                print(
                    f"Quantiles: Min={quantiles[0]:.4f}, 25%={quantiles[1]:.4f}, Median={quantiles[2]:.4f}, 75%={quantiles[3]:.4f}, Max={quantiles[4]:.4f}"
                )

            else:
                print(f"{self.func_name} took {elapsed_time:.4f} seconds")


def profile_function(track_statistics=True, verbose=False):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with ProfileFunction(func.__name__, track_statistics, verbose):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def recursive_apply(inputs, func):
    if isinstance(inputs, list):
        return [recursive_apply(camera, func) for camera in inputs]
    else:
        return func(inputs)


def squeeze_list(nested_list, dim, current_dim=0):
    # If the current dimension is in the list of indices to squeeze
    if isinstance(nested_list, list) and len(nested_list) == 1 and current_dim == dim:
        return squeeze_list(nested_list[0], dim, current_dim + 1)
    elif isinstance(nested_list, list):
        return [squeeze_list(item, dim, current_dim + 1) for item in nested_list]
    else:
        return nested_list


def match_gt(tensor1, tensor2, padding1, padding2, mode: str = "bilinear"):
    """
    Transform each item in tensor1 batch to match tensor2's dimensions and padding.

    Args:
        tensor1 (torch.Tensor): The input tensor to transform, with shape (batch_size, channels, height, width).
        tensor2 (torch.Tensor): The target tensor to match, with shape (batch_size, channels, height, width).
        padding1 (tuple): Padding applied to tensor1 (pad_left, pad_right, pad_top, pad_bottom).
        padding2 (tuple): Desired padding to be applied to match tensor2 (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        torch.Tensor: The batch of transformed tensors matching tensor2's size and padding.
    """
    # Get batch size
    batch_size = len(tensor1)
    src_dtype = tensor1[0].dtype
    tgt_dtype = tensor2[0].dtype

    # List to store transformed tensors
    transformed_tensors = []

    for i in range(batch_size):
        item1 = tensor1[i]
        item2 = tensor2[i]

        h1, w1 = item1.shape[1], item1.shape[2]
        pad1_l, pad1_r, pad1_t, pad1_b = padding1[i] if padding1 is not None else (0, 0, 0, 0)
        pad2_l, pad2_r, pad2_t, pad2_b = padding2[i] if padding2 is not None else (0, 0, 0, 0)
        item1_unpadded = item1[:, pad1_t : h1 - pad1_b, pad1_l : w1 - pad1_r]

        h2, w2 = (
            item2.shape[1] - pad2_t - pad2_b,
            item2.shape[2] - pad2_l - pad2_r,
        )

        item1_resized = F.interpolate(item1_unpadded.unsqueeze(0).to(tgt_dtype), size=(h2, w2), mode=mode)
        item1_padded = F.pad(item1_resized, (pad2_l, pad2_r, pad2_t, pad2_b))
        transformed_tensors.append(item1_padded)

    transformed_batch = torch.cat(transformed_tensors)
    return transformed_batch.to(src_dtype)
