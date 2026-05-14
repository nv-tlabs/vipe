# This file includes code originally from the Geocalib repository:
# https://github.com/cvg/GeoCalib
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

"""Miscellaneous functions and classes for the geocalib_inference package."""

import functools
import inspect
import logging
from typing import Callable, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

# mypy: ignore-errors


def autocast(func: Callable) -> Callable:
    """Cast the inputs of a TensorWrapper method to PyTorch tensors if they are numpy arrays.

    Use the device and dtype of the wrapper.

    Args:
        func (Callable): Method of a TensorWrapper class.

    Returns:
        Callable: Wrapped method.
    """

    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device("cpu")
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


class TensorWrapper:
    """Wrapper for PyTorch tensors."""

    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        """Wrapper for PyTorch tensors."""
        self._data = data

    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying tensor."""
        return self._data.shape[:-1]

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying tensor."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the underlying tensor."""
        return self._data.dtype

    def __getitem__(self, index) -> torch.Tensor:
        """Get the underlying tensor."""
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        """Set the underlying tensor."""
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        """Move the underlying tensor to a new device."""
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        """Move the underlying tensor to the CPU."""
        return self.__class__(self._data.cpu())

    def cuda(self):
        """Move the underlying tensor to the GPU."""
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        """Pin the underlying tensor to memory."""
        return self.__class__(self._data.pin_memory())

    def float(self):
        """Cast the underlying tensor to float."""
        return self.__class__(self._data.float())

    def double(self):
        """Cast the underlying tensor to double."""
        return self.__class__(self._data.double())

    def detach(self):
        """Detach the underlying tensor."""
        return self.__class__(self._data.detach())

    def numpy(self):
        """Convert the underlying tensor to a numpy array."""
        return self._data.detach().cpu().numpy()

    def new_tensor(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_tensor(*args, **kwargs)

    def new_zeros(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_zeros(*args, **kwargs)

    def new_ones(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_ones(*args, **kwargs)

    def new_full(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_full(*args, **kwargs)

    def new_empty(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_empty(*args, **kwargs)

    def unsqueeze(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self.__class__(self._data.unsqueeze(*args, **kwargs))

    def squeeze(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self.__class__(self._data.squeeze(*args, **kwargs))

    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        """Stack a list of objects with the same type and shape."""
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Support torch functions."""
        if kwargs is None:
            kwargs = {}
        return cls.stack(*args, **kwargs) if func is torch.stack else NotImplemented


class EuclideanManifold:
    """Simple euclidean manifold."""

    @staticmethod
    def J_plus(x: torch.Tensor) -> torch.Tensor:
        """Plus operator Jacobian."""
        return torch.eye(x.shape[-1]).to(x)

    @staticmethod
    def plus(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Plus operator."""
        return x + delta


class SphericalManifold:
    """Implementation of the spherical manifold.

    Following the derivation from 'Integrating Generic Sensor Fusion Algorithms with Sound State
    Representations through Encapsulation of Manifolds' by Hertzberg et al. (B.2, p. 25).

    Householder transformation following Algorithm 5.1.1 (p. 210) from 'Matrix Computations' by
    Golub et al.
    """

    @staticmethod
    def householder_vector(x: torch.Tensor) -> torch.Tensor:
        """Return the Householder vector and beta.

        Algorithm 5.1.1 (p. 210) from 'Matrix Computations' by Golub et al. (Johns Hopkins Studies
        in Mathematical Sciences) but using the nth element of the input vector as pivot instead of
        first.

        This computes the vector v with v(n) = 1 and beta such that H = I - beta * v * v^T is
        orthogonal and H * x = ||x||_2 * e_n.

        Args:
            x (torch.Tensor): [..., n] tensor.

        Returns:
            torch.Tensor: v of shape [..., n]
            torch.Tensor: beta of shape [...]
        """
        sigma = torch.sum(x[..., :-1] ** 2, -1)
        xpiv = x[..., -1]
        norm = torch.norm(x, dim=-1)
        if torch.any(sigma < 1e-7):
            sigma = torch.where(sigma < 1e-7, sigma + 1e-7, sigma)
            logger.warning("sigma < 1e-7")

        vpiv = torch.where(xpiv < 0, xpiv - norm, -sigma / (xpiv + norm))
        beta = 2 * vpiv**2 / (sigma + vpiv**2)
        v = torch.cat([x[..., :-1] / vpiv[..., None], torch.ones_like(vpiv)[..., None]], -1)
        return v, beta

    @staticmethod
    def apply_householder(y: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply Householder transformation.

        Args:
            y (torch.Tensor): Vector to transform of shape [..., n].
            v (torch.Tensor): Householder vector of shape [..., n].
            beta (torch.Tensor): Householder beta of shape [...].

        Returns:
            torch.Tensor: Transformed vector of shape [..., n].
        """
        return y - v * (beta * torch.einsum("...i,...i->...", v, y))[..., None]

    @classmethod
    def J_plus(cls, x: torch.Tensor) -> torch.Tensor:
        """Plus operator Jacobian."""
        v, beta = cls.householder_vector(x)
        H = -torch.einsum("..., ...k, ...l->...kl", beta, v, v)
        H = H + torch.eye(H.shape[-1]).to(H)
        return H[..., :-1]  # J

    @classmethod
    def plus(cls, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Plus operator.

        Equation 109 (p. 25) from 'Integrating Generic Sensor Fusion Algorithms with Sound State
        Representations through Encapsulation of Manifolds' by Hertzberg et al. but using the nth
        element of the input vector as pivot instead of first.

        Args:
            x: point on the manifold
            delta: tangent vector
        """
        eps = 1e-7
        # keep norm is not equal to 1
        nx = torch.norm(x, dim=-1, keepdim=True)
        nd = torch.norm(delta, dim=-1, keepdim=True)

        # make sure we don't divide by zero in backward as torch.where computes grad for both
        # branches
        nd_ = torch.where(nd < eps, nd + eps, nd)
        sinc = torch.where(nd < eps, nd.new_ones(nd.shape), torch.sin(nd_) / nd_)

        # cos is applied to last dim instead of first
        exp_delta = torch.cat([sinc * delta, torch.cos(nd)], -1)

        v, beta = cls.householder_vector(x)
        return nx * cls.apply_householder(exp_delta, v, beta)


@torch.jit.script
def J_vecnorm(vec: torch.Tensor) -> torch.Tensor:
    """Compute the jacobian of vec / norm2(vec).

    Args:
        vec (torch.Tensor): [..., D] tensor.

    Returns:
        torch.Tensor: [..., D, D] Jacobian.
    """
    D = vec.shape[-1]
    norm_x = torch.norm(vec, dim=-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)

    if (norm_x == 0).any():
        norm_x = norm_x + 1e-6

    xxT = torch.einsum("...i,...j->...ij", vec, vec)  # (..., D, D)
    identity = torch.eye(D, device=vec.device, dtype=vec.dtype)  # (D, D)

    return identity / norm_x - (xxT / norm_x**3)  # (..., D, D)


@torch.jit.script
def J_focal2fov(focal: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute the jacobian of the focal2fov function."""
    return -4 * h / (4 * focal**2 + h**2)


@torch.jit.script
def J_up_projection(uv: torch.Tensor, abc: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
    """Compute the jacobian of the up-vector projection.

    Args:
        uv (torch.Tensor): Normalized image coordinates of shape (..., 2).
        abc (torch.Tensor): Gravity vector of shape (..., 3).
        wrt (str, optional): Parameter to differentiate with respect to. Defaults to "uv".

    Raises:
        ValueError: If the wrt parameter is unknown.

    Returns:
        torch.Tensor: Jacobian with respect to the parameter.
    """
    if wrt == "uv":
        c = abc[..., 2][..., None, None, None]
        return -c * torch.eye(2, device=uv.device, dtype=uv.dtype).expand(uv.shape[:-1] + (2, 2))

    elif wrt == "abc":
        J = uv.new_zeros(uv.shape[:-1] + (2, 3))
        J[..., 0, 0] = 1
        J[..., 1, 1] = 1
        J[..., 0, 2] = -uv[..., 0]
        J[..., 1, 2] = -uv[..., 1]
        return J

    else:
        raise ValueError(f"Unknown wrt: {wrt}")
