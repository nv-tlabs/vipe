# This file includes code originally from the Geocalib repository:
# https://github.com/cvg/GeoCalib
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

"""Tensor class for gravity vector in camera frame."""

import torch
from torch.nn import functional as F

from .misc import EuclideanManifold, SphericalManifold, TensorWrapper, autocast
from .utils import rad2rotmat

# mypy: ignore-errors


class Gravity(TensorWrapper):
    """Gravity vector in camera frame."""

    eps = 1e-4

    @autocast
    def __init__(self, data: torch.Tensor) -> None:
        """Create gravity vector from data.

        Args:
            data (torch.Tensor): gravity vector as 3D vector in camera frame.
        """
        assert data.shape[-1] == 3, data.shape

        data = F.normalize(data, dim=-1)

        super().__init__(data)

    @classmethod
    def from_rp(cls, roll: torch.Tensor, pitch: torch.Tensor) -> "Gravity":
        """Create gravity vector from roll and pitch angles."""
        if not isinstance(roll, torch.Tensor):
            roll = torch.tensor(roll)
        if not isinstance(pitch, torch.Tensor):
            pitch = torch.tensor(pitch)

        sr, cr = torch.sin(roll), torch.cos(roll)
        sp, cp = torch.sin(pitch), torch.cos(pitch)
        return cls(torch.stack([-sr * cp, -cr * cp, sp], dim=-1))

    @property
    def vec3d(self) -> torch.Tensor:
        """Return the gravity vector in the representation."""
        return self._data

    @property
    def x(self) -> torch.Tensor:
        """Return first component of the gravity vector."""
        return self._data[..., 0]

    @property
    def y(self) -> torch.Tensor:
        """Return second component of the gravity vector."""
        return self._data[..., 1]

    @property
    def z(self) -> torch.Tensor:
        """Return third component of the gravity vector."""
        return self._data[..., 2]

    @property
    def roll(self) -> torch.Tensor:
        """Return the roll angle of the gravity vector."""
        roll = torch.asin(-self.x / (torch.sqrt(1 - self.z**2) + self.eps))
        offset = -torch.pi * torch.sign(self.x)
        return torch.where(self.y < 0, roll, -roll + offset)

    def J_roll(self) -> torch.Tensor:
        """Return the Jacobian of the roll angle of the gravity vector."""
        cp, _ = torch.cos(self.pitch), torch.sin(self.pitch)
        cr, sr = torch.cos(self.roll), torch.sin(self.roll)
        Jr = self.new_zeros(self.shape + (3,))
        Jr[..., 0] = -cr * cp
        Jr[..., 1] = sr * cp
        return Jr

    @property
    def pitch(self) -> torch.Tensor:
        """Return the pitch angle of the gravity vector."""
        return torch.asin(self.z)

    def J_pitch(self) -> torch.Tensor:
        """Return the Jacobian of the pitch angle of the gravity vector."""
        cp, sp = torch.cos(self.pitch), torch.sin(self.pitch)
        cr, sr = torch.cos(self.roll), torch.sin(self.roll)

        Jp = self.new_zeros(self.shape + (3,))
        Jp[..., 0] = sr * sp
        Jp[..., 1] = cr * sp
        Jp[..., 2] = cp
        return Jp

    @property
    def rp(self) -> torch.Tensor:
        """Return the roll and pitch angles of the gravity vector."""
        return torch.stack([self.roll, self.pitch], dim=-1)

    def J_rp(self) -> torch.Tensor:
        """Return the Jacobian of the roll and pitch angles of the gravity vector."""
        return torch.stack([self.J_roll(), self.J_pitch()], dim=-1)

    @property
    def R(self) -> torch.Tensor:
        """Return the rotation matrix from the gravity vector."""
        return rad2rotmat(roll=self.roll, pitch=self.pitch)

    def J_R(self) -> torch.Tensor:
        """Return the Jacobian of the rotation matrix from the gravity vector."""
        raise NotImplementedError

    def update(self, delta: torch.Tensor, spherical: bool = False) -> "Gravity":
        """Update the gravity vector by adding a delta."""
        if spherical:
            data = SphericalManifold.plus(self.vec3d, delta)
            return self.__class__(data)

        data = EuclideanManifold.plus(self.rp, delta)
        return self.from_rp(data[..., 0], data[..., 1])

    def J_update(self, spherical: bool = False) -> torch.Tensor:
        """Return the Jacobian of the update."""
        return SphericalManifold.J_plus(self.vec3d) if spherical else EuclideanManifold.J_plus(self.vec3d)

    def __repr__(self):
        """Print the Camera object."""
        return f"{self.__class__.__name__} {self.shape} {self.dtype} {self.device}"
