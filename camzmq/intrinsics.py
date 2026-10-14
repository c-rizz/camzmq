"""Pinhole intrinsics, and how they change when an image is cropped and resized."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels, OpenCV convention: pixel centers at integer coordinates."""
    fx : float
    fy : float
    cx : float
    cy : float

    def cropped(self, top : int, left : int) -> Intrinsics:
        """Intrinsics of the image cropped to start at row `top` and column `left`."""
        return Intrinsics(self.fx, self.fy, self.cx - left, self.cy - top)

    def resized(self, src_hw : tuple[int,int], out_hw : tuple[int,int]) -> Intrinsics:
        """Intrinsics of the image resized from src_hw to out_hw (same field of view)."""
        sy = out_hw[0]/src_hw[0]
        sx = out_hw[1]/src_hw[1]
        return Intrinsics(self.fx*sx, self.fy*sy, (self.cx + 0.5)*sx - 0.5, (self.cy + 0.5)*sy - 0.5)

    def fovx_deg(self, width : int) -> float:
        return math.degrees(2*math.atan(width/(2*self.fx)))

    def fovy_deg(self, height : int) -> float:
        return math.degrees(2*math.atan(height/(2*self.fy)))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @staticmethod
    def from_dict(d : dict) -> Intrinsics:
        return Intrinsics(fx=float(d["fx"]), fy=float(d["fy"]), cx=float(d["cx"]), cy=float(d["cy"]))

    @staticmethod
    def from_fovy(height : int, width : int, fovy_deg : float) -> Intrinsics:
        """Square pixels and a centered principal point, as a simulated camera has."""
        fy = height/(2*math.tan(math.radians(fovy_deg)/2))
        return Intrinsics(fx=fy, fy=fy, cx=(width - 1)/2, cy=(height - 1)/2)
