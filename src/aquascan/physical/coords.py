"""Common physical coordinate frame (Gate 10, section 19).

Defines the transformation from GNSS coordinates to local survey
coordinates so ERT and seismic observations are alignable via:
profile distance, x/y, elevation, depth.

Elevation types (ellipsoidal vs orthometric vs local z) are tracked
explicitly and never silently mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class ElevationType(StrEnum):
    """Elevation reference frame."""

    ELLIPSOIDAL = "ELLIPSOIDAL"  # h, relative to WGS84 ellipsoid
    ORTHOMETRIC = "ORTHOMETRIC"  # H, relative to geoid (≈ mean sea level)
    LOCAL_Z = "LOCAL_Z"  # arbitrary local vertical datum


@dataclass
class Elevation:
    """An elevation value with an explicit reference frame."""

    value_m: float
    type: ElevationType
    note: str = ""

    def as_orthometric(self, geoid_undulation_m: float | None = None) -> Elevation:
        """Convert to orthometric when a geoid undulation is available.

        H = h - N, where N is the geoid undulation (ellipsoidal minus
        orthometric). Conversion between LOCAL_Z and orthometric requires
        a known local datum offset and is refused here (no silent mixing).
        """
        if self.type == ElevationType.ORTHOMETRIC:
            return Elevation(self.value_m, ElevationType.ORTHOMETRIC, self.note)
        if self.type == ElevationType.ELLIPSOIDAL:
            if geoid_undulation_m is None:
                raise ValueError(
                    "Cannot convert ELLIPSOIDAL elevation to ORTHOMETRIC "
                    "without a geoid undulation (N). No silent mixing."
                )
            return Elevation(
                self.value_m - geoid_undulation_m,
                ElevationType.ORTHOMETRIC,
                f"converted via N={geoid_undulation_m} m",
            )
        raise ValueError(
            "LOCAL_Z elevation cannot be converted to ORTHOMETRIC without a "
            "documented local datum offset. Refusing silent mixing."
        )


def _lla_to_ecef(lat_deg: float, lon_deg: float, h_ell_m: float) -> tuple[float, float, float]:
    """WGS84 geodetic (lat, lon, ellipsoidal h) to ECEF."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x = (n + h_ell_m) * np.cos(lat) * np.cos(lon)
    y = (n + h_ell_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - e2) + h_ell_m) * np.sin(lat)
    return float(x), float(y), float(z)


def _ecef_to_enu(
    x: float, y: float, z: float, lat0_deg: float, lon0_deg: float
) -> tuple[float, float, float]:
    """ECEF to local East-North-Up at origin (lat0, lon0)."""
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    dx, dy, dz = x, y, z  # relative to origin ECEF = (0,0,0) shift handled by caller
    e = -np.sin(lon0) * dx + np.cos(lon0) * dy
    n = -np.sin(lat0) * np.cos(lon0) * dx - np.sin(lat0) * np.sin(lon0) * dy + np.cos(lat0) * dz
    u = np.cos(lat0) * np.cos(lon0) * dx + np.cos(lat0) * np.sin(lon0) * dy + np.sin(lat0) * dz
    return float(e), float(n), float(u)


@dataclass
class SurveyPoint:
    """A point in the local survey frame."""

    easting_m: float
    northing_m: float
    elevation: Elevation
    profile_distance_m: float | None = None
    lateral_m: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class LocalSurveyFrame:
    """Transforms GNSS coordinates to a local survey frame.

    Origin is a reference GNSS point. The survey azimuth defines the
    profile axis; profile distance = distance along the azimuth, lateral
    = perpendicular offset. All GNSS positions are first reduced to a
    common ellipsoidal height before projection.
    """

    def __init__(
        self,
        origin_lat_deg: float,
        origin_lon_deg: float,
        survey_azimuth_deg: float = 0.0,
    ) -> None:
        self.origin_lat_deg = origin_lat_deg
        self.origin_lon_deg = origin_lon_deg
        self.azimuth_rad = np.deg2rad(survey_azimuth_deg)
        self.origin_ecef = _lla_to_ecef(origin_lat_deg, origin_lon_deg, 0.0)

    def gnss_to_local(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation: Elevation,
        geoid_undulation_m: float | None = None,
    ) -> SurveyPoint:
        """Project a GNSS point into the local survey frame.

        All elevations are normalized to orthometric when N is provided;
        otherwise the frame refuses to mix reference frames.
        """
        ortho = elevation.as_orthometric(geoid_undulation_m)
        h_ell = ortho.value_m + (geoid_undulation_m or 0.0)
        ecef = _lla_to_ecef(lat_deg, lon_deg, h_ell)
        ex = ecef[0] - self.origin_ecef[0]
        ey = ecef[1] - self.origin_ecef[1]
        ez = ecef[2] - self.origin_ecef[2]
        east, north, up = _ecef_to_enu(ex, ey, ez, self.origin_lat_deg, self.origin_lon_deg)
        # rotate into survey profile axis. Azimuth is measured clockwise
        # from North (standard survey convention): a point due east of the
        # origin sits at profile_distance = 0, lateral = +east.
        c, s = np.cos(self.azimuth_rad), np.sin(self.azimuth_rad)
        along = east * s + north * c
        lateral = east * c - north * s
        return SurveyPoint(
            easting_m=east,
            northing_m=north,
            elevation=Elevation(up, ElevationType.ORTHOMETRIC, "derived from GNSS + N"),
            profile_distance_m=float(along),
            lateral_m=float(lateral),
        )


def register_survey_points(
    points: list[SurveyPoint],
) -> tuple[list[float], list[float], list[float]]:
    """Return (profile_distance, lateral, orthometric_elevation) arrays.

    Refuses to combine points whose elevations mix reference frames.
    """
    if not points:
        return [], [], []
    first_type = points[0].elevation.type
    if any(p.elevation.type != first_type for p in points):
        raise ValueError("Cannot register survey points mixing elevation frames.")
    return (
        [p.profile_distance_m for p in points],
        [p.lateral_m for p in points],
        [p.elevation.value_m for p in points],
    )
