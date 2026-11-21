"""vkastro: Astropy-style Vulkan GPU astrophysics (SkyCoord, photometry, cosmology, FITS)."""

from .coordinates import (
    SkyCoord,
    aitoff_projection,
    angular_separation,
    galactic_to_icrs,
    icrs_to_galactic,
    spherical_to_cartesian,
)
from .cosmology import FlatLambdaCDM
from .fits import read_fits
from .photometry import (
    absolute_magnitude,
    bp_rp_to_temperature,
    distance_modulus,
    luminosity_solar,
    parallax_to_distance,
)

__all__ = [
    "FlatLambdaCDM",
    "SkyCoord",
    "absolute_magnitude",
    "aitoff_projection",
    "angular_separation",
    "bp_rp_to_temperature",
    "distance_modulus",
    "galactic_to_icrs",
    "icrs_to_galactic",
    "luminosity_solar",
    "parallax_to_distance",
    "read_fits",
    "spherical_to_cartesian",
]
