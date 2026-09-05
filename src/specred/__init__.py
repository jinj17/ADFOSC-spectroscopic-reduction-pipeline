"""
specred: a long-slit spectroscopy reduction pipeline.

Stages (see README for the full walkthrough):
    specred.io               bias combination/subtraction, frame trimming
    specred.calib.flat       flat-field response + flat correction + cosmic-ray removal
    specred.extraction       trace + boxcar (aperture) extraction with sky subtraction
    specred.calib.wavelength wavelength calibration
    specred.calib.flux       flux calibration (sensitivity function, airmass correction)
    specred.output           final spectrum plots/saving

This package is GUI-independent: everything here can be scripted
directly. The PyQt GUI in `gui/` is a thin front-end over these
modules.
"""

__version__ = "0.1.0"
