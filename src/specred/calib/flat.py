"""ADFOSC spectral-flat normalization and flat-field correction.

Python 3.8-compatible. No plotting/GUI dependency. Designed to be called by
PyQt5 / PyQtGraph GUI; returns arrays for the GUI to visualize.
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units as u
from astropy.convolution import convolve, Box1DKernel
import ccdproc
from scipy.interpolate import CubicSpline

__all__ = ['lamp_response', 'flat_correction', 'read_ccd_data']


def read_ccd_data(source):
    """Read a path, CCDData, HDU or HDUList into an independent CCDData (ADU)."""
    if isinstance(source, CCDData):
        return source.copy()
    if isinstance(source, (str, Path)):
        with fits.open(str(source), memmap=False) as hdul:
            hdu = next((h for h in hdul if h.data is not None and np.ndim(h.data) == 2), None)
            if hdu is None:
                raise ValueError('No 2D image found in {}'.format(source))
            return CCDData(np.asarray(hdu.data, dtype=float).copy(), unit=u.adu, meta=hdu.header.copy())
    if isinstance(source, fits.HDUList):
        hdu = next((h for h in source if h.data is not None and np.ndim(h.data) == 2), None)
        if hdu is None:
            raise ValueError('No 2D image found in HDUList')
        return CCDData(np.asarray(hdu.data, dtype=float).copy(), unit=u.adu, meta=hdu.header.copy())
    if isinstance(source, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        return CCDData(np.asarray(source.data, dtype=float).copy(), unit=u.adu, meta=source.header.copy())
    if isinstance(source, np.ndarray) and source.ndim == 2:
        return CCDData(np.asarray(source, dtype=float).copy(), unit=u.adu)
    raise TypeError('Expected CCDData, FITS image, path or 2D numpy array')


def _write_fits(path, image, header=None, history=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = header.copy() if header is not None else fits.Header()
    if history:
        hdr.add_history(history)
    fits.writeto(str(path), np.asarray(image, dtype=np.float32), hdr, overwrite=True)
    return str(path)


def lamp_response(file, flat_combine=False, smooth=False, npix=11, display=False,
                  Saxis=0, Waxis=None, ax=None, func=False, output='nflat.fits',
                  return_response=False):
    """Build a normalized continuum flat by removing its 1D lamp spectrum.

    `file`: FITS path, CCDData, or nonempty list/tuple thereof.
    `Saxis`: spatial axis (0 for usual FITS [y, x], 1 for swapped).
    `smooth`: boxcar smooth the 1D spectral response along dispersion.
    `func`: retained for legacy compatibility; no redundant interpolating
        spline is fitted, since that reproduces exactly the same samples.
    `display` / `ax`: accepted but never open a Matplotlib window.
    `output`: output path; None disables writing.
    Returns normalized flat array, or (flat, response, raw_profile) if
    `return_response=True`.
    """
    if Saxis not in (0, 1) or (Waxis is not None and Waxis != 1 - Saxis):
        raise ValueError('Saxis/Waxis must specify complementary axes 0 and 1')
    sources = file if isinstance(file, (list, tuple)) else [file]
    if not sources:
        raise ValueError('At least one continuum flat is required')
    frames = [read_ccd_data(source) for source in sources]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError('Continuum flats have inconsistent dimensions')
    if len(frames) > 1:
        if not flat_combine:
            raise ValueError('Multiple flats provided: set flat_combine=True to combine them')
        combined = ccdproc.combine(frames, method='median', sigma_clip=True,
                                   sigma_clip_low_thresh=5, sigma_clip_high_thresh=5)
    else:
        combined = frames[0]
    data = np.asarray(combined.data, dtype=float)
    raw_profile = np.nanmedian(data, axis=Saxis)
    response = raw_profile.copy()
    if smooth:
        npix = int(npix)
        if npix < 1 or npix > len(response):
            raise ValueError('npix must be from 1 to the response length')
        #response = convolve(response, Box1DKernel(npix), boundary='extend', nan_treatment='interpolate', preserve_nan=False)
        response = convolve(response, Box1DKernel(npix), boundary='extend')
        if func is False:
            # No 1D profile interpolation is needed when evaluating on its original pixels.
            good = np.isfinite(response) & (response > 0)
            if good.sum() != len(response):
                raise ValueError('Continuum flat response contains nonpositive or invalid values; check illumination and cropping')
            denominator = response[None, :] if Saxis == 0 else response[:, None]
            normalized = np.divide(data, denominator, out=np.full_like(data, np.nan), where=np.isfinite(denominator) & (denominator > 0))
            # Robust global normalization ensures typical illuminated pixel response is ~1.
            finite_positive = normalized[np.isfinite(normalized) & (normalized > 0)]
            if not finite_positive.size:
                raise ValueError('Normalized flat has no usable positive pixels')
            scale = np.median(finite_positive)
            normalized /= scale
        elif func:
            x = np.arange(1,len(response)+1,1)
            cs = CubicSpline(x,response)
            denominator = cs(x)
            normalized = np.divide(data, denominator, out=np.full_like(data, np.nan), where=np.isfinite(denominator) & (denominator > 0))
    if func and not smooth:
        x = np.arange(1,len(response)+1,1)
        cs = CubicSpline(x,response)
        denominator = cs(x)    
        normalized = np.divide(data, denominator, out=np.full_like(data, np.nan), where=np.isfinite(denominator) & (denominator > 0))
            
    if output is not None:
        _write_fits(output, normalized, combined.header,
                    'Continuum spectrum removed; spatial axis {}; smoothed={}'.format(Saxis, bool(smooth)))
    if return_response:
        return normalized, response, raw_profile
    return normalized


def flat_correction(cor_list, flat, cosmic_correction=True, output_dir=None,
                    gain=1.0, readnoise=6.0, sigclip=5.0, skip_arc_cosmics=True):
    """Flat-correct bias-subtracted FITS frames, optionally clean cosmic rays.

    Input filenames retain their original relative naming convention, with
    `fc_` prefixed to each basename. Returns output filenames relative to the
    current working directory if output_dir is None, or full paths otherwise.
    Arc files (com_* and lamp/arc-named files) are not CR-cleaned by default:
    sharp emission features may be mistaken for cosmic rays.
    """
    if not cor_list:
        raise ValueError('No science/standard/arc frames provided')
    if gain <= 0 or readnoise < 0:
        raise ValueError('Gain must be positive and read noise nonnegative')
    flat_data = np.asarray(read_ccd_data(flat).data, dtype=float)
##    bad_flat = ~np.isfinite(flat_data) | (flat_data <= 0)
##    if np.any(bad_flat):
##        raise ValueError('Normalized flat has {} invalid/nonpositive pixels; use a valid illumination mask'.format(int(bad_flat.sum())))
    products = []
    for filename in cor_list:
        frame = read_ccd_data(filename)
        if frame.shape != flat_data.shape:
            raise ValueError('Flat {} does not match image {} shape {}'.format(flat_data.shape, filename, frame.shape))
        corrected = np.asarray(frame.data, dtype=float) / flat_data
        source_name = Path(filename).name
        is_arc = source_name.lower().startswith(('com_', 'bs_com_', 'fc_com_')) or 'lamp' in source_name.lower() or 'arc' in source_name.lower()
        run_cosmic = bool(cosmic_correction and not (skip_arc_cosmics and is_arc))
        if run_cosmic:
            ccd = CCDData(corrected, unit=u.adu, meta=frame.header.copy())
            cleaned = ccdproc.cosmicray_lacosmic(ccd, gain=gain*u.electron/u.adu,
                readnoise=readnoise*u.electron, sigclip=sigclip, gain_apply=False, verbose=False)
            corrected = np.asarray(cleaned.data, dtype=float)
        out_name = 'fc_' + source_name
        dest = Path(output_dir) / out_name if output_dir else Path(out_name)
        _write_fits(dest, corrected, frame.header, 'Normalized flat applied; cosmic rays cleaned={}'.format(run_cosmic))
        products.append(str(dest))
        print('Flat-corrected: {} (cosmic={})'.format(dest, run_cosmic))
    return products
