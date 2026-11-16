"""ADFOSC extraction core: predictive trace, box and profile-weighted extraction.

Python 3.8 compatible. No plotting or interactive input. Coordinates are 0-based.
The optimal method follows the variance-weighted profile estimator of Horne (1986),
with a Gaussian spatial-profile approximation; it is not an exact IRAF apall clone.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.io import fits
from specutils import Spectrum1D
from scipy.interpolate import UnivariateSpline
from datetime import datetime
from astropy.stats import sigma_clip

__all__ = ['trace_sloped', 'trfunc', 'BoxcarExtract', 'boxfunc', 'extract_spectrum']



def log_trace_read(log_file,filename,trace,cen,Xlim,nbins):
    with open(log_file, "a") as log:
        obj_name =filename
        log.write(obj_name+"\n")
        log.write("Trace: " +str(trace) + "\n")
        log.write("y-line:"+str(cen)+"\t"+"Xlim:"+str(Xlim)+"\t"+"nbins:"+str(nbins)+"\n")
        log.write("--"*50 + "\n")
        
def log_ap_read(log_file,filename,apwidth,skyw,skys,skyd):
    with open(log_file,"a") as log:
        obj_name = filename
        log.write("Aperture info for "+obj_name+"\n")
        log.write("Apwidth:"+str(apwidth)+" pix from central dispersion line" + "\n")
        log.write(" Sky Seperation from central dispersion line :"+str(skys)+"\n")
        log.write(" Sky line width :"+str(skyw)+"\n")
        log.write(" Sky degree :"+str(skyd)+"\n")
        
def _array(img, Saxis=0):
    if Saxis not in (0, 1):
        raise ValueError('Saxis must be 0 or 1')
    a = np.asarray(img.data if hasattr(img, 'data') else img, dtype=float)
    if a.ndim != 2:
        raise ValueError('Image must be a 2D array')
    return a if Saxis == 0 else a.T

def _gaussian(y, amp, baseline, center, sigma):
    return baseline + amp * np.exp(-0.5 * ((y-center)/sigma)**2)

def trace_sloped(img, guess, Xlim=None, bin_width=20, search_halfwidth=12, max_shift=8,
                 poly_order=1, Saxis=0, return_details=False):
    """Fit local Gaussian peaks from the central bin outward, then robust polynomial trace."""
    data = _array(img, Saxis)
    ny, nxfull = data.shape
    nx = nxfull if Xlim is None else min(int(Xlim), nxfull)
    if nx < 6 or not 0 <= guess < ny or bin_width < 1 or search_halfwidth < 3 or max_shift <= 0:
        raise ValueError('Invalid trace parameters/initial center')
    if poly_order not in (1, 2):
        raise ValueError('poly_order must be 1 or 2')
    edges = np.r_[np.arange(0, nx, bin_width, dtype=int), nx]
    centers = (edges[:-1] + edges[1:] - 1) / 2.
    measured = np.full(len(centers), np.nan)
    anchor = int(np.argmin(abs(centers - (nx - 1) / 2)))

    def fit_bin(i, prediction):
        if not np.isfinite(prediction) or not 0 <= prediction < ny:
            return np.nan
        lo = max(0, int(np.floor(prediction-search_halfwidth)))
        hi = min(ny, int(np.ceil(prediction+search_halfwidth))+1)
        y = np.arange(lo, hi, dtype=float)
        region = data[lo:hi, edges[i]:edges[i+1]]
        count = np.isfinite(region).sum(axis=1)
        z = np.nansum(region, axis=1)
        valid = (count > 0) & np.isfinite(z)
        if valid.sum() < 5:
            return np.nan
        y, z = y[valid], z[valid]
        baseline = np.median(z)
        p = int(np.argmax(z))
        if z[p] <= baseline or abs(y[p]-prediction) > max_shift:
            return np.nan
        lower = max(float(y.min()), prediction-max_shift)
        upper = min(float(y.max()), prediction+max_shift)
        if lower >= upper:
            return np.nan
        p0 = [z[p]-baseline, baseline, np.clip(y[p], lower+1e-5, upper-1e-5), 2.5]
        try:
            pars, _ = curve_fit(_gaussian, y, z, p0=p0,
                                bounds=([0, -np.inf, lower, 0.5],
                                        [np.inf, np.inf, upper, max(3., 2.*search_halfwidth)]), maxfev=2000)
            return float(pars[2])
        except (ValueError, RuntimeError, FloatingPointError):
            return np.nan

    measured[anchor] = fit_bin(anchor, float(guess))
    if not np.isfinite(measured[anchor]):
        raise RuntimeError('Cannot locate target near initial Y center')
    for direction in (1, -1):
        ids = range(anchor+direction, len(centers) if direction > 0 else -1, direction)
        history = [anchor]
        for i in ids:
            if len(history) >= 2:
                recent = history[-min(5, len(history)):]
                slope, intercept = np.polyfit(centers[recent], measured[recent], 1)
                prediction = slope*centers[i]+intercept
            else:
                prediction = measured[history[-1]]
            measured[i] = fit_bin(i, prediction)
            if np.isfinite(measured[i]):
                history.append(i)
    good = np.isfinite(measured)
    if good.sum() < max(4, poly_order+2):
        raise RuntimeError('Too few reliable trace bins; adjust search/max shift')
    keep = good.copy()
    for _ in range(5):
        coeff = np.polyfit(centers[keep], measured[keep], poly_order)
        residual = measured-np.polyval(coeff, centers)
        center = np.median(residual[keep])
        scatter = 1.4826*np.median(abs(residual[keep]-center))
        if scatter < 1e-6 or not np.isfinite(scatter):
            break
        candidate = good & (abs(residual-center) <= max(3*scatter, 0.5))
        if candidate.sum() < max(4, poly_order+2) or np.array_equal(candidate, keep):
            break
        keep = candidate
    coeff = np.polyfit(centers[keep], measured[keep], poly_order)
    trace = np.polyval(coeff, np.arange(nx))
    if return_details:
        return trace, {'bin_x': centers, 'bin_y': measured, 'accepted': keep, 'coeff': coeff}
    return trace

def trfunc(data, filename=None, nbins=20, window=12, guess=500, Saxis=0, Xlim=1300,
           display=False, otype='src', bin_width=None, max_shift=8, poly_order=1,
           return_details=False, **kwargs):
    """Compatibility wrapper: nbins retained as approximate TOTAL bin count (old GUI meaning).
    Supply bin_width explicitly to use Quicklook-style pixels per bin.
    """
    nx = _array(data, Saxis).shape[1]
    limit = nx if Xlim is None else min(int(Xlim), nx)
    width = int(bin_width) if bin_width is not None else max(1, int(np.ceil(limit/max(4, int(nbins)))))
    return trace_sloped(data, guess, Xlim=limit, bin_width=width, search_halfwidth=window,
                        max_shift=max_shift, poly_order=poly_order, Saxis=Saxis,
                        return_details=return_details)

def extract_spectrum(img, trace_line, apwidth=7, skysep=6, skywidth=7, skydeg=0, method='sum',
                     Saxis=0, gain=1.0, readnoise=6.0, profile_sigma=2.5, profile_smooth=4,
                     reject_sigma=5., otype='src'):
    """Return extraction arrays (raw, sky, net, uncertainty) in ADU.

    Method 'sum': top-hat aperture. Method 'variance'/'optimal': Horne-like profile
    weighted extraction with Gaussian profile, sky subtracted before weighting.
    The raw/sky decomposition for optimal extraction is defined using the same
    weights; its difference is exactly the optimal sky-subtracted result.
    """
    data = _array(img, Saxis)
    ny, nx = data.shape
    trace = np.asarray(trace_line, dtype=float)
    if trace.ndim != 1 or len(trace) < 1 or len(trace) > nx:
        raise ValueError('Invalid trace length')
    if apwidth < 1 or skysep < 1 or skywidth < 1 or skydeg not in (0, 1, 2):
        raise ValueError('Invalid aperture/sky configuration')
    if gain <= 0 or readnoise < 0 or profile_sigma <= 0:
        raise ValueError('Invalid detector parameters')
    mode = {'average': 'sum', 'box': 'sum', 'invvar': 'variance', 'optimal': 'variance'}.get(method, method)
    if mode not in ('sum', 'variance'):
        raise ValueError('Unknown extraction method: {}'.format(method))
    n = len(trace)
    raw = np.full(n, np.nan)
    sky = np.full(n, np.nan)
    net = np.full(n, np.nan)
    err = np.full(n, np.nan)
    profile_model = np.zeros((2*apwidth+1, n), dtype=float)
    sky_var_col = np.full(n, np.nan)
    y_offsets = np.arange(-apwidth, apwidth+1, dtype=int)
    ap_columns = []
    for x, cy in enumerate(trace):
        if not np.isfinite(cy):
            ap_columns.append(None)
            continue
        c = int(np.rint(cy))
        ypos = c+y_offsets
        good_y = (ypos >= 0) & (ypos < ny)
        ap_y = ypos[good_y]
        low = np.arange(c-apwidth-skysep-skywidth, c-apwidth-skysep)
        high = np.arange(c+apwidth+skysep+1, c+apwidth+skysep+skywidth+1)
        sy = np.r_[low, high]
        sy = sy[(sy >= 0) & (sy < ny)]
        sy = sy[np.isfinite(data[sy, x])]
        ap_valid = np.isfinite(data[ap_y, x])
        ap_y, ap_index = ap_y[ap_valid], np.where(good_y)[0][ap_valid]
        if len(ap_y) < 2 or len(sy) < max(3, skydeg+2):
            ap_columns.append(None)
            continue
        values = data[ap_y, x]
        sky_values = data[sy, x]
        if otype == 'lamp':
            sky_bg = np.zeros(len(ap_y))
            bg_residual = sky_values - np.median(sky_values)
        elif skydeg == 0:
            sky_bg = np.full(len(ap_y), np.median(sky_values))
            bg_residual = sky_values-np.median(sky_values)
        else:
            poly = np.polyfit(sy-cy, sky_values, skydeg)
            sky_bg = np.polyval(poly, ap_y-cy)
            bg_residual = sky_values-np.polyval(poly, sy-cy)
        robust_sigma = 1.4826*np.median(abs(bg_residual-np.median(bg_residual)))
        empirical_var = robust_sigma**2 if np.isfinite(robust_sigma) else 0.
        read_var = (readnoise/gain)**2
        sky_var = max(empirical_var, read_var, 1e-12)
        sky_var_col[x] = sky_var
        # Derive a positive, normalized Gaussian spatial profile; smooth model
        # width can be adjusted in GUI without learning PSF from single pixels.
        p = np.exp(-0.5*((ap_y-cy)/profile_sigma)**2)
        p /= p.sum()
        profile_model[ap_index, x] = p
        ap_columns.append((values, sky_bg, p, sky_var, len(sy)))
    if mode == 'variance' and profile_smooth > 0:
        # Smooth Gaussian model along wavelength to avoid jitter in weights.
        profile_model = gaussian_filter1d(profile_model, profile_smooth, axis=1, mode='nearest')
    for x, info in enumerate(ap_columns):
        if info is None:
            continue
        values, background, p_original, sky_var, nsky = info
        cy = trace[x]
        c = int(np.rint(cy))
        all_y = c+y_offsets
        valid = (all_y >= 0) & (all_y < ny)
        ap_y = all_y[valid]
        valid = valid.copy()
        # Need the same non-finite-pixel mask as the first pass.
        mask = np.isfinite(data[ap_y, x]); idx = np.where(valid)[0][mask]
        if mode == 'sum':
            raw[x] = values.sum()
            sky[x] = background.sum()
            net[x] = raw[x]-sky[x]
            # Sky variance includes read noise and background Poisson component;
            # source Poisson adds only positive source counts.
            var = max(net[x], 0)/gain + len(values)*sky_var + len(values)**2*sky_var/nsky
            err[x] = np.sqrt(max(var, 0))
        else:
            p = np.maximum(profile_model[idx, x], 0)
            if p.sum() <= 0:
                continue
            p /= p.sum()
            residual = values-background
            # Approximate variance of each pixel in ADU^2. Include uncertainty
            # of sky-level estimation (single sky estimate from N sky pixels).
            variance = sky_var + np.maximum(residual, 0)/gain + sky_var/nsky
            good = np.isfinite(variance) & (variance > 0)
            for _ in range(3):
                if good.sum() < 2:
                    break
                den = np.sum(p[good]**2/variance[good])
                if den <= 0:
                    break
                f = np.sum(p[good]*residual[good]/variance[good])/den
                rejected = (np.abs(residual-f*p) > reject_sigma*np.sqrt(variance)) & good
                if not rejected.any() or good.sum()-rejected.sum() < 2:
                    break
                good[rejected] = False
            if good.sum() < 2:
                continue
            den = np.sum(p[good]**2/variance[good])
            weights = p[good]/variance[good]/den
            raw[x] = np.sum(weights*values[good])
            sky[x] = np.sum(weights*background[good])
            net[x] = raw[x]-sky[x]
            # Statistical per-pixel variance plus correlated sky-estimate term.
            err[x] = np.sqrt(1/den + sky_var/nsky * np.sum(weights)**2)
    return {'raw': raw, 'sky': sky, 'net': net, 'error': err, 'trace': trace}

def BoxcarExtract(img, trace_line, log_file=None, filename=None, airmass=None, apwidth=8,
                  skysep=7, skywidth=7, skydeg=0, Saxis=0, Waxis=None, method='sum',
                  display=False, ax=None, otype='src', gain=1., readnoise=6.,
                  profile_sigma=2.5, profile_smooth=4, return_arrays=False, **kwargs):
    """Compatibility: returns (raw Spectrum1D, sky Spectrum1D). No Matplotlib windows."""
    result = extract_spectrum(img, trace_line, apwidth, skysep, skywidth, skydeg,
                              method, Saxis, gain, readnoise, profile_sigma, profile_smooth, otype=otype)
    unit = getattr(img, 'unit', u.adu) or u.adu
    pixel = np.arange(len(result['raw']))*u.pixel
    spec = Spectrum1D(spectral_axis=pixel, flux=result['raw']*unit,
                      uncertainty=StdDevUncertainty(result['error']))
    skyspec = Spectrum1D(spectral_axis=pixel, flux=result['sky']*unit)
    if return_arrays:
        return spec, skyspec, result
    return spec, skyspec

def boxfunc(data, trace, filename=None, airmass=None, apwidth=8, skysep=10, skywidth=7,
            skydeg=0, Saxis=0, method='sum', display=False, ax=None, otype='src',
            return_arrays=False, **kwargs):
    return BoxcarExtract(data, trace, filename=filename, airmass=airmass, apwidth=apwidth,
                         skysep=skysep, skywidth=skywidth, skydeg=skydeg, Saxis=Saxis,
                         method=method, otype=otype, return_arrays=return_arrays, **kwargs)
