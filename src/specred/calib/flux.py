import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import UnivariateSpline
from astropy.constants import c as cc
import astropy.units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
import os
import pandas as pd



__all__ = ['mag2flux', 'obs_extinction', 'airmass_cor', 'stdstr',
           'standard_sensfunc', 'apply_sensfunc']


####################33

scpath =os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
## specify the path of the script to direct into the resource folder. Users can change the directory where the resource folder has been saved.  
####################
def close_ele(wave,lines):
    D=[]
    for i in lines[0]:
        d=abs(float(wave)-float(i))
        D.append(d)
        aa = pd.Series(D)
        idx = aa.idxmin()
    print(lines[0][idx],wave)
    return lines[0][idx],wave

def mag2flux(spec_in, zeropt=48.60):

    """
    Convert magnitudes to flux units. This is important for dealing with standards
    and files from IRAF, which are stored in AB mag units. To be clear, this converts
    to "PHOTFLAM" units in IRAF-speak. Assumes the common flux zeropoint used in IRAF

    Parameters
    ----------
    spec_in: a Spectrum1D object
            An input spectrum with wavelength of the data points in Angstroms
            as the ``spectral_axis`` and magnitudes of the data as the ``flux``.
    zeropt : float, optional
        Conversion factor for mag->flux. (Default is 48.60 from AB system)

    Returns
    -------
    Spectrum1D object with ``flux`` now in flux units (erg/s/cm2/A)
    """

    lamb = spec_in.spectral_axis
    mag = spec_in.flux.value

    # caution, getting a bit sloppy with units here, especially in wavelength...
    flux = (10.0 ** ((mag + zeropt) / (-2.5))) * (cc.to('AA/s').value / lamb.value ** 2.0)
    flux = flux * u.erg / u.s / u.angstrom / (u.cm * u.cm)

    # NOTE: we're ommiting the uncertainty here, since this function is
    # most likely used for converting standard ref spec, not observations
    spec_out = Spectrum1D(spectral_axis=lamb, flux=flux)

    return spec_out


def read_obs_extinction(obs_file):
    obs_file = os.path.join(scpath,"resources/extinction/",obs_file)
    print(obs_file)
    file = Table.read(obs_file,format="ascii",names=("wave","X"))
    file['wave'].unit= 'AA'
    return file    
    
    
def read_arclamp(arc_file):
    #file = pd.read_table(os.path.join(scpath,"resources/arc/",arc_file),sep="\t")
    arc_file = os.path.join(scpath,"resources/arc/",arc_file)
    print(arc_file)
    file = Table.read(arc_file,format="ascii",names=("wave","Ele"))
    wave = file['wave'].value
    element =file["Ele"].value
    return wave,element


def airmass_cor(object_spectrum, airmass, Xfile):
    """
    Correct the spectrum based on the airmass. Requires observatory extinction file

    Parameters
    ----------
    object_spectrum : Spectrum1D object
    airmass : float
        The value of the airmass. Note: NOT the header keyword.
    Xfile : astropy table
        The extinction table from `obs_extinction`, with columns ('wave', 'X')
        that have standard units of: (angstroms, mag/airmass)

    Returns
    -------
    The airmass-corrected Spectrum1D object

    """
    
    obj_wave, obj_flux = object_spectrum.wavelength, object_spectrum.flux

    # linear interpol airmass extinction onto observed wavelengths
    new_X = np.interp(obj_wave.value, Xfile['wave'], Xfile['X'])

    # air_cor in units of mag/airmass, convert to flux/airmass
    airmass_ext = 10.0**(0.4 * airmass * new_X)

    # output_spec = object_spectrum.multiply(object_spectrum.flux, airmass_ext)
    # output_spec.flux = obj_flux * airmass_ext
    # return output_spec
    return object_spectrum.multiply(airmass_ext * u.dimensionless_unscaled)


def onedstd(stdstar):
    """
    Load the one-dimensional standard star from the supplied library
    "onedstd", originally from IRAF. The provenance of these reference
    spectra are varied, and future work includes creating a uniform set.

    Parameters
    ----------
    stdstar : str
        Name of the standard star file in the kosmos/resources/onedstds
        directory to be used for the flux calibration. The user must
        provide the subdirectory and file name. For example:

        >>> standard_sensfunc(obj_spec, standard, stdstar='spec50cal/bd284211.dat', mode='spline')  \
        # doctest: +SKIP

        If no standard is supplied, or an improper path is given,
        will raise a ValueError.

    Returns
    -------
        astropy Table with onedstd data
    """

def stdstr(stdstar):
    std_dir = os.path.join(scpath,"resources",'onedstar')
    print("path to standard star mag files :",std_dir)
    '''
    S = str(input("the name of the standard star given is {w}, do you want to continue (type c) or rewrite the name (type r)".format(w=stdstar)))
    if S.lower() =="c":
        pass
    elif S.lower() =="r":
        name = str(input("enter the name of standard star :"))
        stdstar = name	
    '''	
    if not os.path.isfile(os.path.join(std_dir, stdstar)):
        msg2 = "No valid standard star found at: " + os.path.join(std_dir, stdstar)
        raise ValueError(msg2)

    # read the ASCII table in
    standard = Table.read(os.path.join(std_dir, stdstar), format='ascii',
                          names=('wave', 'mag', 'width'))

    # add standard units to everything
    standard['wave'].unit = u.angstrom
    standard['width'].unit = u.angstrom
    standard['mag'].unit = u.mag

    # standard star spectrum is stored in magnitude units (IRAF conventions)
    # convert to flux
    std_flux_spec = mag2flux(Spectrum1D(flux=standard['mag'].quantity,
                                   spectral_axis=standard['wave'].quantity))
    std_wave = std_flux_spec.wavelength.value
    std_flux = std_flux_spec.flux.value
   
    # add column with flux units to the Table
    Stdflux = std_flux_spec.flux*u.Jy
    print(Stdflux)
    standard.add_column(Stdflux, name='flux')
    
    
    plt.plot(std_wave,std_flux)
    plt.show()
    return standard



def standard_sensfunc(object_spectrum, standard, mode='spline', polydeg=9,
                      badlines=None, display=False, ax=None):
    """
    Compute the standard star sensitivity function. First down-samples the
    observed standard star spectrum to the reference spectrum, then computes
    log_10(Reference Flux / Observed Flux). This log sensfunc is then
    interpolated using the specified `mode` back to the entire observed
    wavelength range, and the normal (i.e. not log10) sensfunc is returned.

    Parameters
    ----------
    object_spectrum : Spectrum1D object
        The observed standard star spectrum
    standard : astropy table
        output from `onedstd`, has columns ('wave', 'width', 'mag', 'flux')
    mode : str, optional {'linear', 'spline', 'poly', 'interp'}
        (Default is spline)
    polydeg : float, optional
        if mode='poly', this is the order of the polynomial to fit through
        (Default is 9)
    display : bool, optional
        If True, plot the sensfunc (Default is False)
    badlines : array-like, optional
        A list of values (lines) to mask out of when generating sensfunc
    ax : matplotlib axes or subplot object, optional
        axes or subplot to be plotted onto. If not specified one will be 
        created. (Default is None)

    Returns
    -------
    sensfunc_spec : Spectrum1D object
            The sensitivity function in the covered wavelength range
            for the given standard star, stored as a Spectrum1D

    """

    obj_wave, obj_flux = object_spectrum.wavelength, object_spectrum.flux

    #plt.plot(standard["wave"]
    # Automatically exclude some lines b/c resolution dependent response
    if badlines is None:
        badlines = [4341, 4861, 6563]
    badlines = np.array(badlines, dtype='float')  # Balmer lines

    # down-sample (ds) the observed flux to the standard's bins
    # IMPROVEMENT: could this be done w/ `specutils.manipulation.FluxConservingResampler`?
    obj_flux_ds = np.array([], dtype=float)
    obj_wave_ds = np.array([], dtype=float)
    std_flux_ds = np.array([], dtype=float)
    for i in range(len(standard['flux'])):
        # IMPROVEMENT: this could be done faster/better w/o using np.where...
        rng = np.where((obj_wave.value >= standard['wave'][i] - standard['width'][i] / 2.0) &
                       (obj_wave.value < standard['wave'][i] + standard['width'][i] / 2.0))[0]

        IsH = np.where((badlines >= standard['wave'][i] - standard['width'][i] / 2.0) &
                       (badlines < standard['wave'][i] + standard['width'][i] / 2.0))[0]

        # does this bin contain observed spectra, and e.g. no Balmer lines?
        #print(f"rng:{ len(rng)} | Ish {len(IsH)}")
        if (len(rng) > 1) and (len(IsH) == 0):
            obj_flux_ds = np.append(obj_flux_ds, np.nanmean(obj_flux.value[rng]))
            obj_wave_ds = np.append(obj_wave_ds, standard['wave'][i])
            std_flux_ds = np.append(std_flux_ds, standard['flux'][i])

    # the ratio between the standard star catalog flux and observed flux
    ratio = np.abs(std_flux_ds / obj_flux_ds)

    # Use the log of this sensfunc ratio, since IRAF does everything in mag units
    LogSensfunc = np.log10(ratio)
    print(f" before applying the sensfunc \n, objwave {obj_wave.value} | \n ds {obj_wave_ds} \n | {LogSensfunc}") 
    # if invalid interpolation mode selected, make it spline
    if mode.lower() not in ('linear', 'spline', 'poly'):
        mode = 'spline'
        import warnings
        warnings.warn("WARNING: invalid mode set. Changing to default mode 'spline'")

    # interpolate the calibration (sensfunc) on to observed wavelength grid
    if mode.lower() == 'linear':
        sensfunc2 = np.interp(obj_wave.value, obj_wave_ds, LogSensfunc)
    elif mode.lower() == 'spline':
        spl = UnivariateSpline(obj_wave_ds, LogSensfunc, ext=0, k=3 ,s=0.0025)
        sensfunc2 = spl(obj_wave.value)
        print("k=3")
    elif mode.lower() == 'poly':
        fit = np.polyfit(obj_wave_ds, LogSensfunc, polydeg)
        sensfunc2 = np.polyval(fit, obj_wave.value)

    sensfunc_out = (10 ** sensfunc2) * (standard['flux'].unit / obj_flux.unit)
    sensfunc_spec = Spectrum1D(spectral_axis=obj_wave, flux=sensfunc_out)
    print("obj_wave =",obj_wave)
    bin2 = np.where((obj_wave.value >3000) & (obj_wave.value < 9000))
    print("bin2 = ",bin2,bin2[0])
    
    if display is True:
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(obj_wave[bin2], obj_flux[bin2] * sensfunc_out[bin2], c='C0',
                    label='Observed x sensfunc', alpha=0.5)
        ax.scatter(obj_wave_ds, std_flux_ds, color='C1', alpha=0.75,label="observed")

        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')

        ax.set_xlim(np.nanmin(obj_wave.value), np.nanmax(obj_wave.value))
        ax.set_ylim(np.nanmin(obj_flux.value * sensfunc_out.value)*0.98,
                 np.nanmax(obj_flux.value * sensfunc_out.value) * 1.02)
        plt.legend()
        plt.show()

    return sensfunc_spec


def apply_sensfunc(object_spectrum, sensfunc_spec):
    """
    Apply the derived sensitivity function, converts observed units (e.g. ADU/s)
    to physical units (e.g. erg/s/cm2/A).

    Sensitivity function is first linearly interpolated onto the wavelength scale
    of the observed data, and then directly multiplied.

    Parameters
    ----------
    object_spectrum : Spectrum1D object
        the observed object spectrum to apply the sensfunc to
    sensfunc_spec : Spectrum1D object
        the output of `standard_sensfunc`
    Returns
    -------
    The sensfunc-corrected spectrum, a Spectrum1D object
    """

    obj_wave, obj_flux = object_spectrum.wavelength, object_spectrum.flux

    # sort, in case the sensfunc wavelength axis is backwards, interp can get angry
    srt = np.argsort(sensfunc_spec.wavelength.value)

    # interpolate the sensfunc onto the observed wavelength axis, in case they don't match
    # IMPROVEMENT: could this be done w/ `specutils.manipulation.FluxConservingResampler`?
    sensfunc2 = np.interp(obj_wave.value, sensfunc_spec.wavelength.value[srt], sensfunc_spec.flux.value[srt])

    # multiply the observed object spectrum by the interpolated sensitivity function
    # *should* update units properly, for both flux and uncertainty
    fluxcal_spec = object_spectrum.multiply(sensfunc2 * sensfunc_spec.flux.unit)
    return fluxcal_spec


######################################################################################
# The three functions below were originally embedded directly as GUI methods
# (_load_calibrated_spectrum, _calibrate_flux, _save_flux_calibrated) inside
# gui/main_window.py. They are pure I/O + calibration logic with no dependency
# on Qt, so they've been moved here to keep the GUI a thin front-end and to
# make flux calibration scriptable/testable without launching the GUI.
######################################################################################

def load_calibrated_spectrum(filename):
    """
    Load a wavelength-calibrated spectrum from a FITS file written by the
    wavelength-calibration stage (expects 'WAVELENGTH' and optionally
    'UNCERTAINTY' extensions alongside the primary flux array).

    Returns
    -------
    Spectrum1D
    """
    with fits.open(filename) as hdul:
        flux = hdul[0].data
        wave = hdul['WAVELENGTH'].data
        uncertainty = hdul['UNCERTAINTY'].data if 'UNCERTAINTY' in hdul else None

    return Spectrum1D(
        flux=flux * u.Unit('count'),
        spectral_axis=wave * u.AA,
        uncertainty=StdDevUncertainty(uncertainty) if uncertainty is not None else None
    )


def calibrate_flux(sci_spec, std_spec, extinction_file, sci_airmass, std_airmass,
                    std_file, sens_mode='spline', display=False):
    """
    Perform flux calibration for a single science/standard pair:
    airmass-correct both spectra, build a sensitivity function from the
    standard, and apply it to the science spectrum.

    Parameters
    ----------
    sci_spec, std_spec : Spectrum1D
        Wavelength-calibrated science and standard-star spectra
        (see load_calibrated_spectrum).
    extinction_file : str
        Path to the observatory extinction curve (read_obs_extinction).
    sci_airmass, std_airmass : float
        Airmass at the time of the science/standard exposures.
    std_file : str
        Name/path identifying the standard star's reference flux table
        (passed to stdstr()).
    sens_mode : str
        Fitting mode passed through to standard_sensfunc.
    display : bool
        Whether standard_sensfunc should plot diagnostics.

    Returns
    -------
    Spectrum1D
        The flux-calibrated science spectrum.
    """
    if not os.path.exists(extinction_file):
        raise FileNotFoundError(f"Extinction file not found: {extinction_file}")

    Xfile = read_obs_extinction(extinction_file)

    CAL = airmass_cor(std_spec, std_airmass, Xfile)
    SCI = airmass_cor(sci_spec, sci_airmass, Xfile)

    if not std_file:
        raise ValueError("No standard star file selected")

    star = stdstr(std_file)

    sens_func = standard_sensfunc(CAL, star, mode=sens_mode, display=display)

    return apply_sensfunc(SCI, sens_func)


def save_flux_calibrated(spectrum, filename, calsrc=None, calmode=None):
    """
    Save a flux-calibrated Spectrum1D to a FITS file, writing flux in the
    primary HDU (BUNIT erg/s/cm^2/Angstrom) plus WAVELENGTH and, if present,
    UNCERTAINTY extensions.

    Parameters
    ----------
    spectrum : Spectrum1D
    filename : str
        Output path.
    calsrc : str, optional
        Recorded in the CALSRC header keyword (e.g. the standard-star file used).
    calmode : str, optional
        Recorded in the CALMODE header keyword (e.g. the sensfunc fit mode used).
    """
    primary_hdu = fits.PrimaryHDU(spectrum.flux.value)
    primary_hdu.header['BUNIT'] = 'erg/s/cm^2/Angstrom'

    hdus = [primary_hdu]

    if spectrum.uncertainty is not None:
        uncert_hdu = fits.ImageHDU(spectrum.uncertainty.array, name='UNCERTAINTY')
        hdus.append(uncert_hdu)

    wave_hdu = fits.ImageHDU(spectrum.spectral_axis.value, name='WAVELENGTH')
    wave_hdu.header['BUNIT'] = 'Angstrom'
    hdus.append(wave_hdu)

    if calsrc is not None:
        primary_hdu.header['CALSRC'] = calsrc
    if calmode is not None:
        primary_hdu.header['CALMODE'] = calmode
    primary_hdu.header['CRVAL1'] = spectrum.spectral_axis[0].value
    primary_hdu.header['CDELT1'] = np.mean(np.diff(spectrum.spectral_axis.value))

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(filename, overwrite=True)
