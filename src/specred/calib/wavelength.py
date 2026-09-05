import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, interp1d
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler, gaussian_smooth
from specutils.utils.wcs_utils import air_to_vac as a2v
import os
import pandas as pd
from specutils.fitting import find_lines_derivative
from scipy.signal import find_peaks, peak_widths,savgol_filter
from astropy.table import Table
from specred.calib import flux as fluxcal
from specred import utils as add
from astropy import units as u
from scipy.interpolate import CubicSpline
import matplotlib.gridspec as gridspec

from sklearn.linear_model import RANSACRegressor, HuberRegressor
from numpy.polynomial import Chebyshev

# NOTE (bug fix): this used to be os.path.dirname(os.path.abspath(__name__)),
# which resolves the *module name string* against the current working
# directory rather than this file's actual location -- so `scpath` (used
# below for resources/arc/*.dat lookups) silently pointed at whatever
# directory the process happened to be launched from. This is the same
# bug already fixed in specred/utils.py; it was missed here in the first
# pass because the two files weren't grepped for the exact same pattern
# together. Fixed to use __file__ -- and note this file lives one level
# deeper (specred/calib/) than utils.py (specred/), so it needs the extra
# dirname() to land on the same specred/resources/ folder (matches the
# pattern already used correctly in specred/calib/flux.py). The leftover
# debug print was also removed.
scpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _gaus(x, a, b, x0, sigma):
    """
    Define a simple Gaussian curve
    Could maybe be swapped out for astropy.modeling.models.Gaussian1D
    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian
    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def Find_peaks(wave, flux, pwidth=10, pthreshold=0.97, minsep=1):
    """
    Given a slice thru an arclamp image, find the significant peaks.
    Originally from PyDIS
    Parameters
    ----------
    wave : `~numpy.ndarray`
        Wavelength (could be approximate)
    flux : `~numpy.ndarray`
        Flux
    pwidth : float (default=10)
        the number of pixels around the "peak" to fit over
    pthreshold : float (default = 0.97)
        Peak threshold, between 0 and 1
    minsep : float (default=1)
        Minimum separation
    Returns
    -------
    Peak Pixels, Peak Wavelengths
    """
    # sort data, cut top x% of flux data as peak threshold
    flux_thresh = np.percentile(flux, pthreshold*100)

    # find flux above threshold
    high = np.where((flux >= flux_thresh))[0]

    # find  individual peaks (separated by > 1 pixel)
    # this is horribly ugly code... but i think works
    pk = high[1:][((high[1:]-high[:-1]) > minsep)]

    # offset from start/end of array by at least same # of pixels
    pk = pk[pk > pwidth]
    pk = pk[pk < (len(flux) - pwidth)]

    pcent_pix = np.zeros_like(pk, dtype='float')
    wcent_pix = np.zeros_like(pk, dtype='float')

    # for each peak, fit a gaussian to find center
    for i in range(len(pk)):
        xi = wave[pk[i] - pwidth:pk[i] + pwidth]
        yi = flux[pk[i] - pwidth:pk[i] + pwidth]

        pguess = (np.nanmax(yi), np.nanmedian(flux), float(np.nanargmax(yi)), 2.)
        try:
            popt, pcov = curve_fit(_gaus, np.arange(len(xi), dtype='float'),
                                   yi, p0=pguess)

            # the gaussian center of the line in pixel units
            pcent_pix[i] = (pk[i]-pwidth) + popt[2]
            # and the peak in wavelength units
            wcent_pix[i] = xi[np.nanargmax(yi)]

        except RuntimeError:
            pcent_pix[i] = float('nan')
            wcent_pix[i] = float('nan')

    wcent_pix, ss = np.unique(wcent_pix, return_index=True)
    pcent_pix = pcent_pix[ss]
    okcent = np.where((np.isfinite(pcent_pix)))[0]
    return pcent_pix[okcent], wcent_pix[okcent]


def loadlinelist(file):
    df = Table.read(file,format="ascii",names=("wave","line"))
    arc = df['wave'].value
    #print("ARC=",arc)
    return arc
    
    
    
    
    
def identify_wavelength(observed_spectrum,arc_file,height=None,prominence =0.0,distance=5,nopks = 5):
    """
    Paramters :
    -----------
    observed_spectrum : 1D-array , signal containing peaks
    
    height : threshold value for height above which the peaks will be taken for line estimation
    
    nopks : if height is Nome, the integer number for lines to be considered for line estimation
    
    
    Return :
    -----------
    Two 1D-array containing pixel value of the peak and corresponding wavelenght value 
   
    
    """
    lines = fluxcal.read_arclamp(arc_file)
    peaks=[]
    spec = observed_spectrum
    if height is not None:
        print("1")
        peaks, _ = find_peaks(observed_spectrum,height=height, prominence=prominence, distance=distance)
        print(len(peaks))
    else:
        print("2")
        pks,_ = find_peaks(spec,height=np.mean(spec)*4,prominence=prominence, distance=distance)
        print(len(pks))
        a = sorted(spec[pks],reverse=True)
        print(a)
        if len(pks) < nopks:
            nopks = len(pks)
        else:
            pass
        height = a[nopks-1]
        print("peaks = ",pks,"len = ",len(pks),"second max = ",pks[nopks-1],"height = ",height)
        print("Value at peaks =",a)
        peaks, _ = find_peaks(observed_spectrum,height=height, prominence=prominence, distance=distance)
        print("len = ",len(peaks))
        
    fig,ax1 = plt.subplots(figsize=(10, 4), nrows=1, sharex=True)
    ax1.plot(observed_spectrum)
    ax1.axhline(height,color="r",ls="dashed",alpha=0.5)
    ax1.axhline(np.mean(spec)*4,color="k",ls="dashed",alpha=0.5)

    for i in peaks:
        ax1.axvline(i,color="k",ls="dashed",alpha=0.5)
        
    plt.show()
    
    
    WV=[]
    PIX=[]
    for i in peaks:
        fig2,ax2 = plt.subplots(figsize=(10, 4), nrows=1, sharex=True)

        ax2.plot(observed_spectrum)
        ax2.axvline(i,color="k",alpha=0.5)
        plt.show()
        pk = i
        wv = input("enter the corresponding wavelength value for peak- %d :"%i)
        if wv != str():
            PIX.append(pk)
            wv = float(wv)
            vw = fluxcal.close_ele(wv,lines)[0]
            print(" the wavelenght {} {}".format(wv,vw))
            WV.append(vw)
        else:
       	 pass
    return (PIX,WV)
        
        
        
def fit_wavelength(spec,arcspec, xpoints, wpoints,arc_file,z_redshift=0,offset=None,disp=None,
                   display=False,mode='spline', deg=7, GPRscale=101,
                   returnpoints=False, returnvar=False,nopks=15):
    """
    # adding the arcspec for the reidentification of lines and then applying that solution to
    # the science frames.
    Fit the wavelength solution from a series of (pixel, Wavelength)
    datapoints, and apply it a spectrum
    Parameters
    ----------
    spec : Spectrum1D
        the object spectrum to have a new wavelength axis added
    xpoints : array-like object
        the pixel values of identified arcline features
    wpoints : astropy Quantity
        the corresponding wavelengthsfed pixels.
        NOTE: Must have sensible units like angstroms, which will be
        applied to the resulting spectrum.
    display : bool, optional (default is False)
        should we plot the (pixel,wavelength) fit residuals?
    mode : str, ['poly', 'spline', 'interp', 'gp']
        which fitting mode should be used? (Default is 'poly')
        Select between Polynomial, UnivariateSpline, Interpolation, and
        a Gaussian Process (via `george`, using ExpSquaredKernel)
    deg : int, optional (default is 7)
        if mode='poly', set the polynomial degree to use
        if mode='interp', set the interpolation degree (passed as
        `kind=deg` to `interp1d()`).
    GPRscale : int, optional (default is 101)
        If mode='gp', the Rscale parameter to use with ExpSquaredKernel
    returnpoints : bool, optional (default is False)
        If set, return just the fit values corresponding to the input
        (xpoints, wpoints)
    returnvar : bool, optional (default is False)
        If set and mode='gp', additionally return the variance on the
        resulting wavelength axis
    Returns
    -------
    outspec : Sepctrum1D object
        the same input spectrum, but with the newly fit wavelength
        axis added.
    if returnvar=True, then return:
        outspec, wavelength_variance
    """

    # Improvements Needed
    # ------------
    # should the fit and apply steps be separated?

    # sort, just in case
    srt = np.argsort(xpoints)
    xpt = np.array(xpoints)[srt]
    wpt = np.array(wpoints.value)[srt]
    Flux= np.array(spec.flux.value[::-1])[srt]
    fpt = np.zeros_like(xpt)  # the fit wavelength points
    
    lines,ele = fluxcal.read_arclamp(arc_file)
    
    #if mode.lower() == 'xfit':
        #XPT = xpt
    if mode.lower() == 'poly':
        fit = np.polyfit(xpt, wpt, deg=1)
        wavesolved = np.polyval(fit, spec.spectral_axis.value)
        fpt = np.polyval(fit, xpt)

    if mode.lower() == 'spline':
        spl = UnivariateSpline(xpt, wpt, ext=0, k=3, s=1e3)
        
        wavesolved = spl(spec.spectral_axis.value)
        waveset    = spl(lines)
        fpt = spl(xpt)
    
        

    if mode.lower() == 'interp':
        spl = interp1d(xpt, wpt, kind=deg, fill_value='extrapolate')
        wavesolved = spl(spec.spectral_axis.value)
        fpt = spl(xpt)

    if mode.lower() =="jin":
        if offset is None:
            offset = float(input("enter the offset value "))

        if disp is None:
            disp = float(input("enter the dispersion value "))
        
        plt.plot(Flux)
        plt.title(" FLUX in jin")
        plt.show()   
        popt = add.Wavesolution(xpt,Flux,arc_file,offset,disp)
        wavesolved = add.XFIT(xpt,*popt)
        spec = Spectrum1D(flux = spec.flux[::-1])
        
    if mode.lower() =="linefit":
        #xpt = add.XFIT(xpt,offset,disp)
        arcFlux = arcspec.flux.value[::-1]
        Flux = spec.flux.value[::-1]
        #wave = spec.spectral_axis.value
        '''
        plt.plot(arcFlux)
        plt.title("before linefit - ARC ")
        plt.show()
        plt.plot(Flux)
        plt.title("Spec Flux")
        plt.show()
        '''
        A,B,fit = add.linefit(wpt,arcFlux,sciflux=Flux,file=arc_file,nopks=nopks)
        wavesolved=A
        Flux=B
        print("wavesolved=",wavesolved)
        #spec = Spectrum1D(flux = spec.flux[::-1])
    if mode.lower() == 'gp':
        # assume 1/2 pixel precision of centering arc lines (prob better actually)
        yerr = np.ones_like(xpt) * np.mean(np.abs(np.diff(spec.spectral_axis.value))) / 2

        # follow BASIC tutorial from "george"
        # https://george.readthedocs.io/en/latest/tutorials/first/
        import george
        from george import kernels
        from scipy.optimize import minimize

        # Rscale = 100 # the magic scale param... hopefully OK, YMMV
        kernel = np.var(wpt) * kernels.ExpSquaredKernel(GPRscale)
        gp = george.GP(kernel, fit_mean=True)
        gp.compute(xpt, yerr)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(wpt)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(wpt)

        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, method='L-BFGS-B')
        # print(result)
        gp.set_parameter_vector(result.x)

        wavesolved, wavesolved_var = gp.predict(wpt, spec.spectral_axis.value, return_var=True)
        fpt = gp.predict(wpt, xpt, return_var=False)
        
       
        wavesolved = wavesolved#/(1+z_redshift)

    if display:
        plt.plot(spec.spectral_axis.value,wavesolved)
        plt.scatter(xpt,fpt,color='k')
        plt.xlabel('Xpoints')
        plt.ylabel('New WAVE')
        #plt.show()
        
        #plt.plot(spec.spectral_axis.value,wavesolved,"o-")
        #plt.xlabel('PIXELS')
        #plt.ylabel('WAVE')
        #plt.show()
        '''
        plt.scatter(xpt, wpt - fpt)
        plt.xlabel('Xpoints')
        plt.ylabel('Residuals')
        plt.show()
        
        plt.scatter(len(waveset),lines-waveset)
        plt.xlabel('lines - model(lines)')
        plt.ylabel('Residuals')
        plt.show()
        '''
    if mode.lower() !="linefit":
        outspec = Spectrum1D(spectral_axis=wavesolved * wpoints.unit,
                             flux=spec.flux,
                             uncertainty=spec.uncertainty
                             )
                         
    if returnpoints:
        return fpt

    if returnvar is True and mode.lower() == 'gp':
        # since there's no way to package uncertainty/variance w/ the Spectrum1D object currently
        return outspec, wavesolved_var
    
    if mode.lower()=="linefit":
        outspec = Spectrum1D(spectral_axis=wavesolved * wpoints.unit,flux=Flux*u.adu)
                         
        return outspec
        
        
    return outspec


def air_to_vac(spec):
    """
    Simple wrapper for the `air_to_vac` calculation within `specutils.utils.wcs_utils`
    Parameters
    ----------
    spec : Spectrum1D object
    Returns
    -------
    Spectrum1D object with spectral_axis converted from air to vaccum units
    """
    new_wave = a2v(spec.wavelength)
    outspec = Spectrum1D(spectral_axis=new_wave,
                         flux=spec.flux,
                         uncertainty=spec.uncertainty
                         )
    return outspec



def identify_nearest2(arcspec, wapprox=None, linelist=None, linewave=None,
                     autotol=25, silent=False):

    """
    Identify arc lines using a simple greedy "nearest neighbor" approach.
    Requires an approximate wavelength solution (e.g. as provided by
    image header keywords). Peaks are first detected in the 1d spectrum.
    Starting from the center of the spectrum, the closest lines within a
    tolerance are picked. A linear interpolation solution is iteratively
    fit with each successive line added.

    Parameters
    ----------
    arcspec : Spectrum1D
        the 1d spectrum of the arc lamp to be fit.
    wapprox : astropy Quantity, or None
        the approximate wavelenth solution, as e.g. provided by the
        image header. Must have sensible units, like Angstroms.
        NOTE: If set to None, assumes the `arcspec` object has the
        approximate wavelength axis.
    linelist : str, optional
        name of linelist to load, is passed to `loadlinelist()`
    linewave : numpy array or None, optional
        Optionally pass an array of arclines to fit, as returned by e.g.
        `loadlinelist()`
    autotol : int, optional (default is 25)
        the tolerance in pixel units to allow nearest matches within.
    silent : bool, optional (default is False)
        suppress a few helpful summary messages

    Returns
    -------
    xpoints, wpoints : the pixel and wavelength values of the
        successfully identified lines.
    """

    if linelist is not None:
        linelist = os.path.join(scpath, "resources/arc/", linelist)
        linewave = loadlinelist(linelist)

    if linewave is None:
        msg_fail = print('linewave must be an array of known line wavelengths.')
        raise ValueError(msg_fail)

    # the fluxes within the arc-spectrum
    flux = arcspec.flux.value
    #print("flux = ",flux)
    if wapprox is not None:
        xpixels = wapprox
    else:
        xpixels = arcspec.spectral_axis
    print("xpixels=",xpixels)        
    # in this mode, the xpixel input array is actually the approximate
    # wavelength solution (e.g. from the header info)
    pcent_pix, wcent_pix = Find_peaks(xpixels, flux, pwidth=5,
                                     pthreshold=0.87)
    print("find peaks =",pcent_pix,wcent_pix)
    Ppix = []
    Wpix =[]
    Ppix.append(pcent_pix)
    Wpix.append(wcent_pix)
    # A simple, greedy, line-finding solution.
    # Loop thru each detected peak, from center outwards. Find nearest
    # known list line. If no known line within tolerance, skip

    # PLAN: predict solution w/ spline, start in middle, identify nearest match,
    # every time there's a new match, recalc the spline sol'n, work all the way out
    # this both identifies lines, and has byproduct of ending w/ a spline model

    xpoints = np.array([], dtype=float)  # pixel line centers
    wpoints = np.array([], dtype=float)  # wavelength line centers

    # find center-most lines, sort by dist from center pixels
    ss = np.argsort(np.abs(wcent_pix - np.nanmedian(xpixels)))
    print("SS = ",ss)
    # 1st guess is the peak locations in the wavelength units as given by user
    wcent_guess = wcent_pix

    for i in range(len(pcent_pix)):
        # if there is a match within the tolerance
        if np.nanmin(np.abs(wcent_guess[ss][i] - linewave)) < autotol:
            # add corresponding pixel and known wavelength to output vectors
            xpoints = np.append(xpoints, pcent_pix[ss[i]])
            wpoints = np.append(wpoints, linewave[np.nanargmin(np.abs(wcent_guess[ss[i]] - linewave))])

            # start guessing new wavelength model after first few lines identified
            if len(wpoints) > 4:
                xps = np.argsort(xpoints)
                # spl = UnivariateSpline(xpoints[xps], wpoints[xps], ext=0, k=3, s=1e3)
                # wcent_guess = spl(pcent_pix)
                spl = interp1d(xpoints[xps], wpoints[xps], kind=1, fill_value='extrapolate')
                wcent_guess = spl(pcent_pix)

    inrng = sum((linewave >= np.nanmin(wcent_guess)) & (linewave <= np.nanmax(wcent_guess)))
    if not silent:
        print(str(len(wpoints)) + ' lines matched from ' + str(inrng) +' within estimated range.')

    # at this point we have (xpoints, wpoints), so next is generic interpolation.
    #       should this be part of another routine, used by all identify modes?

    # sort the points, just in case the method (or prev run) returns in weird order
    srt = np.argsort(xpoints)
    xpoints = xpoints[srt]
    wpoints = wpoints[srt]
    
    plt.plot(arcspec.flux.value,color="blue")
    plt.xlabel("PIXEL",fontsize=12)
    plt.suptitle("Wavelength Calibration",fontsize=12)
    for i,w in zip(xpoints,wpoints):
    	plt.axvline(x=i,color="r",ls="dashed",alpha=0.5)
    	plt.text(x=i,y=9e5,s=w,rotation=90)
    plt.show()

    return xpoints, wpoints,Ppix,Wpix




def fit_arclines1(wavespec,fluxspec,arcline,autotol=10,pthreshold=0.94,deg="cubic",display=True):

    linelist = os.path.join(scpath, "resources/arc/",arcline)
    linewave = loadlinelist(linelist)

    
    xpixels = wavespec
    flux = fluxspec
    sci_xp = np.arange(0,len(flux))  # start the array from 0 bcoz the extrapolation starts from value 0.
    pcent_pix, wcent_pix = Find_peaks(xpixels, flux, pwidth=12,pthreshold=pthreshold,minsep=10)

    print(f"no. of lines taken {len(pcent_pix)}")

    xpoints = np.array([], dtype=float)  # pixel line centers
    wpoints = np.array([], dtype=float)
    wcent_guess = wcent_pix
    autotol = autotol
    silent =False
    for i in range(len(pcent_pix)):
            # if there is a match within the tolerance
            if np.nanmin(np.abs(wcent_guess[i] - linewave)) < autotol:
                # add corresponding pixel and known wavelength to output vectors
                xpoints = np.append(xpoints,np.round(pcent_pix[i],4))
                wpoints = np.append(wpoints, linewave[np.nanargmin(np.abs(wcent_guess[i]-linewave))])
                #print(f" line: {linewave[np.nanargmin(np.abs(wcent_guess[i]-linewave))]} | guessed {wcent_guess[i]}")
                # start guessing new wavelength model after first few lines identified
                '''
                if len(wpoints) > int(len(pcent_pix)*0.4):
                    xps = np.argsort(xpoints)
                    #spl = UnivariateSpline(xpoints[xps], wpoints[xps], ext=0, k=3, s=len(pcent_pix)*100)
                    #wcent_guess = spl(pcent_pix)
                    
                    spl = interp1d(xpoints[xps], wpoints[xps], kind=1, fill_value='extrapolate')
                    wcent_guess = spl(pcent_pix)
                    wcent_guess1 = spl(xpoints)
                '''

    inrng = sum((linewave >= np.nanmin(wcent_guess)) & (linewave <= np.nanmax(wcent_guess)))
    if not silent:
        print(str(len(wpoints)) + ' lines matched from ' + str(inrng) +' within estimated range.')

    # at this point we have (xpoints, wpoints), so next is generic interpolation.
    #       should this be part of another routine, used by all identify modes?

    # sort the points, just in case the method (or prev run) returns in weird order
    srt = np.argsort(xpoints)
    xpoints = xpoints[srt]
    wpoints = wpoints[srt]
    #'''
    
    '''
    plt.scatter(wpoints,wpoints-wcent_guess1)
    plt.xlabel("ref wave")
    plt.ylabel("ref - estimated wavelength")
    plt.show()
    '''
    if deg=="linear":  # trying alternative for the wavelength solution
        fit = np.polyfit(xpoints, wpoints, deg=1)
        wavesolved = np.polyval(fit, sci_xp)
    else:
        spl = interp1d(xpoints, wpoints, kind=deg, fill_value='extrapolate')   
        wavesolved = spl(sci_xp)

    pcent_pix1, wcent_pix1 = Find_peaks(wavesolved, flux, pwidth=10,pthreshold=pthreshold,minsep=5)
    A,B,C =add.closeval2(wcent_pix1,linewave,5)
    B = np.array(B)
    C = np.array(C)
    mean_obs = np.mean(B)
    SS_res = (B - C) ** 2
    SS_tot = (B - mean_obs) ** 2

    # Calculate R-squared
    R_squared = 1 - (np.sum(SS_res) / np.sum(SS_tot))
    print("R-squared:", R_squared)

    #plt.scatter(B,B-C)
    #plt.title("After wavelength solution")
    #plt.show()
    if display:
        fig,ax = plt.subplots(2,figsize=(20,10))
        ax[0].plot(flux)
        for i,w in zip(xpoints,wpoints):
            ax[0].axvline(x=i,color="r",ls="dashed",alpha=0.5)
            ax[0].text(x=i,y = int(np.max(flux)-np.std(flux)),s=w,rotation=90)
            ax[0].set_title("Lines identified to closest value in ref. table")
            ax[0].set_xlabel("pixel")
            ax[0].set_ylabel("counts")
        
        ax[1].plot(wavesolved,flux)
        for i in wcent_pix1:
            ax[1].axvline(i,color="r",ls="dashed",alpha=0.5)
            ax[1].text(x=i,y=np.max(flux)-np.std(flux),s=np.round(i,3),rotation=90)
        for ii in C:
            ax[1].axvline(ii,color="k",ls="dotted",alpha=0.8)

        ax[1].set_title("wavelength solved spectra")
        ax[1].set_xlabel(r"wavelength $\AA$")

        plt.tight_layout()
        plt.show()

    return wavesolved


############################################################################
## new try 17072k25

def normalize_pixels(pixels, xmin, xmax):
    return (2 * pixels - (xmax + xmin)) / (xmax - xmin)
def compute_z(n, order):
    z = np.zeros((len(n), order))
    z[:, 0] = 1  # z_1 = 1
    if order >= 2:
        z[:, 1] = n  # z_2 = n
    for i in range(3, order + 1):
        z[:, i-1] = ((2*i - 3) * n * z[:, i-2] - (i - 2) * z[:, i-3]) / (i - 1)
    return z

def legnedre_sol(Xpxl,Wpnts,pix_array,order):
    """ 
    Parameters:  
    Xpxl :  array of the pixel centriod of the spectral line
    Wpnts : array of the corresponding wavelength value of Xpxl
    pix_array: array of the pixel for which the solution is to find (here Xpxl will be subset of pix_array)
    order: order of the polynomial function

    Returns: 
    The wavelength solution for the pix_array

    """
    pixels = Xpxl
    wavelengths = Wpnts

    ## normalizing the array
    x_min, x_max = pix_array.min(), pix_array.max()
    n = (2 * pixels - (x_max + x_min)) / (x_max - x_min)

    z = compute_z(n, order)

    # Solve for coefficients c
    A = z  # Design matrix
    c, residuals, rank, singular_values = np.linalg.lstsq(A, wavelengths, rcond=None)

    nccd = normalize_pixels(pix_array,pix_array.min(),pix_array.max())

    zccd = compute_z(nccd,order)

    return zccd@c


def fit_arclines_old(wavespec,fluxspec,arcline,autotol=10,pthreshold=0.94,func="linear",deg=1,saveplot=False,plotoutdir=None):
    wave_lw= 3500
    wave_uw = 9000

    linelist = os.path.join(scpath, "resources/arc/",arcline)
    linewave = loadlinelist(linelist)

    xpixels = wavespec
    flux = fluxspec
    sci_xp = np.arange(len(flux))
    pcent_pix, wcent_pix = Find_peaks(xpixels, flux, pwidth=10,pthreshold=pthreshold,minsep=8)

    print(f"no. of lines taken {len(pcent_pix)}")

    xpoints = np.array([], dtype=float)  # pixel line centers
    wpoints = np.array([], dtype=float)
    wcent_guess = wcent_pix
    autotol = autotol
    silent =False
    for i in range(len(pcent_pix)):
            # if there is a match within the tolerance
            if np.nanmin(np.abs(wcent_guess[i] - linewave)) < autotol:
                # add corresponding pixel and known wavelength to output vectors
                xpoints = np.append(xpoints,np.round(pcent_pix[i],4))
                wpoints = np.append(wpoints, linewave[np.nanargmin(np.abs(wcent_guess[i]-linewave))])
                #print(f" line: {linewave[np.nanargmin(np.abs(wcent_guess[i]-linewave))]} | guessed {wcent_guess[i]}")
                # start guessing new wavelength model after first few lines identified
                if len(wpoints) > 20:
                    xps = np.argsort(xpoints)
                    spl = UnivariateSpline(xpoints[xps], wpoints[xps], ext=0, k=3, s=len(pcent_pix)*10)
                    wcent_guess = spl(pcent_pix)
        

    inrng = sum((linewave >= np.nanmin(wcent_guess)) & (linewave <= np.nanmax(wcent_guess)))
    if not silent:
        print(str(len(wpoints)) + ' lines matched from ' + str(inrng) +' within estimated range.')

    # at this point we have (xpoints, wpoints), so next is generic interpolation.
    #       should this be part of another routine, used by all identify modes?

    # sort the points, just in case the method (or prev run) returns in weird order
    srt = np.argsort(xpoints)
    xpoints = xpoints[srt]
    wpoints = wpoints[srt]

    def XFIT1(x,a,b):
        return a*x+b
    #params, _ = curve_fit(XFIT1, xpoints, wpoints,maxfev=2000)
    #wavesolved = XFIT1(sci_xp, *params)

    #print("for deg ",deg)
    if func=="polyfit":
        fit = np.polyfit(xpoints, wpoints, deg=deg)
        wavesolved = np.polyval(fit, sci_xp)
    #fpt = np.polyval(fit, xpoints)
    if func in ("linear","slinear","zero","quadratic"):
        spl = interp1d(xpoints, wpoints, kind=func, fill_value='extrapolate')
        wavesolved = spl(sci_xp)
    if func == "spline":
        spl = UnivariateSpline(xpoints, wpoints, ext=0, k=3, s=len(pcent_pix)*10)
        wavesolved = spl(sci_xp)
    if func =="legendre":
        wavesolved = legnedre_sol(xpoints,wpoints,sci_xp,deg)
    if func =="cubic":
        cs = CubicSpline(xpoints, wpoints, bc_type='natural')  # Natural cubic spline
        wavesolved = cs(sci_xp)
    
    pcent_pix1, wcent_pix1 = Find_peaks(wavesolved, flux, pwidth=10,pthreshold=pthreshold,minsep=autotol)
    A,B,C =add.closeval2(wcent_pix1,linewave,autotol)
    B = np.array(B)
    C = np.array(C)
    # mean_obs = np.mean(B)
    # SS_res = (B - C) ** 2
    # SS_tot = (B - mean_obs) ** 2

    # # Calculate R-squared
    # R_squared = 1 - (np.sum(SS_res) / np.sum(SS_tot))
    # print("R-squared:", R_squared)

    # Create figure
    fig = plt.figure(figsize=(20, 8))

    # Create a GridSpec with 2 rows and 2 columns
    # width_ratios controls the relative width of each column
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 4], height_ratios=[2, 2.5])

    # Create subplots using the GridSpec
    ax00 = fig.add_subplot(gs[0, 0])  # top-left
    ax10 = fig.add_subplot(gs[1, 0])  # bottom-left
    ax01 = fig.add_subplot(gs[0, 1])  # top-right (rectangle)
    ax11 = fig.add_subplot(gs[1, 1])

    ax00.scatter(linewave,linewave,marker="*",label=f"ref {len(linewave)} data")
    ax00.scatter(wcent_guess,wcent_guess,alpha=0.5,label=f"estimated {len(wcent_guess)} data")
    ax00.set_xlabel(r"wavelength $\AA$")
    ax00.set_ylabel(r"wavelength $\AA$")
    ax00.legend()
    ax00.set_title("Data points used")
    ax10.scatter(B,B-C)
    ax10.axhline(0,color="k",ls="dotted",lw=2,alpha=0.5)
    ax10.set_xlabel("wavelength")
    ax10.set_ylabel("residual")
    ax10.set_title("After wavelength solution")
    ax10.set_xlim([wave_lw,wave_uw])

    ax01.plot(fluxspec)
    for i,w in zip(xpoints,wpoints):
        ax01.axvline(x=i,color="r",ls="dashed",alpha=0.5)
        ax01.text(x=i+2,y=0.65,s=w,rotation=90)
    ax01.set_title("Lines identified to closet value in ref. table")
    ax01.set_xlabel("pixel")
    ax01.set_ylabel("counts")
    ax01.xaxis.set_inverted(True)
    solvedbin = np.where((wavesolved > wave_lw) & (wavesolved < wave_uw))[0]
    srt1 = np.argsort(wavesolved[solvedbin])
    ax11.plot(wavesolved,flux)
    for i in wcent_pix1:
        if np.logical_and(i > wave_lw,i < wave_uw):
            ax11.axvline(i,color="r",ls="dashed",alpha=0.5)
            ax11.text(x=i,y=np.max(flux)-2*np.std(flux),s=np.round(i,3),rotation=90)

    for ii in C:
        ax11.axvline(ii,color="brown",ls="dotted",alpha=0.8)

    for j,jj in enumerate(wcent_pix):
        ax11.axvline(jj,color="k",alpha=0.2,lw=6)
        #plt.text(jj,y=0.9,s=j,rotation=45)

    ax11.set_title("wavelength solved spectra")
    ax11.set_xlabel(r"wavelength $\AA$")
    ax11.set_xlim([wave_lw,wave_uw])
    plt.tight_layout()
    if saveplot:
        if plotoutdir !=None: plt.savefig(os.path.join(plotoutdir,"final_wavecalib.png"))
        else: print("provide directory to save wave calibrated plot")
    plt.show()


    return wavesolved

def find_peaks_robust(flux, prominence=None, distance=5, threshold_percentile=90, width=2):
    """
    Robust peak finding with background subtraction.
    """
    # 1. Estimate and subtract background using a median filter
    # (Using a window size relative to the spectrum length, e.g., 5%)
    #window_size = int(len(flux) * 0.05)
    #if window_size % 2 == 0: window_size += 1
    # Simple rolling median approximation (or use scipy.ndimage.median_filter)
    # For speed/simplicity here, we'll just use a percentile threshold if background is flat-ish,
    # but let's try a simple minimum filter for background estimation if needed.
    # For now, the user's method of percentile is okay, but let's make it safer.
    
    if prominence is None:
        # Auto-calculate prominence based on data range
        prominence = (np.percentile(flux, 95) - np.percentile(flux, 5)) * 0.05

    height_thresh = np.percentile(flux, threshold_percentile)
    
    peaks, properties = find_peaks(flux, height=height_thresh, 
                                  distance=distance, prominence=prominence, width=width)
    
    # Refine peak centers using centroid or gaussian fit around the peak
    refined_centers = []
    for p in peaks:
        # Simple centroid in a small window
        window = 3
        start = max(0, p - window)
        end = min(len(flux), p + window + 1)
        y_segment = flux[start:end]
        x_segment = np.arange(start, end)
        
        # Weighted average (centroid)
        if np.sum(y_segment) > 0:
            centroid = np.sum(x_segment * y_segment) / np.sum(y_segment)
            refined_centers.append(centroid)
        else:
            refined_centers.append(p)
            
    return peaks, np.array(refined_centers), properties['peak_heights']

def robust_poly_fit(x, y, deg=3, reject_sigma=3.0, max_iterations=5):
    """
    Fit a Chebyshev polynomial with iterative outlier rejection (sigma clipping).
    Chebyshev polynomials are more stable at the edges than standard polynomials.
    """
    mask = np.ones(len(x), dtype=bool)
    domain = [x.min(), x.max()] # Domain for Chebyshev mapping
    
    fit_res = None
    
    for i in range(max_iterations):
        x_fit = x[mask]
        y_fit = y[mask]
        
        if len(x_fit) < deg + 2:
            break
            
        # Fit Chebyshev polynomial
        # domain mapping is handled automatically by the class if we use the fit classmethod properly,
        # but numpy.polynomial.Chebyshev.fit maps x to [-1, 1] based on the domain of x provided or inferred.
        try:
            c_fit = Chebyshev.fit(x_fit, y_fit, deg, domain=domain)
        except Exception:
            # Fallback to standard polyfit if Chebyshev fails (rare)
            break
            
        residuals = y_fit - c_fit(x_fit)
        std_res = np.std(residuals)
        
        if std_res == 0:
            break
            
        # Identify outliers
        # We calculate residuals for ALL points to check if any deleted ones should come back?
        # Standard sigma clipping usually just removes.
        all_residuals = y - c_fit(x)
        new_mask = np.abs(all_residuals) < (reject_sigma * std_res)
        
        if np.array_equal(mask, new_mask):
            fit_res = c_fit
            break
            
        mask = new_mask
        fit_res = c_fit
        
    return fit_res, mask

def match_lines_iterative(observed_peaks, reference_waves, initial_guess_func, tolerance=10):
    """
    Iteratively match peaks to reference lines.
    1. Predict wavelengths using current solution.
    2. Find nearest reference line.
    3. Update solution.
    """
    matches_x = []
    matches_w = []
    
    # Initial prediction
    predicted_waves = initial_guess_func(observed_peaks)

    # First pass matching
    for i, pred_w in enumerate(predicted_waves):
        # Find closest reference line
        diffs = np.abs(reference_waves - pred_w)
        min_idx = np.argmin(diffs)
        min_diff = diffs[min_idx]
        
        if min_diff < tolerance:
            matches_x.append(observed_peaks[i])
            matches_w.append(reference_waves[min_idx])
            
    matches_x = np.array(matches_x)
    matches_w = np.array(matches_w)
    
    if len(matches_x) < 5:
        return matches_x, matches_w
        
    # Refine fit and re-match (Iterative "Zipper" effect)
    # We use a lower order for the first pass to get the global slope right
    # Then higher order to catch curvature
    
    # Pass 1: Robust Linear/Quadratic Fit
    poly, mask = robust_poly_fit(matches_x, matches_w, deg=2, reject_sigma=3.0)
    
    # Re-predict ALL peaks with this new solution
    new_predicted = poly(observed_peaks)
    
    final_matches_x = []
    final_matches_w = []
    
    # Stricter tolerance for final matching
    final_tolerance = tolerance * 0.8
    
    for i, pred_w in enumerate(new_predicted):
        diffs = np.abs(reference_waves - pred_w)
        min_idx = np.argmin(diffs)
        min_diff = diffs[min_idx]
        
        if min_diff < final_tolerance:
            final_matches_x.append(observed_peaks[i])
            final_matches_w.append(reference_waves[min_idx])
            
    return np.array(final_matches_x), np.array(final_matches_w)

def calibrate_spectrum(flux, linelist, initial_guess_waves=None,pthreshold=90,autotol=10, order=3, plot=False):
    """
    Main calibration function.
    
    Args:
        flux: 1D array of spectral flux
        linelist: list of reference wavelengths
        initial_guess_waves: 1D array of wavelengths corresponding to pixels (approximate solution)
        order: Polynomial order for final fit
    """
    x_pixels = np.arange(len(flux))
    flux = flux/np.nanmax(flux)

    # 1. Robust Peak Finding
    # We use a high percentile to get only strong lines first
    peak_idxs, peak_centers, peak_heights = find_peaks_robust(flux, threshold_percentile=pthreshold, distance=4)
    
    print(f"Found {len(peak_centers)} peaks.")
    
    if len(peak_centers) < 5:
        print("Error: Too few peaks found.")
        return None, None
    
    # 2. Initial Guess Function
    if initial_guess_waves is not None:
        # Create an interpolator for the initial guess
        # Ensure sorted
        if initial_guess_waves[1] < initial_guess_waves[0]:
             # Handle reverse direction if needed, but usually wavelength increases with pixel
             pass
        guess_func = lambda x: np.interp(x, x_pixels, initial_guess_waves)
    else:
        # Fallback: Assume linear coverage of min/max linelist (very rough)
        print("Warning: No initial guess provided. Assuming linear coverage.")
        guess_func = lambda x: np.interp(x, [0, len(flux)], [min(linelist), max(linelist)])

    # 3. Iterative Matching
    # Start with a generous tolerance
    linewave = loadlinelist(linelist)

    matched_x, matched_w = match_lines_iterative(peak_centers, linewave, guess_func, tolerance=autotol)
    
    print(f"Matched {len(matched_x)} lines after iteration.")
    
    if len(matched_x) < order + 2:
        print("Error: Not enough matches for polynomial fit.")
        return None, None
        
    # 4. Final Robust Fit (Chebyshev)
    final_poly, mask = robust_poly_fit(matched_x, matched_w, deg=order, reject_sigma=2)
    
    # Filter outliers from the match list for reporting
    clean_x = matched_x[mask]
    clean_w = matched_w[mask]
    
    # 5. Generate Solution
    wavelength_solution = final_poly(x_pixels)
    
    # Calculate RMS
    residuals = clean_w - final_poly(clean_x)
    rms = np.sqrt(np.mean(residuals**2))
    print(f"Final RMS: {rms:.4f} A with {len(clean_x)} lines used.")
    
    if plot:
        plt.figure(figsize=(15, 9))
        
        # Top: Spectrum + Matches
        plt.subplot(2, 1, 1)
        plt.plot(wavelength_solution, flux, 'k-', alpha=0.6, label='Calibrated Spectrum')
        plt.scatter(clean_w, flux[clean_x.astype(int)], c='r', marker='x', label=f'Used Matches {len(clean_w)}')
        
        # Mark outliers
        outliers_x = matched_x[~mask]
        if len(outliers_x) > 0:
             plt.scatter(matched_w[~mask], flux[outliers_x.astype(int)], c='orange', marker='o', facecolors='none', label=f'Outliers {len(outliers_x)}')
             
        #for lw in linelist:
         #   plt.axvline(lw, color='g', alpha=0.2, ls=':')
        
        for xxi,idxx,res in zip(clean_w,clean_x,residuals):
            idxx=int(idxx)
            plt.axvline(xxi,color="brown",ls="dotted",alpha=0.7)
            plt.text(xxi,flux[idxx]+0.05,np.round(xxi,2),rotation=89,fontsize=8)
            plt.text(xxi,flux[idxx]+0.2,np.round(res,3),color="g",rotation=80) 

        plt.legend()
        plt.title(f"Wavelength Calibration (RMS={rms:.3f})")
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.xlim([3800,8900])
        plt.ylim([-0.01,1.2])
        
        # Bottom: Residuals
        plt.subplot(2, 1, 2)
        plt.scatter(clean_w, residuals, c='b')
        plt.axhline(0, color='k', ls='--')
        plt.xlabel("Pixel")
        plt.ylabel("Residual (A)")
        plt.title("Residuals vs Wavelength")
        
        plt.tight_layout()
        plt.show()
        
    return wavelength_solution, (clean_x, clean_w, residuals)

def fit_arclines(wavespec,fluxspec,arcline,autotol=10,pthreshold=0.94,func="linear",deg=1,saveplot=True,plotoutdir=None):
    if pthreshold < 1: pthreshold=pthreshold*100
    wave,_ = calibrate_spectrum(flux=fluxspec, linelist=arcline, initial_guess_waves=wavespec,pthreshold=pthreshold,autotol=autotol, order=deg, plot=saveplot)
    return wave
    
