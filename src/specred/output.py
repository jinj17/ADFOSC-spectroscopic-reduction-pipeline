from specred.calib import flux as fluxcal
import numpy as np
import matplotlib.pylab as plt
import os
from astropy.convolution import convolve, Box1DKernel


def final(SCI,standardstar,display=True, name=None):
    final_spectrum = fluxcal.apply_sensfunc(SCI,standardstar)
    
    bin2 = np.where((final_spectrum.spectral_axis.value >4000) & (final_spectrum.spectral_axis.value <8000))
    wave = final_spectrum.spectral_axis[bin2].value
    flux = final_spectrum.flux[bin2].value
    data = np.column_stack((wave,flux))
    if name is not None:
        np.savetxt(name+".csv",data,delimiter=",",
                   header='Wave, Flux', comments='')
    else:
        np.savetxt("final_spectrum.csv",data,delimiter=",",
                   header='Wave, Flux', comments='')
        
    if display:
        plt.figure(figsize=(10,6))
        plt.plot(wave, flux,alpha=0.5, color='r')
        plt.xlabel('Wavelength ['+str(final_spectrum.wavelength.unit)+']',fontsize=12)
        plt.ylabel('Flux ['+str(final_spectrum.flux.unit)+']',fontsize=12)
        if name is not None:
            plt.title(name,fontsize=12)

        plt.show()
def final_plot(spec,name=None,wbin=True,lw=4500,uw=9000):
    box_kernel = Box1DKernel(5)
    
    wave = spec.spectral_axis.value
    flux = spec.flux.value
    if wbin:
        newbin = np.where((wave > lw) & (wave < uw))
        wave = wave[newbin]
        flux = flux[newbin]
    newflx = convolve(flux,box_kernel)
    plt.figure(figsize=(10,6))
    plt.plot(wave, flux,alpha=0.5, color='r')
    plt.plot(wave,newflx,color='k')
    plt.xlabel('Wavelength ['+str(spec.spectral_axis.unit)+']',fontsize=12)
    plt.ylabel('Flux ['+str(spec.flux.unit)+']',fontsize=12)
    if name is not None:
        plt.title(name,fontsize=12)
    plt.grid(alpha=0.5,ls="--")
    plt.minorticks_on()
    plt.show()

def preview_arrays(spec, wbin=False, lw=4500, uw=9000, redshift=0.0, binsize=5):
    """Return raw and non-overlapping mean-binned samples for display only."""
    wave = np.asarray(spec.spectral_axis.value, dtype=float)
    flux = np.asarray(spec.flux.value, dtype=float)
    if redshift <= -1 or binsize < 1:
        raise ValueError('Invalid redshift or bin size.')
    if wbin:
        if lw >= uw:
            raise ValueError('Lower wavelength cutoff must be below upper cutoff.')
        mask = (wave >= lw) & (wave <= uw)
        wave, flux = wave[mask], flux[mask]
    if not np.any(np.isfinite(wave) & np.isfinite(flux)):
        raise ValueError('No finite spectrum in the selected range.')
    wave = wave / (1 + redshift)
    bw, bf = [], []
    for start in range(0, len(wave), binsize):
        x, y = wave[start:start+binsize], flux[start:start+binsize]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.any():
            bw.append(np.mean(x[valid]))
            bf.append(np.mean(y[valid]))
        else:
            bw.append(np.nan)
            bf.append(np.nan)
    return wave, flux, np.asarray(bw), np.asarray(bf)


def save_plot(spec,name=None,wbin=True,lw=4500,uw=9000,redshift=0.0,binsize=5,display=True):
    wave, flux, bw, bf = preview_arrays(spec, wbin, lw, uw, redshift, binsize)
    plt.figure(figsize=(10,6))
    plt.plot(wave, flux,alpha=0.5, color='r', label='Observed spectrum')
    if binsize > 1:
        plt.plot(bw,bf,color='k',label=f'Binned ({binsize} px)')
    plt.ylabel(f'Flux [{spec.flux.unit}]',fontsize=12)
    if redshift >0.0:
        plt.axvline(6563,color="green",alpha=0.7,ls="dashed",label="H alpha")
        plt.axvline(5007,color="violet",alpha=0.7,ls="dashed",label="OIII")
        plt.axvline(4861,color="brown",alpha=0.7,ls="dashed",label="H beta")
        plt.xlabel('Rest Frame Wavelength ['+str(spec.spectral_axis.unit)+']',fontsize=12)
    else: plt.xlabel('Wavelength ['+str(spec.spectral_axis.unit)+']',fontsize=12)
    if name is not None:
        plt.title(name,fontsize=12)
    imgpath = os.path.join(os.getcwd(),f"final_{name}.png")
    plt.legend()
    plt.grid(alpha=0.5,ls="--")
    plt.minorticks_on()
    plt.savefig(imgpath,dpi=200,bbox_inches='tight')
    if display:
        plt.show()
    else:
        plt.close()
               
def save_spectrum(spec,name):
    wave = spec.spectral_axis.value
    flux = spec.flux.value
    fluxerr = spec.uncertainty.array
    data = np.column_stack((wave,flux,fluxerr))
    box_kernel = Box1DKernel(5)
    newflx = convolve(flux,box_kernel)
    newflxerr = convolve(fluxerr,box_kernel)
    newdata = np.column_stack((wave,newflx,newflxerr))
    if name is not None:
        np.savetxt(name+".csv",data,delimiter=",",
                   header='Wave, Flux, Fluxerr', comments='')
        np.savetxt(name+"binned.csv",newdata,delimiter=",",
                   header='Wave, Flux, Fluxerr', comments='')
    else:
        np.savetxt("final_spectrum.csv",data,delimiter=",",
                   header='Wave, Flux, Fluxerr', comments='')
        np.savetxt("final_spectrum_binned.csv",newdata,delimiter=",",
                   header='Wave, Flux, Fluxerr', comments='')
    
    
