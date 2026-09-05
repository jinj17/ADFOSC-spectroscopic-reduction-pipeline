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

def save_plot(spec,name=None,wbin=True,lw=4500,uw=9000,redshift=0.0):
    box_kernel = Box1DKernel(5)
    wave = spec.spectral_axis.value
    flux = spec.flux.value
    if wbin:
        newbin = np.where((wave > lw) & (wave < uw))
        wave = wave[newbin]
        flux = flux[newbin]
        
    newflx = convolve(flux,box_kernel)
    Z = 1+redshift
    plt.figure(figsize=(10,6))
    plt.plot(wave/Z, flux,alpha=0.5, color='r')
    plt.plot(wave/Z,newflx,color='k',label="smoothed (5px)")
    plt.ylabel(r'Flux [$ergs/s/cm^2/\AA$]',fontsize=12)
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
    plt.savefig(imgpath,dpi=200)
    plt.grid(alpha=0.5,ls="--")
    plt.minorticks_on()
    plt.show()
               
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
    
    
