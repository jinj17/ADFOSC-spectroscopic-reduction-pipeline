import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from specutils import Spectrum1D
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from specred import utils as add
import ccdproc
import subprocess
from matplotlib.widgets import Button, Slider
from astropy.io import fits
from datetime import datetime
from astropy.stats import sigma_clip

__all__ = ['trace', 'BoxcarExtract','trfunc','boxfunc']





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
        
        
'''  
def FileName(Filename):
    print("Filename = ",Filename)
    filename = Filename.split("_")
    if len(filename) > 3:
    	name = filename[2]
    else
    #file = fits.open(Filename)
    #hdr = file[0].header
    d = "OBJECT" in hdr
    if d==True:
    	name = hdr["OBJECT"]
    	if name !=[]:
    	     print("Name in hdr=",name)
    	else:
    	     name=filename
    else:
    	name = filename
    	print("given name = ",name)
    
    name  = filename
    print("Name = ",name)
    return name
    
'''




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


def trace(img,log_file,filename, nbins=20, guess=None, window=None,
          Saxis=0, Waxis=1,Xlim=None, display=False, ax=None,otype="None"):
    """
    Trace the spectrum aperture in an image

    Assumes wavelength axis is along the X, spatial axis along the Y.
    Chops image up in bins along the wavelength direction, fits a Gaussian
    within each bin to determine the spatial center of the trace. Finally,
    draws a cubic spline through the bins to up-sample trace along every X pixel.

    Parameters
    ----------
    img : 2d numpy array, or CCDData object
        This is the image to run trace over
    nbins : int, optional
        number of bins in wavelength (X) direction to chop image into. Use
        fewer bins if trace is having difficulty, such as with faint
        targets (default = 20, but minimum must be 4)
    guess : int, optional
        A guess at where the desired trace is in the spatial direction (Y). If set,
        overrides the normal max peak finder. Good for tracing a fainter source if
        multiple traces are present.
    window : int, optional
        If set, only fit the trace within a given region around the guess position.
        Useful for tracing faint sources if multiple traces are present, but
        potentially bad if the trace is substantially bent or warped.
    display : bool, optional
        If set to true display the trace over-plotted on the image
    Saxis : int, optional
        Set which axis is the spatial dimension. For DIS, Saxis=0
        (corresponds to NAXIS2 in header). For KOSMOS, Saxis=1.
        (Default is 0)
    Waxis : int, optional
        Set which axis is the wavelength dimension. For DIS, Waxis=1
        (corresponds to NAXIS1 in the header). For KOSMOS, Waxis=0.
        (Default is 1)
        NOTE: if Saxis is changed, Waxis will be updated, and visa versa.
    ax : matplotlib axes or subplot object, optional
        axes or subplot to be plotted onto. If not specified one will be 
        created. (Default is None)
    otype : str
    	 tells the object type for which trace of the dispersion line is to be find.

    Returns
    -------
    my : array
        The spatial (Y) positions of the trace, interpolated over the
        entire wavelength (X) axis

    """

    # Improvements Needed
    # -------------------
    # 1) switch to astropy models for Gaussian (?)
    # 2) return info about trace width (?)
    # 3) add re-fit trace functionality (or break off into another method)
    # 4) add other interpolation modes besides spline, maybe via
    #     specutils.manipulation methods?

    # define the wavelength & spatial axis, if we want to enable swapping programatically later
    # defined to agree with e.g.: img.shape => (1024, 2048) = (spatial, wavelength)

    # Require at least 4 big bins along the trace to define shape. Sometimes can get away with very few
    if nbins < 4:
        raise ValueError('nbins must be >= 4')

    # old DIS default was Saxis=0, Waxis=1, shape = (1028,2048)
    # KOSMOS is swapped, shape = (4096, 2148)
    #img = ccdproc.CCDData(img.data.T,unit="adu")
    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0
        
    else:
        Saxis = 0
        Waxis = 1
        
    print("Saxis = ",Saxis,"Waxis= ",Waxis)
    #############################################################
    if Xlim is None:
        Xlim = img.shape[Waxis]

    # Pick the highest peak, bad if mult. obj. on slit...
    ztot = np.nansum(img.data, axis=Waxis) / img.shape[Waxis]  # average image data across all wavelengths
    yy = np.arange(len(ztot))
    peak_y = np.nanargmax(ztot)

    # if interact:
    #     guess = ap_interac(yy, ztot)

    # if the user set a guess for where the peak was, adopt that
    if guess is not None:
        peak_y = guess

    # guess the peak width as the FWHM, roughly converted to gaussian sigma
    width_guess = np.size(yy[ztot > (np.nanmax(ztot)/2.)]) / 2.355
    # enforce some (maybe sensible?) rules about trace peak width
    if width_guess < 2.:
        width_guess = 2.
    if width_guess > 25:
        width_guess = 25

    # [avg peak height, baseline, Y location of peak, width guess]
    peak_guess = [np.nanmax(ztot), np.nanmedian(ztot), peak_y, width_guess]

    # fit a Gaussian to peak for fall-back answer, but don't use yet
    popt_tot, pcov = curve_fit(_gaus, yy, ztot, p0=peak_guess)

    if window is not None:
        ilum2 = yy[np.arange(peak_y-window, peak_y+window, dtype=int)]
    else:
        ilum2 = yy

    xbins = np.linspace(0,Xlim, nbins+1, dtype='int')  #    xbins = np.linspace(0, img.shape[Waxis], nbins+1, dtype='int')

    ybins = np.zeros(len(xbins)-1, dtype='float') * np.nan

    for i in range(0, len(xbins)-1):
        # fit gaussian within each window
        if Waxis == 1:
            zi = np.nansum(img.data[ilum2, xbins[i]:xbins[i+1]], axis=Waxis)
        if Waxis == 0:
            zi = np.nansum(img.data[xbins[i]:xbins[i+1], ilum2], axis=Waxis)

        peak_y = ilum2[np.nanargmax(zi)]
        width_guess = np.size(ilum2[zi > (np.nanmax(zi) / 2.)]) / 2.355

        if width_guess < 2.:
            width_guess = 2.
        if width_guess > 25:
            width_guess = 25
        pguess = [np.nanmax(zi), np.nanmedian(zi), peak_y, width_guess]
        try:
            popt, _ = curve_fit(_gaus, ilum2, zi, p0=pguess)

            # if gaussian fits off chip, then fall back to previous answer
            if (popt[2] <= min(ilum2)) or (popt[2] >= max(ilum2)):
                ybins[i] = popt_tot[2]
            else:
                ybins[i] = popt[2]
                popt_tot = popt  # if a good Gaussian fit, switch to these parameters for next fall-back

        except RuntimeError:
            popt = pguess

    # recenter the bin positions
    xbins = (xbins[:-1] + xbins[1:]) / 2.

    yok = np.where(np.isfinite(ybins))[0]
    if len(yok) > 0:
        xbins = xbins[yok]
        ybins = ybins[yok]

        # run a cubic spline thru the bins
        ap_spl = UnivariateSpline(xbins, ybins, k=3, s=32)

        # interpolate the spline to 1 position per column
        mx = np.arange(0, Xlim)   #mx = np.arange(0, img.shape[Waxis])
        my = ap_spl(mx)
    else:
        mx = np.arange(0, Xlim)   #mx = np.arange(0, img.shape[Waxis])
        my = np.zeros_like(mx) * np.nan
        import warnings
        warnings.warn("TRACE ERROR: No Valid points found in trace")
    
    vmn = None
    vmx = None
    if display is True:
        if ax is None:
        	 pass
            #fig, ax = plt.subplots(1,1)
        #im = ax.imshow(img, origin='lower', aspect='auto', cmap=plt.cm.Greys_r) 
        vmn,vmx,ax1 = add.imap_slider(img.data,vmn,vmx)
        #im = ax.imshow(img, origin='lower', aspect='auto', cmap="gray",vmin = np.mean(img.data)-4*np.std(img.data),vmax  = np.mean(img.data)+3*np.std(img.data))
        #im.set_clim(np.percentile(img, (5, 98)))
        if Waxis == 1:
            ax1.scatter(xbins, ybins, alpha=0.5)
            ax1.plot(mx, my)
        if Waxis == 0:
            ax1.scatter(ybins, xbins, alpha=0.5)
            ax1.plot(my, mx)
        if otype is None:
        	 plt.title(" ")
        else:
            Filename = filename.replace(".fit","")[2:]
            if otype=="std":
             plt.title(" standard star ")
             plt.savefig(f"trace_{Filename}.png")
            elif otype=="src":
             plt.title(" source ")
             plt.savefig(f"trace_{Filename}.png")
        	 
        	
        #plt.show()
        
        plt.scatter(xbins,ybins)
        plt.plot(my)
        chi =[]
        for i in range(len(xbins)):
        	df = (my[int(xbins[i])] - ybins[i])**2/ybins[i]
        	chi.append(df)
        chisq = np.sum(chi)
        print("chisq = ",chisq)
        plt.figtext(0.8,0.8,"Chi_sq :%.2e"%chisq)
        plt.show()
        
    log_trace_read(log_file,filename,my,peak_guess[2],Xlim,nbins)
    return my


def BoxcarExtract(img, trace_line,log_file, filename,airmass=0,apwidth=8, skysep=7, skywidth=7, skydeg=0,
                  Saxis=0, Waxis=1,method="sum",display=True, ax=None,otype=None):
    """
    This is nearly identical to specreduce.extract.BoxcarExtract,
    because that was based on the same PyDIS source code as this.

    1. Extract the spectrum using the trace. Simply add up all the flux
    around the aperture within a specified +/- width.

    Note: implicitly assumes wavelength axis is perpendicular to
    the trace.

    2. Fits a polynomial to the sky at each column

    3. Computes the uncertainty in each pixel

    Parameters
    ----------
    img : CCDData object
        This is the image to run extract over
    trace_line : 1-d array
        The spatial positions (Y axis) corresponding to the center of the
        trace for every wavelength (X axis), as returned from trace
    apwidth : int, optional
        The width along the Y axis on either side of the trace to extract.
        Note: a fixed width is used along the whole trace.
        (default is 8 pixels, must be at least 1 pixel)
    skysep : int, optional
        The separation in pixels from the aperture to the sky window.
        (Default is 7, must be at least 1 pixel)
    skywidth : int, optional
        The width in pixels of the sky windows on either side of the
        aperture. (Default is 7, must be at least 1 pixel)
    skydeg : int, optional
        The polynomial order to fit between the sky windows.
        (Default is 0)
    Saxis : int, optional
        Set which axis is the spatial dimension. For DIS, Saxis=0
        (corresponds to NAXIS2 in header). For KOSMOS, Saxis=1.
        (Default is 0)
    Waxis : int, optional
        Set which axis is the wavelength dimension. For DIS, Waxis=1
        (corresponds to NAXIS1 in the header). For KOSMOS, Waxis=0.
        (Default is 1)
        NOTE: if Saxis is changed, Waxis will be updated, and visa versa.
    ax : matplotlib axes or subplot object, optional
        axes or subplot to be plotted onto. If not specified one will be 
        created. (Default is None)

    Returns
    -------
    spec : Spectrum1D object
        The extracted spectrum
    skyspec : Spectrum1D object
        The sky spectrum used in the extraction process

    """

    # Improvements Needed
    # -------------------
    # 1. take a wavelength solution for the trace, interpolate sky region
    #     onto same wavelengths as the trace (BIG IMPROVEMENT)
    #     Maybe use specutils.manipulation.FluxConservingResampler for the
    #     interpolation over the sky regions?
    # 2. optionally allow mode to be either simple aperture (current) or the
    #     "optimal" (variance weighted) extraction algorithm

    # old DIS default was Saxis=0, Waxis=1, shape = (1028,2048)
    # KOSMOS is swapped, shape = (4096, 2148)
    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0

    if apwidth < 1:
        raise ValueError('apwidth must be >= 1')
    if skysep < 1:
        raise ValueError('skysep must be >= 1')
    if skywidth < 1:
        raise ValueError('skywidth must be >= 1')

    onedspec = np.zeros_like(trace_line)
    skysubflux = np.zeros_like(trace_line)
    fluxerr = np.zeros_like(trace_line)

    flux_1d = np.zeros_like(trace_line)
    variance_1d = np.zeros_like(trace_line)
    x = np.arange(len(trace_line))
    read_noise=6 #ADFOSC
    gain = 1  #ADFOSC
    print(f" the method for extraction :{method}")
    for i in range(0, len(trace_line)):
        # first do the aperture flux
        # juuuust in case the trace gets too close to an edge
        widthup = apwidth
        widthdn = apwidth
        if (trace_line[i]+widthup > img.shape[Saxis]):
            widthup = img.shape[Saxis]-trace_line[i] - 1
        if (trace_line[i]-widthdn < 0):
            widthdn = trace_line[i] - 1

        
        if method=="sum":
            # simply add up the total flux around the trace_line +/- width
            if Saxis==0:
                #onedspec[i] = np.nansum(img.data[i,int(trace_line[i]-widthdn):int(trace_line[i]+widthup+1)])   
                onedspec[i] = np.nansum(img.data[int(trace_line[i] - widthdn):int(trace_line[i] + widthup + 1),i])
            if Saxis==1:
                #onedspec[i] = np.nansum(img.data[int(trace_line[i] - widthdn):int(trace_line[i] + widthup + 1),i])
                onedspec[i] = np.nansum(img.data[i,int(trace_line[i]-widthdn):int(trace_line[i]+widthup+1)])

        if method == "variance":
            if Saxis == 0:
                data_slice = img.data[int(trace_line[i] - widthdn):int(trace_line[i] + widthup + 1), i]
            else:
                data_slice = img.data[i, int(trace_line[i] - widthdn):int(trace_line[i] + widthup + 1)]

            variance_slice = read_noise**2 + data_slice * gain
            clipped_data = sigma_clip(data_slice, sigma=5, maxiters=3)
            good_mask = ~clipped_data.mask

            profile = np.nan_to_num(clipped_data.data * good_mask, nan=0.0)
            profile_sum = np.sum(profile)
            '''
            if profile_sum == 0 or np.sum(good_mask) < 3:
                flux_1d[i] = np.nan
                variance_1d[i] = np.inf
                continue
            ''' 
            profile /= profile_sum
            P = profile
            D = data_slice
            V = variance_slice

            numerator = np.sum(P[good_mask] * D[good_mask] / V[good_mask])
            denominator = np.sum(P[good_mask]**2 / V[good_mask])

            flux_1d[i] = numerator / denominator if denominator != 0 else np.nan
            variance_1d[i] = 1.0 / denominator if denominator != 0 else np.inf

        # now do the sky fit
        itrace_line = int(trace_line[i])
        y = np.append(np.arange(itrace_line-apwidth-skysep-skywidth, itrace_line-apwidth-skysep),
                      np.arange(itrace_line+apwidth+skysep+1, itrace_line+apwidth+skysep+skywidth+1))

        if Saxis == 0:
            z = img.data[y, i]
        if Saxis == 1:
            z = img.data[i, y]

        if skydeg>0:
            # fit a polynomial to the sky in this column
            pfit = np.polyfit(y, z, skydeg)
            # define the aperture in this column
            ap = np.arange(trace_line[i]-apwidth, trace_line[i]+apwidth+1)
            # evaluate the polynomial across the aperture, and sum
            skysubflux[i] = np.nansum(np.polyval(pfit, ap))
        elif skydeg == 0:
            skysubflux[i] = np.nanmedian(z)*(apwidth*2.0 + 1)

        # finally, compute the error in this pixel
        sigB = np.nanstd(z)  # stddev in the background data
        N_B = float(len(y))  # number of bkgd pixels
        N_A = apwidth * 2. + 1  # number of aperture pixels

        # based on aperture phot err description by F. Masci, Caltech:
        # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
        if method =="sum":
            fluxerr[i] = np.sqrt(np.nansum((onedspec[i]-skysubflux[i])) +(N_A + N_A**2. / N_B) * (sigB**2.))
    if method=="variance":
        fluxerr = variance_1d
        onedspec = flux_1d
        
    vmn = None
    vmx = None
    if display:
        if otype != None:
            if ax is None:
                     pass
                #fig, ax = plt.subplots(1,1)
            vmn,vmx,ax1 = add.imap_slider(img.data,vmn,vmx)            
            #im = ax.imshow(img, origin='lower', aspect='auto', cmap=plt.cm.Greys_r)
            #im.set_clim(np.percentile(img, (5, 98)))

            if Saxis == 0:
                ax1.plot(np.arange(len(trace_line)), trace_line, c='C0')
                #ax1.fill_between(np.arange(len(trace_line)), trace_line + apwidth, trace_line-apwidth, color='C0', alpha=0.2)
                ax1.plot(np.arange(len(trace_line)), trace_line + apwidth, color='C0', alpha=0.4)
                ax1.plot(np.arange(len(trace_line)), trace_line-apwidth, color='C0', alpha=0.4)
                ax1.fill_between(np.arange(len(trace_line)), trace_line + apwidth + skysep, trace_line + apwidth + skysep + skywidth, color='C1', alpha=0.2)
                ax1.fill_between(np.arange(len(trace_line)), trace_line - apwidth - skysep, trace_line - apwidth - skysep - skywidth, color='C1', alpha=0.2)
            if Saxis == 1:
                ax1.plot(trace_line, np.arange(len(trace_line)), c='C0')
                ax1.plot(trace_line + apwidth, np.arange(len(trace_line)), c='C0', alpha=0.5)
                ax1.plot(trace_line - apwidth, np.arange(len(trace_line)), c='C0', alpha=0.5)
                ax1.plot(trace_line - apwidth - skysep - skywidth, np.arange(len(trace_line)), c='C1', alpha=0.5)
                ax1.plot(trace_line - apwidth - skysep, np.arange(len(trace_line)), c='C1', alpha=0.5)
                ax1.plot(trace_line + apwidth + skysep + skywidth, np.arange(len(trace_line)), c='C1', alpha=0.5)
                ax1.plot(trace_line + apwidth + skysep, np.arange(len(trace_line)), c='C1', alpha=0.5)

            #plt.show()
            onedspec = onedspec#[::-1]
            skysubflux = skysubflux#[::-1]
            fluxerr = fluxerr#[::-1]
            hshift = np.mean(onedspec)+2*np.std(onedspec)
            plt.figure(figsize=(15,6))
            plt.plot(onedspec+hshift,label="extracted spectrum")
            plt.plot(skysubflux-hshift,label="sky spectrum")
            plt.plot(onedspec-skysubflux,label="sky sub spectrum")
            plt.legend()
            if otype is None:
                     plt.title(" ")
            else:
                    Filename = filename.replace(".fit","")[2:]
                    if otype=="std":
                     plt.title(" standard star ")
                     plt.savefig(f"STD_ap_{Filename}.png")
                    elif otype=="src":
                     plt.title(" source ")
                     plt.savefig(f"SRC_ap_{Filename}.png")
            plt.show()


    spec = Spectrum1D(spectral_axis=np.arange(len(onedspec)) * u.pixel,
                      flux=onedspec * img.unit,
                      uncertainty=StdDevUncertainty(fluxerr)
                      )
    skyspec = Spectrum1D(spectral_axis=np.arange(len(onedspec)) * u.pixel,
                         flux=skysubflux * img.unit
                         )
    name = filename
    print("otype = ",otype)
    print(f" spec error : {spec.uncertainty}")
    if airmass != None:
        img.header["airmass"] = airmass
       
    if otype=="lamp":
        print("for lamp")
        Sky_sub_data = onedspec
    else:
        Sky_sub_data = onedspec - skysubflux
        
        fluxerr_hdul = fits.PrimaryHDU(data = fluxerr,header=img.header)  
        fluxerr_hdul.writeto("fluxerr_"+name,overwrite=True)
        
        flux_hdul = fits.PrimaryHDU(data = onedspec,header = img.header)
        flux_hdul.writeto("raw_"+name,overwrite=True)
        
        
        
        
    src_hdul = fits.PrimaryHDU(data=Sky_sub_data,header=img.header)
    src_hdul.writeto("ap_"+name,overwrite=True)
    
    
    
    
    sky_hdul = fits.PrimaryHDU(data=skysubflux,header=img.header)
    sky_hdul.writeto("skyline_"+name,overwrite=True)
    if log_file!=None:
        log_ap_read(log_file,filename,apwidth,skywidth,skysep,skydeg)
    return spec, skyspec

# work on a new extract that first de-warps the wavelength along the spatial axis...
#    and maybe does something more PSF-like in the extraction

# need to add "optimal extraction" method for weighting data vs sky


def sides(f):
    # side code snippet to seperate the inputs if provided either as int or tuple
    if isinstance(f,int):
        u=f
        l=f
    elif isinstance(f,tuple):
        if len(f)!=2:
            raise TypeError("plz provide two integer value")
        else:
            u=f[0]
            l=f[1]

    print("u=",u," l=",l)
    return u,l


def bkg_sub(sc_file,Saxis=0,Waxis=1,width=150,shift=200,display=False):

    """ 
    Parameters
    -----------
    sc_file: CCDData object
            img for background subtraction
    Saxis: if 0, the dispersion line is along the lines or parallel to column

    width: int or tuple. Default value is 150 pixels
          width value for left and right side region for background subtraction  
    shift: int or tuple. Default value is 200
          region to start from the dispersion line on both the sides for background subtraction

    Returns
    -------
    img: CCDData object
         background subtracted image in 

    """

    width = sides(width)
    shift = sides(shift)

    
    blu = int(sc_file.shape[Saxis]/2-shift[0])
    bll = int(sc_file.shape[Saxis]/2-(shift[0]+width[0]))
    bul = int(sc_file.shape[Saxis]/2+shift[1])
    buu = int(sc_file.shape[Saxis]/2+(shift[1]+width[1]))
    bkg_x = np.append(np.arange(bll,blu),np.arange(bul,buu))
    

    xrows = np.arange(sc_file.shape[Saxis])
    bkg = np.zeros_like(sc_file)
    for col in np.arange(sc_file.shape[Waxis]):
        pfit = np.polyfit(bkg_x,sc_file[bkg_x,col].data,2)
        bkg[:,col] = np.polyval(pfit,xrows)


    img_bkg = sc_file.data - bkg
    img = ccdproc.CCDData(img_bkg,unit="adu")

    if display:

        fig, axs = plt.subplots(1, 3, figsize=(15, 9))

        # Plot data on each subplot
        axs[0].imshow(sc_file, cmap="gray",vmin=np.mean(bkg)-np.std(bkg), vmax=np.mean(bkg)+np.std(bkg))
        axs[0].set_title('Original')
        axs[0].axhline(blu,color="cyan",ls="dashed",alpha=0.5)
        axs[0].axhline(bll,color="cyan",ls="dashed",alpha=0.5)
        axs[0].axhline(bul,color="magenta",ls="dashed",alpha=0.5)
        axs[0].axhline(buu,color="magenta",ls="dashed",alpha=0.5)

        axs[1].imshow(bkg, cmap="gray",vmin=np.mean(bkg)-np.std(bkg), vmax=np.mean(bkg)+np.std(bkg))
        axs[1].set_title('Background')
        

        axs[2].imshow(img, cmap="gray",vmin=np.mean(bkg)-3*np.std(bkg), vmax=np.mean(bkg)+3*np.std(bkg))
        axs[2].set_title('Bkg Subtracted')

       
        plt.show()
   
    return img

    



def boxfunc(data,trace,filename,airmass=0,apwidth=8,skysep=10,skywidth=7,skydeg=0,Saxis=0,method="sum",display=True,ax=None,otype=None):
        #tr = aperture.trace(data, nbins=nbins, window=window, guess=guess,Saxis=Saxis,Xlim=Xlim, display=display,otype=otype)
        log_file = filename + "_database" + ".txt"
        while True:
                BoxEx,sky = BoxcarExtract(data,trace,log_file,filename,airmass=airmass,apwidth=apwidth,skysep=skysep,skywidth=skywidth,
                skydeg=skydeg,Saxis=Saxis,display=display,otype=otype,method=method)
                

                stng = "y"# input("Type the keyword of the parameter to be changed ('apw', 'skys', 'skyw', 'skyd', or 'y' to exit): ")
                if stng.lower()=="apw":
                        apwidth = int(input("enter the width for aperture extraction"))
                elif stng.lower()=="skyw":
                        skywidth = int(input("enter the width for sky extraction"))
                elif stng.lower()=="skys":
                        skysep = int(input("enter the value of pixels for sky selection region"))
                elif stng.lower()=="skyd":
                        skydeg = int(input("enter the degree of polynomial for sky line fitting"))
                elif stng.lower()=="y":
                        break
                else:
                        print("Invalid input. Please enter a valid integer or 'y'.")
                    
                    
        #log_file = filename + "_database" + ".txt" 
        #log_ap_read(log_file,filename,apwidth,skywidth,skysep,skydeg)        
        return BoxEx,sky
        
        
        
def trfunc(data,filename,nbins=7,window=20,guess=500,Saxis=0,Xlim=1300,display=True,otype="src",):
        #tr = aperture.trace(data, nbins=nbins, window=window, guess=guess,Saxis=Saxis,Xlim=Xlim, display=display,otype=otype)
        
        log_file = filename + "_database" + ".txt"
        print("filename=",filename)             
        with open(log_file, "w") as log:
        	     # Write a header to the log file
                log.write("Database: "+"\n")
                log.write("--"*50 + "\n")
                
        while True:
                tr = trace(data,log_file,filename, nbins=nbins, window=window, guess=guess,Saxis=Saxis,Xlim=Xlim, display=display,otype=otype)

                stng = "y" #input("Type the keyword of the parameter to be changed ('nb', 'xlim', 'cen', or 'y' to continue | 'br' to exit): ")
                if stng.lower()=="nb":
                        nbins = int(input("enter the number of bins"))
                elif stng.lower()=="xlim":
                        Xlim = int(input("enter the Xlim"))
                elif stng.lower()=="cen":
                        guess = int(input("enter the line value"))
                elif stng.lower()=="y":
                        break
                elif stng.lower()=="br":
                	  raise ValueError("Keyword not matched")
                        
                else:
                        print("Invalid input. Please enter a valid integer or 'y'.")     
        
                        
            
               
        
        return tr
   
