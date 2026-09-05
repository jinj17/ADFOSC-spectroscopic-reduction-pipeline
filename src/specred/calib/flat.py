import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel
from astropy import units as u
from astropy.nddata import CCDData
import ccdproc
from astropy.io import fits
from scipy.interpolate import CubicSpline
from astropy.convolution import convolve, Box1DKernel
from ccdproc import Combiner, trim_image
from astropy.stats import mad_std

__all__ =["lamp_response","flat_correction"]



# this cell contains function to find the response curve and form the normailised flat frame.
# also defined the function to do the flat correction.


def read_ccd_data_old(file):
    # since there is some issue with the updated header and ccdproc fits file read format so use fits.open"
    if isinstance(file, str):
        return ccdproc.CCDData.read(file, unit="adu")
    elif isinstance(file, ccdproc.CCDData):
        return file
    elif isinstance(file,fits.hdu.hdulist.HDUList):
        return ccdproc.CCDData(file[0].data,unit="adu")
    
    else:
        raise ValueError("Invalid file format")

def read_ccd_data(file):
    if isinstance(file, str):
        print("read ccd 1")
        #print(f"the file provided in read_ccd_data func {file}")
        #print(f" the header  {fits.open(file)[0].header}")
        return fits.open(file)[0] #ccdproc.CCDData.read(file, unit="adu")
    elif isinstance(file, ccdproc.CCDData):
        print("read ccd 2 ")
        return file
    elif isinstance(file,fits.hdu.hdulist.HDUList):
        print("read ccd 3")
        return fits.open(file[0].data)[0] #ccdproc.CCDData(file[0].data,unit="adu")
    
    else:
        raise ValueError("Invalid file format")
    



def lamp_response(file,flat_combine = False,smooth=False, npix=11, display=False,Saxis=0, Waxis=1, ax=None, func = False):
    
    """
    Divide out the spatially-averaged spectrum response from the flat image.
    This is to remove the spectral response of the flatfield (e.g. Quartz) lamp.

    Input flat is first averaged along the spatial dimension to make a 1-D flat.
    This is optionally smoothed, and then the 1-D flat is divided out of each row
    of the image.

    Note: implicitly assumes spatial and spectral axes are orthogonal, i.e. does not
    trace lines of constant wavelength for normalization.

    If n number of files been provided then at first the flat combine will be done, 
    which is optional. Afterwards the procedure for finding response curve will done

    -----------------------------------------------------------------------------------------

    Parameters
    ----------
    file : flat file or list of flat files
        each file is called as CCDData Object
    flat_combine: bool (default=False)
        if yes, then the files provided in the list will be flat combined using ccdproc.combine    
    smooth : bool (default=False)
        Should the 1-D, mean-combined flat be smoothed before dividing out?
    npix : int (default=11)
        if `smooth=True`, how big of a boxcar smooth kernel should be used (in pixels)?
    display : bool (default=False)
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
    func: bool (default=False)
        fits the spline curve over the smooth-curve or 1-D mean-combined flat

    Returns
    -------
    flat : CCDData object  |
    also saved as fits file (nflats.fits)

    """
    

    #print("Saxis =",Saxis,"Waxis = ",Waxis)
    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0
    else:
        Saxis=0
        Waxis=1
    print("Saxis =",Saxis,"Waxis = ",Waxis)

    
    
    fltlist=[]
    flatdata=[]
    
    if isinstance(file,type([])):
        print(" in flat correction its array")
        print("len = ",len(file))
        if len(file)>1:
            if flat_combine !=False:
                stng = str(input("doing the flat combine first, continue(y|n) ?:"))
                # NOTE (bug fixes, this branch): three issues found during
                # the public-release audit -- not reachable via the current
                # GUI (which always passes a single filename, not a list),
                # but real bugs for anyone calling lamp_response() directly
                # with a list + flat_combine=True:
                #   1. `y` was a bare undefined name, not the string "y".
                #   2. `mad_std` was used but never imported in this file
                #      (now imported above, matching specred.io's usage).
                #   3. `for i in len(file):` tries to iterate over an int;
                #      needs range(len(file)).
                if stng == "y":
                    print("combining the files")
                    for i in range(len(file)):
                        file_data  = read_ccd_data(file[i]).data
                        flatdata.append(file_data)
                        
                        
                mflat = ccdproc.combine(flatdata,method='median',sigma_clip=True,sigma_clip_low_thresh=5,
                                                sigma_clip_high_thresh=5,sigma_clip_func=np.ma.median,sigma_clip_dev_func=mad_std)
                mflat_header = file[0][0].header

                
        elif len(file)==1:
            print("for flat correction, single file")
            file_data = read_ccd_data(file)
            mflat = file_data
            mflat_header = file_data.header
    else:
        print(" for flat correction, idk file format")
        file_data = read_ccd_data(file)
        mflat = file_data.data
        mflat_header = file_data.header
    
        
        
    medflat = mflat
    #print("medflat = ",medflat)
    #medflat = ccdproc.CCDData(mflat,unit = "adu")

    

    # average the data together along the "spatial" axis
    flat_1d = np.nanmedian(medflat, axis=Saxis)
    Flat1d=flat_1d
    #print("flat 1d = ",flat_1d)
    '''
    for i in range(len(flat_1d)):
        	if int(flat_1d[i]) > 61000:
        		print("max count obtained = ",int(flat_1d[i]))
        		msg2 = "the arc lamp contains value more than 60000"
        		raise ValueError(msg2)
    '''   		
  

    # optionally: add boxcar smoothing to the 1-D average
    if smooth:
        sflat_1d = convolve(flat_1d, Box1DKernel(npix), boundary='extend')
        flat_1d = sflat_1d
        
    	 
    if func:
        
        
        x = np.arange(1,medflat.shape[Waxis]+1,1)
        cs = CubicSpline(x,flat_1d)
        
        
        flat_1d = cs(x)
        if display:
            plt.plot(Flat1d,label="median combine of observed data")
            plt.plot(sflat_1d,label="smoothed data")
            plt.plot(flat_1d,label="fit")
            plt.xlabel("pixel")
            plt.ylabel("counts")
            plt.legend()
            plt.title("spectral response")
            #plt.legend()
            plt.show()
    
    '''
    def residual(Cdata,refdata):
        Diff = []
        if len(Cdata)==len(refdata):
            for i in range(len(Cdata)):
                diff = Cdata[i] - refdata[i]
                Diff.append(diff)

        else:
            print("do  not contain same length element")

        return Diff
                

    plt.scatter(x,residual(flat_1d,Flat1d))
    plt.title("residual")
    plt.show()

    plt.scatter(x,residual(flat_1d,sflat_1d))
    plt.title("residual smooth")
    plt.show()

    '''
    
    # ADD? this averaged curve could be modeled w/ spline, polynomial, etc

    # divide the spectral response from the flat lamp (e.g. quartz lamp)
    ## the old way w/ numpy
    # flat = np.zeros_like(medflat)
    # for i in range(medflat.shape[Saxis]):
    #     flat[i, :] = medflat[i, :] / flat_1d

    ## the new way w/ CCDdata objects... i hope!
    # flat = medflat.divide(flat_1d)
    # NOPE, b/c CCDData divide doesn't like dividing arrays across the X-axis (works for Y-axis)

    flat = np.zeros_like(medflat)
    #print("flat= ",flat,"shape=",flat.shape[Saxis])
    #print("medflat = ",medflat)
    #print("flat_1d = ",flat_1d)
    for i in range(flat.shape[Saxis]):
        if Saxis == 0:
            flat[i, :] = medflat[i, :]/flat_1d #medflat[i, :].divide(flat_1d).data
        if Saxis == 1:
            flat[:, i] = medflat[:, i]/flat_1d #medflat[:, i].divide(flat_1d).data
            
    #flat = ccdproc.CCDData(flat, unit=medflat.unit)
    #print("divided flat=",flat)
    

    # once again normalize, since (e.g. if haven't trimmed illumination region)
    # averaging could be skewed by including some non-illuminated portion.
    #flat = flat.divide(np.nanmedian(flat.data))
    #print("re-divided flat=",flat)
    # the resulting flat should just show the pixel-to-pixel variations we're after
    if display:
        if ax is None:
            fig, ax = plt.subplots(1,1)
        im = ax.imshow(flat, origin='lower', aspect='auto', cmap=plt.cm.inferno)
        plt.colorbar(mappable=im)
        
        ax.set_title('flat')
        plt.show()
    mflat_header["comment"]= " Flat corrected"
    print(" made the nflat")
    fits.writeto("nflat.fits",flat,header=mflat_header,overwrite=True)

    return flat
    
######################################################################################
######################################################################################



def flat_correction(cor_list,flat,cosmic_correction=True):
    
    """
    After getting the response curve, it is neccesary to make flat correction to 
    all the source frames (science, standard source, lamp).

    ----------------------------------------------------------------------------

    Parameters:
    -----------
    cor_list : list or file
        list of the files for flat correction
        all files are then called as CCDData object
    
    flat : CCDData object or fits file
        normalised flat file to be used for correction.
    
    Returns
    ---------

    flat corrected source frames in fits format, |
    list of the flat-corrected frames
    

    """
    namelst =[]

    ## first check the flat file
    flat_file = read_ccd_data(flat)    ## reading the normalised flat frame

    for file in cor_list:
        cor_file = read_ccd_data(file)   ## reading all the source frame for flat correction
        hdr = cor_file.header
        new_stng =  str("fc_")+file
        nm = new_stng
        namelst.append(nm)
        #print(f" the file passed {cor_file.data[0]} \n and the nflat {flat_file.data}")
        
        data = np.divide(cor_file.data,flat_file.data) # ccdproc.flat_correct(ccdproc.CCDData(cor_file.data,unit="adu"),ccdproc.CCDData(flat_file.data,unit="adu")) # not using CCDData.divide or simple arithmatic division (i.e cor_file.data/flat_file.data)
                                                        # bcoz it wont preserve the header information.
        #data=ccdproc.flat_correct(ccdproc.CCDData(cor_file.data,unit="adu"),ccdproc.CCDData(flat_file.data,unit="adu"))
        #print("data= ",data[0])
        print(" flat normalized data shape",np.shape(data))
        #data2,_ = ccdproc.cosmicray_median(data.data,rbox=10,error_image=None)
        if cosmic_correction: data2,_ = ccdproc.cosmicray_lacosmic(data, readnoise=6*u.electron,gain=1*u.electron/u.adu, sigclip=5, verbose=True)
        else: data2 = data
        #print(f" cosmicray corrected data shape {np.shape(data2)}")
        #plt.imshow(data2[0],cmap="gray",label="1")
        #plt.imshow(data2[1],cmap="gray",label="2")
        #plt.legend()
        #plt.show()
        #data2 =ccdproc.CCDData(data2,unit="adu")  # for timebeing till gui is prepared
        #print(" the cosmic-ray corrected file shape",np.shape(data2))
        hdr["comment"] = " Cosmic ray removed"
        print("done the cosminc ray correction")
        
        fits.writeto(new_stng,data2,hdr,overwrite=True)
     
     
    return namelst 
    

######################################################################################
###################################################################################### 
