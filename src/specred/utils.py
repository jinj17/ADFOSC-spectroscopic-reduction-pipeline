import numpy as np
import matplotlib.pylab as plt
import ccdproc
from specutils import Spectrum1D
from astropy import units as u
import os
from specutils.fitting import find_lines_derivative
from astropy.convolution import convolve, Box1DKernel
from scipy.signal import find_peaks, peak_widths
import pandas as pd
from matplotlib.widgets import Button, Slider
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, interp1d, CubicSpline
from numpy.polynomial import Legendre as legd


__all__ = ['mshw','line_pks',"stdfile","redshift","redshift2"]

# NOTE (bug fix): this used to be os.path.abspath(__name__), which
# resolves the *module name string* against the current working
# directory rather than the file's actual location -- so `scpath`
# silently pointed at whatever directory the GUI/script happened to be
# launched from, not this package. That made resources/arc/*.dat
# lookups fail whenever this was imported from elsewhere on disk.
scpath = os.path.dirname(os.path.abspath(__file__))
#############################################################################
#############################################################################   
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


def Find_peaks(wave, flux, pwidth=10, pthreshold=0.97, minsep=3):
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
    #print(f' the pcent_pix :{pcent_pix[okcent]}, wcent_pix :{wcent_pix[okcent]} \n for idx of {okcent}')
    return pcent_pix[okcent], wcent_pix[okcent], okcent


### to plot the source frame in greyscale

def closeval2(X, Y, tol=5):
    """
    Finds the close values present in arrays X and Y.
    Returns the index and values of the close elements in X and Y respectively.
    """
    X = np.array(X)
    Y = np.array(Y)
    A=[]
    XX = []
    YY = []
    for i in X:
        D = [abs(float(i) - float(y)) for y in Y]
        aa = pd.Series(D)
        idx = aa.idxmin()  # Get the index of the minimum difference
        
        if D[idx] < tol:  # Check if the closest difference is within tolerance
            A.append((i, Y[idx]))
            XX.append(i)
            YY.append(Y[idx])
    
    return A,XX,YY
    
def closeval3(x,X, Y, tol=5):
    """
    Finds the close values present in arrays X and Y.
    Returns the index and values of the close elements in X and Y respectively.
    """
    X = np.array(X)
    Y = np.array(Y)
    A=[]
    XX = []
    YY = []
    for i in X:
        D = [abs(float(i) - float(y)) for y in Y]
        aa = pd.Series(D)
        idx = aa.idxmin()  # Get the index of the minimum difference
        
        if D[idx] < tol:  # Check if the closest difference is within tolerance
            A.append((i, Y[idx]))
            XX.append(i)
            YY.append(Y[idx])
    
    xbin = np.where(np.isin(X,XX))
    
    return x[xbin],XX,YY
    

def closeval(X, Y,tol=5):
    """
    Finds the closest values in arrays X and Y.
    Returns a list of pairs (x_closest, y_closest) for each element in X and Y.
    """
    X = np.array(X)
    Y = np.array(Y)
    
    closest_pairs = []
    for x in X:
        closest_idx = np.abs(Y - x).argmin()
        
        closest_pairs.append([x, Y[closest_idx]])
    return closest_pairs


    
def mshw(name,cmap="gray"):
    plt.imshow(name,cmap="gray",vmin = np.mean(name)-4*np.std(name),vmax  = np.mean(name)+3*np.std(name))
    #plt.imshow(name, origin='lower', aspect='auto', cmap=plt.cm.Spectral_r)
    #rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
    #plt.gca().add_patch(Rectangle((0,500),2048,500,linewidth=1,edgecolor='r',facecolor='none'))
    #plt.show()


#############################################################################
#############################################################################   

##### to find the line center or peaks in the signal
def line_pks(File,line_no,guess=None,window=150,Saxis=1,Waxis=0,display=True):
    CC=[]
    """
    this following functions find the number of dispersion line as well as the value of the
    dispersion line on axis

    Parameters
    ----------
    spec   :  CCDdata object

    line_no: int type, array
            pass the number or set of numbers in array for the required peaks
    guess  : the pixel value of the disperion line
            if None, takes the central value of the spatial axis 
    window : region within the guess value to detect the dispersion axis
            default 150 pixels
    display: if True, shows the region selected to detect the peak line and plots showing the max peak/s


    
    return
    -------
    1D array carrying pixel value of corresponding peak/s

            
    """

    
    
    #if type(spec_gg) == type(test):
    if isinstance(File,ccdproc.CCDData):
        print("yes")
        #print(spec.header["object"])
    else:
        print("the given obj is of"+type(File))
        print("required object type is Spectrum1D object")
        

    if Saxis == 1:
        Waxis = 0
        print("0k")
    else:
        # Saxis = 0
        Waxis = 1
        print("1k")

    data = File.data.T

    if guess == None:
        guess = int(data.shape[Saxis]/2)
        if int(data.shape[Saxis]/2) >= 200:
            print("1")
            window = 200
        else:
            print("2")
            window = int(data.shape[Saxis]/3)
    else:
        guess = guess
        window = window
    print("guess = ",guess,"window = ",window)
    gl = guess-window
    gu = guess+window
    #mshw(data)
    #plt.show()
    if Saxis ==0:
        plt.axhline(gl,alpha=0.5)
        plt.axhline(gu,alpha=0.5)
        sp = data[gl:gu,:]
        

    else:
        #Saxis = 0
        plt.axvline(gl,alpha=0.5)
        plt.axvline(gu,alpha=0.5)
        sp = data[:,gl:gu]

    #mshw(data)
    #plt.show()
    mshw(sp)
    if Saxis == 1:
        plt.axhline(300)
    else:
        plt.axvline(300)
    plt.show()

    #flux_t = spec.data.T
    ft = np.nanmedian(data,axis=Waxis)
    spec_T = Spectrum1D(flux=ft*u.Jy,spectral_axis=np.arange(1,len(ft)+1,1)*u.AA )
    thld = np.max(spec_T.flux.value) - 4*np.std(spec_T.flux.value)
    print("threshold =",thld)
    
    if display:
        mshw(data)
        plt.axhline(y=gl,color="r")
        plt.axhline(y=gu,color="r")
        plt.show()
        plt.plot(spec_T.spectral_axis.value,spec_T.flux.value)
        plt.axhline(y = thld,color="r")
        plt.show()
   

    ### for flux threshold
    ''' 
    lines = find_lines_derivative(spec_T,flux_threshold=thld)
    print("lines = ",lines)
    lines[lines['line_type']=='emission']
    print("lines = ",lines)
    cc = int(lines["line_center"][0].value)
    '''
    signal = spec_T.flux.value
    #signal = convolve(signal, Box1DKernel(10), boundary='extend')
    #lines = find_peaks(signal, np.max(signal))
    lines = find_peaks(signal, thld)
    print("line =",lines)
    for i in range(0,line_no+1):
        cc = int(lines[0][i])
        print("line center =",cc)
        usr_input= 'y' #str(input("if agree, press Y or enter the column number in integer"))
        
    
        if usr_input.lower() == "y":
            cc= cc
        elif isinstance(usr_input,int()):
            cc = int(usr_input) 
        CC.append(cc)

        if display:
            
            plt.plot(spec_T.spectral_axis.value,spec_T.flux.value)
            plt.axvline(x = int(cc), color = "k",ls="dotted",alpha =0.5)
    plt.show()
    
    
    return CC




def line_pks2(spec,line_no,guess=None,window=150,Saxis=1,Waxis=0,display=True):
    CC=[]
    """
    this following functions find the number of dispersion line as well as the value of the
    dispersion line on axis

    Parameters
    ----------
    spec   :  CCDdata object

    line_no: int type, array
            pass the number or set of numbers in array for the required peaks
    guess  : the pixel value of the disperion line
            if None, takes the central value of the spatial axis 
    window : region within the guess value to detect the dispersion axis
            default 150 pixels
    display: if True, shows the region selected to detect the peak line and plots showing the max peak/s


    
    return
    -------
    1D array carrying pixel value of corresponding peak/s

            
    """

    
    if guess==None:
        guess = spec.shape[Saxis]/2
    else:
        sp =spec.data
        spp = sp[guess-window:guess+window]
        spec = ccdproc.CCDData(spp,unit="adu")
   
    #if type(spec_gg) == type(test):
    if isinstance(spec,ccdproc.CCDData):
        print("yes")
        #print(spec.header["object"])
    else:
        print("the given obj is of"+type(spec))
        print("required object type is Spectrum1D object")
        

    if (Saxis == 1) | (Waxis == 0):
        # if either axis is swapped, swap them both to be sure!
        Saxis = 1
        Waxis = 0
        flux_t = spec.data
    else:
        Saxis=0 
        Waxis =1
        flux_t = spec.data.T

    n =int(200) #int(spec.shape[Saxis]/2-50) #find the center over the wavelength axis
    m = int(200+250)
    print("cut line = ",n,m)
    
    #flux_t = spec.data.T
    ft = np.nanmedian(flux_t[n:m],axis = Waxis )
    spec_T = Spectrum1D(flux=ft*u.Jy,spectral_axis=np.arange(1,spec.shape[Waxis]+1,1)*u.AA )
    thld = np.max(spec_T.flux.value) - 4*np.std(spec_T.flux.value)
    print("threshold =",thld)
    
    if display:
        mshw(spec.data.T)
        plt.axhline(y=n,color="r")
        plt.axhline(y=m,color="r")
        plt.show()
        plt.plot(spec_T.spectral_axis.value,flux_t[n])
        plt.axhline(y = thld,color="r")
        plt.show()
   

    ### for flux threshold
    ''' 
    lines = find_lines_derivative(spec_T,flux_threshold=thld)
    print("lines = ",lines)
    lines[lines['line_type']=='emission']
    print("lines = ",lines)
    cc = int(lines["line_center"][0].value)
    '''
    signal = spec_T.flux.value
    #signal = convolve(signal, Box1DKernel(10), boundary='extend')
    #lines = find_peaks(signal, np.max(signal))
    lines = find_peaks(signal, thld)
    print("line =",lines)
    for i in range(0,line_no+1):
        cc = int(lines[0][i])
        print("line center =",cc)
        usr_input= 'y' #str(input("if agree, press Y or enter the column number in integer"))
        
    
        if usr_input.lower() == "y":
            cc= cc
        elif isinstance(usr_input,int()):
            cc = int(usr_input) 
        CC.append(cc)

        if display:
            plt.plot(spec_T.spectral_axis.value,flux_t[n])
            #plt.plot(spec_T.spectral_axis.value,spec_T.flux.value)
            plt.axvline(x = int(cc), color = "k",ls="dotted",alpha =0.5)
    plt.show()
    
    
    return CC


        


#############################################################################
#############################################################################   



folder = str()
def stdfile(name):
    """
    Try to find the dat file of the required standard star from /resources/onedstar directory.
    if not found then raises the error " not found in any of the folders"
    
    Parameters:
    -----------
    name : dat file | name of the standard star
    
    Return
    -------
    folder : name of the folder containing the dat file
    """
    global folder

    # NOTE (bug fix): this used to be dirname(dirname(__file__)), which
    # pointed one level higher than where scpath (above) looks for
    # resources/arc/ -- the two lookups disagreed on where the shared
    # `resources/` folder actually lives. Standardized to package_dir/resources/.
    maindir = os.path.dirname(os.path.realpath(__file__))

    source_dir =os.path.join(maindir,"resources/onedstar/")

    for i in os.listdir(source_dir):
        print("folder =",i)
        if os.path.exists(os.path.join(source_dir,i,name)) == True :
                          print("present in {} folder".format(i))
                          folder = i
         
    if folder == str():
        print(" not found in any of the folders")
    else:
        return folder 
    

        
                      
                      



#############################################################################
#############################################################################   

from astroquery.simbad import Simbad

def redshift2(object_name):
    # Define the object name (e.g., the name of the astronomical object you're interested in)
    print("finding the redshift of ",object_name)

    # Query Simbad for the object
    Simbad.add_votable_fields("z_value")
    result_table = Simbad.query_object(object_name)
    

    if result_table is not None and 'Z_VALUE' in result_table.colnames:
        z_redshift = result_table['Z_VALUE'][0]
        print(f"Redshift for {object_name}: {z_redshift}")
    else:
        print(f"Redshift information not found for {object_name}")
    #else:
    #   print(f"Simbad query for {object_name} returned no results.")

    return z_redshift


def RA_DEC(object_name):
    Simbad.add_votable_fields("ra")
    result_table = Simbad.query_object(object_name)
    
    if result_table is not None and "RA" in result_table.colnames:
    	RA = result_table["RA"][0]
    	DEC = result_table["DEC"][0]
    	
    return RA,DEC

#############################################################################
#############################################################################   


from astroquery.ned import Ned

def redshift(object_name):
    # Define the object name (e.g., the name of the astronomical object you're interested in)
    print("finding the redshift of ",object_name)

    # Query NED for redshift information
    result = Ned.query_object(object_name)

    # Print the redshift value
    if result is not None and 'Redshift' in result.colnames:
        redshift = result['Redshift'][0]
        print(f"Redshift for {object_name}: {redshift}")
    else:
        print(f"Redshift information not found for {object_name}")



#############################################################################
#############################################################################   

def spectrum_combine(fileflux):
    spec_wavelength =[]
    for i in fileflux:
         spec_wavelength.append(i.spectral_axis.value)
         

#############################################################################
#############################################################################

def imap_slider(data,vmin=None,vmax=None):
    #print("data = ",data)
    if isinstance(data,ccdproc.CCDData):
        data = data.data
    elif isinstance(data,type([])):
        data=data
    mn = np.nanmean(data)
    stdd = np.nanstd(data)
    #print("mn = ",mn,"std = ",stdd)
    fig,ax = plt.subplots()
    if vmin==None:
        vmin = mn-2*stdd
    if vmax == None:
        vmax = mn+2*stdd
    Im = ax.imshow(data,cmap="gray",vmin = vmin ,vmax =vmax,origin="lower")
    
    fig.subplots_adjust(left=0.25, bottom=0.25)
    axvmin = fig.add_axes([0.25,0.1,0.65,0.03])
    vmin_sdr = Slider(ax=axvmin,label = "vmin",valmin=mn-4*stdd,valmax =mn,valinit=vmin)

    axvmax = fig.add_axes([0.1,0.25,0.03,0.65])
    vmax_sdr = Slider(ax=axvmax,label = "vmax",valmin=mn,valmax =mn+4*stdd,valinit=vmax,
                      orientation="vertical")
    def update(val):
        
        #ax.imshow(data,vmin=vmin_sdr.val,vmax = vmax_sdr.val)
        Im.set_clim(vmin = vmin_sdr.val,vmax = vmax_sdr.val)
        fig.canvas.draw_idle()
        
    vmin_sdr.on_changed(update)
    vmax_sdr.on_changed(update)


    #plt.show()

    return vmin_sdr.val, vmax_sdr.val,ax
#############################################################################
#############################################################################
'''
# initial
def linepckr(X,Y,ax=None,color="k",label=None,nopks = None,height=0.1,distance=None,display=False):
    if nopks==None:
        nopks = 8
    else:
        nopks=nopks
    #print("nopks = ",nopks)
    if label==None:
        label=" "
    else:
        pass

    if ax==None:
        fig,ax = plt.subplots(figsize=(10,7))
    else:
        pass
    if distance == None:
        dist = 1
    else:
        dist = distance
    Y = Y/np.max(Y)
    #print("heihgt before peaks=",height,"distance = ",dist)
    peaks,_ = find_peaks(Y,height=height,distance=dist)
    
    #a = sorted(Y[peaks],reverse=True)
    #print("peaks detected = ",len(peaks),"no. of lines = ",nopks)
    i = len(peaks)
    if nopks >= len(peaks):
        #print("height=",height)
        prev_len = len(peaks)
        H=float(height)
        while i < nopks:
            
            H -= 0.005
            peaks, _ = find_peaks(Y, height=H, distance=dist)
            #print("new H = ",H)
            i = len(peaks)
            #if i == prev_len:
             #   break
            #prev_len = len(peaks)
        height= H		
        nopks=i
        #print("new nopks",nopks)
    else:
        print("if no change, nopks=",nopks)
        pass
        a = sorted(Y[peaks],reverse=True)
        height = a[nopks-1]
    #print("Height for {} pks={}".format(nopks,height))
    pks,_ = find_peaks(Y,height=height,distance=dist)
    
    if display:
        for i,j in zip(X[pks],Y[pks]):
            ax.axvline(i,ymax=j,color=color,label=label)
            ax.text(x=i,y=j+0.01,s="{:.2f}".format(i),rotation=90,color=color)
            plt.show()
    
    return pks
'''
#############################################################################
#############################################################################
def linepckr2(X,Y,ax=None,color="k",label=None,nopks = None,height=0.97,distance=None,display=False):
    if nopks==None:
        nopks = 8
    else:
        nopks=nopks
    #print("nopks = ",nopks)
    if label==None:
        label=" "
    else:
        pass

    if ax==None:
        fig,ax = plt.subplots(figsize=(10,7))
    else:
        pass
    if distance == None:
        dist = 2
    else:
        dist = distance
    Y = Y/np.max(Y)
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.show()
    print("heihgt before peaks=",height,"distance = ",dist)
    _,_,peaks = Find_peaks(X,Y,pthreshold=height)
    #peaks,_ = find_peaks(Y,height=0.01,distance=dist)
    print("heihgt after peaks=",height,"distance = ",dist)
    #a = sorted(Y[peaks],reverse=True)
    print("peaks detected = ",len(peaks),"no. of lines = ",nopks)
    i = len(peaks)
    if nopks >= len(peaks):
        #print("height=",height)
        prev_len = len(peaks)
        H=float(height)
        while i < nopks:
            
            H -= 0.005
            #peaks, _ = find_peaks(Y, height=H, distance=dist)
            _,_,peaks = Find_peaks(X,Y,pthreshold=height)
            #print("new H = ",H)
            i = len(peaks)
            if i == prev_len:
                break
            prev_len = len(peaks)
        height= H		
        nopks=i
        #a = sorted(Y[peaks],reverse=True)
        #height = a[nopks-1]
        print("H = ",H,"height of reverse = ",height)
        #print("new nopks",nopks)
    else:
        print("if no change, nopks=",nopks)
        pass
        a = sorted(Y[peaks],reverse=True)
        height = a[nopks-1]
    print("Height for {} pks={}".format(nopks,height))
    _,_,pks = Find_peaks(X,Y,pthreshold=height)
    '''
    if display:
        for i,j in zip(X[pks],Y[pks]):
            ax.axvline(i,ymax=j,color=color,label=label)
            ax.text(x=i,y=j+0.01,s="{:.2f}".format(i),rotation=90,color=color)
            plt.show()
    '''

    return pks
   
#############################################################################
#############################################################################

def zipplot(X,Y,ax=None,color="k",label=None):
    Y = Y/np.max(Y)
    

    if ax==None:
        fig,ax = plt.subplots()
        ax.set_ylim(0,None)
    else:
        pass
        
    if label is None:
        label = " "

    for i, (x, y) in enumerate(zip(X, Y)):
        # Draw vertical line
        ax.axvline(x=x, ymax=y, color=color, label=label if i == 0 else "")
        # Place text slightly above the line
        ax.text(x=x, y=y + 0.01, s="{:.2f}".format(x), rotation=90, color="g")
    if label != " ":
        ax.legend()
              
    
    #plt.show()
 
 
   

def zipplot_norm(X,Y,ax=None,color="k",label=None,norm_x =None):
    if norm_x != None:
        closest_value= []
        for x in X:
            closest_idx = np.abs(norm_x - x)
            closest_value.append(closest_idx)
        Y = Y/Y[np.argmin(closest_value)]
    else:
        Y = Y/np.max(Y)
    if label==None:
        label=" "
    else:
        pass

    if ax==None:
        fig,ax = plt.subplots()
        ax.set_ylim(0,None)
    else:
        pass

    for x,y in zip(X,Y):
        ax.axvline(x=x,ymax = y,color=color,label=label)
        ax.text(x=x,y=y+0.01,s="{:.2f}".format(x),rotation=90,color="g")
        
#############################################################################
#############################################################################

def line_slider(xpt,ypt,file):
    ## making middel element of xpt as origin
    xpt = xpt-xpt[len(xpt)//2]
    print("xpt = ",xpt)
    file = os.path.join(scpath,"resources/arc/",file)
    print(file)
    ARC = Table.read(file,format="ascii",names=("wave","flux"))
    wave = ARC["wave"].value
    flux = ARC["flux"].value
    wave = wave[::-1]
    flux = flux[::-1]
    #scale = np.max(ypt)/np.max(flux)
    wbin = np.where((wave >5000)&(wave<8000))
    wave = wave[wbin]
    flux = flux[wbin] #*scale

    print("WAVE = ",wave)
    print("Flux = ",flux)

    
    
    ypt = ypt/np.max(ypt)
    dispersion = 1
    offset = 5000
    fig,ax = plt.subplots(figsize=(15,8))
    ax.set_ylim(0,None)

    #for i,j in zip(wave,flux):
     #   ax.axvline(i,ymax=j)
    zipplot(wave,flux,ax,color="r")    
    #plt.plot(wave,flux)
    Xpt = offset + xpt*dispersion
    #fit = np.poly1d(np.polyfit(xpt,Xpt,6))
    #Xpt = fit(Xpt)
    

    
    [lines] = ax.plot(Xpt,ypt)

    fig.subplots_adjust(left=0.25,bottom=0.25)
    axoffset = fig.add_axes([0.25,0.1,0.65,0.03])
    offset_sdr = Slider(ax=axoffset,label="offset",valmin=1000,valmax=10000,valinit=offset)

    fig.subplots_adjust(left=0.25, bottom=0.25)
    axdisp_A = fig.add_axes([0.1,0.25,0.03,0.65])
    A_disp_sdr = Slider(ax=axdisp_A,label="dispersion",valmin=-5,valmax=5,valinit=dispersion,
                      orientation="vertical")
    axdisp_B = fig.add_axes([0.05,0.25,0.03,0.65])
    B_disp_sdr = Slider(ax=axdisp_B,label="dispersion",valmin=-1,valmax=1,valinit=0.0,
                      orientation="vertical")
    #pks_ob = linepckr(wave,flux,ax=ax1,display=True)
    
    def update(val):
        new_xpt = offset_sdr.val+ xpt*(A_disp_sdr.val+B_disp_sdr.val)
        #ax.plot(new_xpt,ypt)
        lines.set_xdata(new_xpt)
        fig.canvas.draw_idle()
        


    offset_sdr.on_changed(update)
    A_disp_sdr.on_changed(update)
    B_disp_sdr.on_changed(update)

    
    plt.show()
    disp = A_disp_sdr.val+B_disp_sdr.val
    return offset_sdr.val,disp



from matplotlib.widgets import Slider, Button

def line_slider_button(xpt, ypt, file):
    import matplotlib
    matplotlib.use('Qt5Agg')
    ## making middle element of xpt as origin
    xpt = xpt - xpt[len(xpt) // 2]
    print("xpt =", xpt)
    file = os.path.join(scpath, "resources/arc/", file)
    print(file)

    ARC = Table.read(file, format="ascii", names=("wave", "flux"))  ## the reference arc lamp spectrum
    wave = ARC["wave"].value[::-1]
    flux = ARC["flux"].value[::-1]

    wbin = np.where((wave > 3500) & (wave < 9000))
    wave = wave[wbin]
    flux = flux[wbin]
    
    print("WAVE =", wave)
    print("Flux =", flux)

    ypt = ypt / np.nanmax(ypt)
    dispersion = -3.35
    offset = 6326
    Xpt = offset + xpt * dispersion
    print(f" Xpt = {Xpt} | \n ypt= {ypt}")
    plt.plot(Xpt,ypt)
    plt.show()
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_ylim(0, None)
    
    zipplot(wave,flux,ax,color="r",label="Reference Spectrum")
    #zipplot_norm(wave,flux,ax,color="r",norm_x=4358)
    
    [lines] = ax.plot(Xpt, ypt,label="observed spectrum")
    ax.legend()

    fig.subplots_adjust(left=0.25, bottom=0.25)
    axoffset = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    offset_sdr = Slider(ax=axoffset, label="offset", valmin=1000, valmax=10000, valinit=offset)

    axdisp_A = fig.add_axes([0.1, 0.25, 0.03, 0.65])
    A_disp_sdr = Slider(ax=axdisp_A, label="dispersion", valmin=-5, valmax=5, valinit=dispersion,
                        orientation="vertical")
    axdisp_B = fig.add_axes([0.05, 0.25, 0.03, 0.65])
    B_disp_sdr = Slider(ax=axdisp_B, label="dispersion", valmin=-1, valmax=1, valinit=0.0,
                        orientation="vertical")
    # Create buttons for finer adjustments
    button_ax = fig.add_axes([0.8, 0.15, 0.1, 0.05])
    apply_button = Button(button_ax, 'Apply Values')
    
    ## Adding buttons for incrementing offset and dispersion
    increment_offset_button_ax = fig.add_axes([0.9, 0.01, 0.045, 0.03])
    increment_offset_button = Button(increment_offset_button_ax, 'Offset +', hovercolor='0.975')

    decrement_offset_button_ax = fig.add_axes([0.9, 0.04, 0.045, 0.03])
    decrement_offset_button = Button(decrement_offset_button_ax, 'Offset -', hovercolor='0.975')
    
    increment_disp_button_ax = fig.add_axes([0.03, 0.1, 0.045, 0.03])
    increment_disp_button = Button(increment_disp_button_ax, 'disp +', hovercolor='0.975')

    decrement_disp_button_ax = fig.add_axes([0.075, 0.1, 0.045, 0.03])
    decrement_disp_button = Button(decrement_disp_button_ax, 'Disp -', hovercolor='0.975')

    def update(val):
        print(f" the updated value being passed {offset_sdr.val} {A_disp_sdr.val + B_disp_sdr.val}")
        new_xpt = offset_sdr.val + xpt * (A_disp_sdr.val + B_disp_sdr.val)
        lines.set_xdata(new_xpt)
        fig.canvas.draw_idle()

    offset_sdr.on_changed(update)
    A_disp_sdr.on_changed(update)
    B_disp_sdr.on_changed(update)

    def increment_offset(event):
        current_offset = offset_sdr.val
        offset_sdr.set_val(current_offset + 5)  # increment by 100 units
    def decrement_offset(event):
        current_offset = offset_sdr.val
        offset_sdr.set_val(current_offset - 5)  # increment by 100 units

    def increment_disp(event):
        current_disp = A_disp_sdr.val + B_disp_sdr.val
        A_disp_sdr.set_val(A_disp_sdr.val + 0.01)  # increment A_disp by 0.1 units
    def decrement_disp(event):
        current_disp = A_disp_sdr.val + B_disp_sdr.val
        A_disp_sdr.set_val(A_disp_sdr.val - 0.01) 

    increment_offset_button.on_clicked(increment_offset)
    increment_disp_button.on_clicked(increment_disp)
    decrement_offset_button.on_clicked(decrement_offset)
    decrement_disp_button.on_clicked(decrement_disp)
    #plt.legend()
    plt.show(block=True)
    disp = A_disp_sdr.val + B_disp_sdr.val

    final_values = [None, None]
    def apply_and_close(event):
        final_values[0] = offset_sdr.val
        final_values[1] = disp
        #plt.close(fig)
    
    apply_button.on_clicked(apply_and_close)
    #plt.show()
    
    # Return the results after window closes
    return final_values[0],final_values[1]
    #return offset_sdr.val, disp


def XFIT(X,offset,disp):
    new_X = offset +(X-X[len(X)//2])*disp
    return new_X
'''
def line_slider_button(xpt, ypt, file):
    """Interactive wavelength solution adjustment with proper value return"""
    #import matplotlib
    #matplotlib.use('Qt5Agg')
    plt.ion()
    # Center the x-points
    xpt = xpt - xpt[len(xpt) // 2]
    
    # Load reference spectrum
    file = os.path.join(scpath, "resources/arc/", file)
    ARC = Table.read(file, format="ascii", names=("wave", "flux"))
    wave = ARC["wave"].value[::-1]
    flux = ARC["flux"].value[::-1]
    
    # Filter wavelength range
    wbin = np.where((wave > 3500) & (wave < 9000))
    wave = wave[wbin]
    flux = flux[wbin]
    
    # Normalize observed spectrum
    ypt = ypt / np.nanmax(ypt)
    
    # Initial parameters
    dispersion = -3.35
    offset = 6326
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_ylim(0, None)
    
    # Plot reference and observed spectra
    zipplot(wave, flux, ax, color="r", label="Reference Spectrum")
    [lines] = ax.plot(offset + xpt * dispersion, ypt, label="Observed Spectrum")
    ax.legend()

    # Adjust layout for sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)
    
    # Create sliders
    axoffset = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    offset_sdr = Slider(ax=axoffset, label="Offset", valmin=1000, valmax=10000, valinit=offset)

    axdisp_A = fig.add_axes([0.1, 0.25, 0.03, 0.65])
    A_disp_sdr = Slider(ax=axdisp_A, label="Dispersion (coarse)", valmin=-5, valmax=5, 
                       valinit=dispersion, orientation="vertical")

    axdisp_B = fig.add_axes([0.05, 0.25, 0.03, 0.65])
    B_disp_sdr = Slider(ax=axdisp_B, label="Dispersion (fine)", valmin=-1, valmax=1, 
                       valinit=0.0, orientation="vertical")

    # Create buttons
    button_axes = {
        'offset_inc': fig.add_axes([0.9, 0.01, 0.045, 0.03]),
        'offset_dec': fig.add_axes([0.9, 0.04, 0.045, 0.03]),
        'disp_inc': fig.add_axes([0.03, 0.1, 0.045, 0.03]),
        'disp_dec': fig.add_axes([0.075, 0.1, 0.045, 0.03])
    }
    
    buttons = {
        'offset_inc': Button(button_axes['offset_inc'], 'Offset +'),
        'offset_dec': Button(button_axes['offset_dec'], 'Offset -'),
        'disp_inc': Button(button_axes['disp_inc'], 'Disp +'),
        'disp_dec': Button(button_axes['disp_dec'], 'Disp -')
    }
    
    callbacks = {
        'offset': offset_sdr.on_changed(update),
        'disp_A': A_disp_sdr.on_changed(update),
        'disp_B': B_disp_sdr.on_changed(update),
        'btn_offset_inc': increment_offset_button.on_clicked(increment_offset),
        'btn_offset_dec': decrement_offset_button.on_clicked(decrement_offset),
        'btn_disp_inc': increment_disp_button.on_clicked(increment_disp),
        'btn_disp_dec': decrement_disp_button.on_clicked(decrement_disp)
    }
    
    # Update function
    def update(val):
        new_xpt = offset_sdr.val + xpt * (A_disp_sdr.val + B_disp_sdr.val)
        lines.set_xdata(new_xpt)
        fig.canvas.draw_idle()

    # Connect sliders
    for slider in [offset_sdr, A_disp_sdr, B_disp_sdr]:
        slider.on_changed(update)

    # Button callbacks
    def adjust_offset(step):
        offset_sdr.set_val(offset_sdr.val + step)
    
    def adjust_disp(step):
        A_disp_sdr.set_val(A_disp_sdr.val + step)

    buttons['offset_inc'].on_clicked(lambda x: adjust_offset(5))
    buttons['offset_dec'].on_clicked(lambda x: adjust_offset(-5))
    buttons['disp_inc'].on_clicked(lambda x: adjust_disp(0.01))
    buttons['disp_dec'].on_clicked(lambda x: adjust_disp(-0.01))

    # Create a button to confirm selection
    #confirm_ax = fig.add_axes([0.8, 0.9, 0.1, 0.05])
    #confirm_button = Button(confirm_ax, 'Confirm', color='lightgoldenrodyellow')

    confirm_ax = fig.add_axes([0.8, 0.9, 0.1, 0.05])
    confirm_btn = Button(confirm_ax, 'Confirm')
    confirm_btn.on_clicked(lambda x: plt.close(fig))
     Variable to store results
    result = {'offset': None, 'dispersion': None}
    
    def on_confirm(event):
        result['offset'] = offset_sdr.val
        result['dispersion'] = A_disp_sdr.val + B_disp_sdr.val
        plt.close(fig)
    
    #confirm_button.on_clicked(on_confirm)
    
    plt.show(block=True)
    
    # Return the results after window closes
    return result['offset'], result['dispersion']
'''

def line_slider_button_old(xpt, ypt, arc_file, initial_offset=6326, initial_disp=-3.35):
    """Interactive wavelength solution estimation with sliders.
    
    Args:
        xpt: Pixel positions (zero-centered)
        ypt: Spectrum flux values
        arc_file: Path to reference arc spectrum file
        initial_offset: Starting guess for offset
        initial_disp: Starting guess for dispersion
        
    Returns:
        tuple: (final_offset, final_dispersion)
    """
    # Normalize input spectrum
    xpt = xpt - xpt[len(xpt) // 2]  # Center on middle pixel
    ypt = ypt / np.nanmax(ypt)
    
    # Load reference spectrum
    try:
        arc_data = Table.read(arc_file, format="ascii", names=("wave", "flux"))
        wave = arc_data["wave"].data
        flux = arc_data["flux"].data
        
        # Filter reasonable wavelength range
        mask = (wave > 3500) & (wave < 9000)
        wave = wave[mask]
        flux = flux[mask]
    except Exception as e:
        raise ValueError(f"Could not load arc file {arc_file}: {str(e)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_ylim(0, None)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalized Flux")
    
    # Plot reference spectrum
    ax.plot(wave, flux, 'r-', label="Reference Spectrum")
    
    # Plot initial observed spectrum
    initial_wave = initial_offset + xpt * initial_disp
    obs_line, = ax.plot(initial_wave, ypt, 'b-', label="Observed Spectrum")
    ax.legend()
    
    # Create slider axes
    fig.subplots_adjust(left=0.25, bottom=0.25)
    ax_offset = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax_disp = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    
    # Create sliders with reasonable ranges
    offset_slider = Slider(ax_offset, "Offset (Å)", 1000, 10000, initial_offset)
    disp_slider = Slider(ax_disp, "Dispersion (Å/pix)", -5, 5, initial_disp)
    
    # Create buttons for finer adjustments
    button_ax = fig.add_axes([0.8, 0.15, 0.1, 0.05])
    apply_button = Button(button_ax, 'Apply Values')
    
    # Update function
    def update(val):
        new_wave = offset_slider.val + xpt * disp_slider.val
        obs_line.set_xdata(new_wave)
        fig.canvas.draw_idle()
    
    # Connect sliders
    offset_slider.on_changed(update)
    disp_slider.on_changed(update)
    
    # Button to close and return values
    final_values = [None, None]
    def apply_and_close(event):
        final_values[0] = offset_slider.val
        final_values[1] = disp_slider.val
        plt.close(fig)
    
    apply_button.on_clicked(apply_and_close)
    
    plt.show()
    
    if final_values[0] is None:
        raise RuntimeError("Slider dialog was closed without applying values")
    
    return final_values[0], final_values[1]


def line_slider_button2( xpt, ypt, file):
    # NOTE: this function is not called anywhere in the library or GUI
    # (verified during the public-release audit). It also has a
    # pre-existing bug independent of this refactor: the `return
    # new_offset, new_dispersion` below references names that only
    # exist inside a commented-out nested function further down, so
    # calling this as written would raise NameError. Left as dead code
    # rather than guessing at the intended fix -- if you need this
    # function, it likely should return offset_slider.val / disp_slider.val
    # (the values apply_and_close actually sets) instead.
    """Version designed to work within Qt applications"""
    #from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
    #from matplotlib.figure import Figure
    
    # Create figure and canvas
    fig = plt.figure(figsize=(15, 8))
    #canvas = FigureCanvasQTAgg(fig)
    ax = fig.add_subplot(111)
    '''
    # Add to a new dialog
    dialog = QDialog(parent_widget)
    layout = QVBoxLayout(dialog)
    layout.addWidget(NavigationToolbar2QT(canvas, dialog))
    layout.addWidget(canvas)
    '''
    # [Rest of your plotting code using ax instead of plt...]

    ## making middle element of xpt as origin
    xpt = xpt - xpt[len(xpt) // 2]
    print("xpt =", xpt)
    file = os.path.join(scpath, "resources/arc/", file)
    print(file)

    ARC = Table.read(file, format="ascii", names=("wave", "flux"))  ## the reference arc lamp spectrum
    wave = ARC["wave"].value[::-1]
    flux = ARC["flux"].value[::-1]

    wbin = np.where((wave > 3500) & (wave < 9000))
    wave = wave[wbin]
    flux = flux[wbin]
    
    print("WAVE =", wave)
    print("Flux =", flux)

    ypt = ypt / np.nanmax(ypt)
    dispersion = -3.35
    offset = 6326
    Xpt = offset + xpt * dispersion
    print(f" Xpt = {Xpt} | \n ypt= {ypt}")
    #plt.plot(Xpt,ypt)
    #plt.show()
    #fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_ylim(0, None)
    
    zipplot(wave,flux,ax,color="r",label="Reference Spectrum")
    #zipplot_norm(wave,flux,ax,color="r",norm_x=4358)
    
    [lines] = ax.plot(Xpt, ypt,label="observed spectrum")
    ax.legend()

    fig.subplots_adjust(left=0.25, bottom=0.25)
    axoffset = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    offset_sdr = Slider(ax=axoffset, label="offset", valmin=1000, valmax=10000, valinit=offset)

    axdisp_A = fig.add_axes([0.1, 0.25, 0.03, 0.65])
    A_disp_sdr = Slider(ax=axdisp_A, label="dispersion", valmin=-5, valmax=5, valinit=dispersion,
                        orientation="vertical")
    axdisp_B = fig.add_axes([0.05, 0.25, 0.03, 0.65])
    B_disp_sdr = Slider(ax=axdisp_B, label="dispersion", valmin=-1, valmax=1, valinit=0.0,
                        orientation="vertical")

    ## Adding buttons for incrementing offset and dispersion
    increment_offset_button_ax = fig.add_axes([0.9, 0.01, 0.045, 0.03])
    increment_offset_button = Button(increment_offset_button_ax, 'Offset +', hovercolor='0.975')

    decrement_offset_button_ax = fig.add_axes([0.9, 0.04, 0.045, 0.03])
    decrement_offset_button = Button(decrement_offset_button_ax, 'Offset -', hovercolor='0.975')
    
    increment_disp_button_ax = fig.add_axes([0.03, 0.1, 0.045, 0.03])
    increment_disp_button = Button(increment_disp_button_ax, 'disp +', hovercolor='0.975')

    decrement_disp_button_ax = fig.add_axes([0.075, 0.1, 0.045, 0.03])
    decrement_disp_button = Button(decrement_disp_button_ax, 'Disp -', hovercolor='0.975')

    def update(val):
        new_xpt = offset_sdr.val + xpt * (A_disp_sdr.val + B_disp_sdr.val)
        lines.set_xdata(new_xpt)
        fig.canvas.draw_idle()

    offset_sdr.on_changed(update)
    A_disp_sdr.on_changed(update)
    B_disp_sdr.on_changed(update)

    def increment_offset(event):
        current_offset = offset_sdr.val
        offset_sdr.set_val(current_offset + 5)  # increment by 100 units
    def decrement_offset(event):
        current_offset = offset_sdr.val
        offset_sdr.set_val(current_offset - 5)  # increment by 100 units

    def increment_disp(event):
        current_disp = A_disp_sdr.val + B_disp_sdr.val
        A_disp_sdr.set_val(A_disp_sdr.val + 0.01)  # increment A_disp by 0.1 units
    def decrement_disp(event):
        current_disp = A_disp_sdr.val + B_disp_sdr.val
        A_disp_sdr.set_val(A_disp_sdr.val - 0.01) 

    increment_offset_button.on_clicked(increment_offset)
    increment_disp_button.on_clicked(increment_disp)
    decrement_offset_button.on_clicked(decrement_offset)
    decrement_disp_button.on_clicked(decrement_disp)
    
    
    # Connect confirm button
    #confirm_btn = QPushButton("Confirm", dialog)
    #confirm_btn.clicked.connect(dialog.accept)
    #layout.addWidget(confirm_btn)
    
    ##    if dialog.exec_() == QDialog.Accepted:
    ##        return offset_sdr.val, A_disp_sdr.val + B_disp_sdr.val
    ##    return None, None
    #return offset_sdr.val, A_disp_sdr.val + B_disp_sdr.val

    #result = {'offset': None, 'dispersion': None}
    '''
    def on_confirm(event):
        new_offset = offset_sdr.val
        new_dispersion = A_disp_sdr.val + B_disp_sdr.val
        plt.close(fig)
    
    confirm_button.on_clicked(on_confirm)
    '''
    final_values = [None, None]
    def apply_and_close(event):
        final_values[0] = offset_slider.val
        final_values[1] = disp_slider.val
        plt.close(fig)
    
    apply_button.on_clicked(apply_and_close)
    plt.show(block=True)
    
    # Return the results after window closes
    return new_offset, new_dispersion

def Wavesolution(X,Y,file,offset,disp):
    file = os.path.join(scpath,"resources/arc/",file)
    print(file)
    ARC = Table.read(file,format="ascii",names=("wave","flux"))
    wave = ARC["wave"].value
    flux = ARC["flux"].value

    #scale = np.max(ypt)/np.max(flux)
    wbin = np.where((wave >4000)&(wave<7300))
    wave = wave[wbin]
    flux = flux[wbin]
    zipplot(wave,flux,label="REF")
    plt.show()
    
    Y= Y/np.max(Y)
    new_xpt = XFIT(X,offset,disp)
    print("new_xpt=",new_xpt)
    if new_xpt[0] > new_xpt[-1]:
        print(" wave list is in decreasing order, inverting the list")
        new_xpt = new_xpt[::-1]
        #Y = Y[::-1]
    elif new_xpt[0] < new_xpt[-1]:
        print("wave list is in increasing order")
        
    #new_xpt = new_xpt[::-1]
    Y = Y[::-1]
    fig,ax = plt.subplots(figsize=(10,6))
    pks_ob = linepckr(new_xpt,Y,ax=ax,display=True,color="r",label="ob",nopks=10)
    ob_wave = new_xpt[pks_ob]
    print("ob_wave = ",ob_wave,"pks = ",pks_ob)
    #plt.show()
    pks_ref = linepckr(wave,flux,ax=ax,display=True,label = "ref",nopks=10)
    ref_wave = wave[pks_ref]
    print("ref_wave =",ref_wave,"pks = ",pks_ref)
    plt.legend()
    plt.show()
    #disp =abs(disp)
    
    print(" doing curve fit optimize to estimate the offset and dispersion")
    popt,pcov = curve_fit(XFIT,pks_ob,ob_wave,p0=[offset,disp],bounds=([3000,-3],[7000,3]))

    Nxpt = XFIT(X,*popt)
    #zipplot(Nxpt,Y,label="new")
    #fig2,ax2 = plt.subplots()
    #plt.show()
    ### for residuals

    New_xpt = XFIT(X,*popt)

    new_pks = linepckr(New_xpt,Y,ax=ax,display=True,color="r",label="ob")
    New_wave = New_xpt[new_pks]
    plt.show()
    pair = closeval(New_wave,ref_wave)


    A=[]
    D=[]
    for i in pair:
        d = i[0] - i[1]
        D.append(d)
        A.append(i[0])
    plt.scatter(A,D)
    plt.show()
    

    return popt,pair
    

    

    
    
     
    
        
    
    


    #return coeff of the function we defined and deg if doing non-linear fitting (here offset,dispersion,deg)
    #and this function will be applied to the spectral_axis of source spectrum
    
    



#############################################################################
#############################################################################


## trying for non-linear equation for wavelength calibration


def line_slider2(xpt,ypt,file,order=2,fig=None,ax=None):

    """
    
    Note: need to plot a fig,ax subplot :| fig, ax = plt.subplots(figsize=(15,8)) | and
    if there are multiple plots then better to run plt.close(fig) and plt.show(block=True) once 
    this function is completed
    
    
    """

    xpt = xpt-xpt[len(xpt)//2]
    
    file = os.path.join(scpath,"resources/arc/",file)
    print(file)
    ARC = Table.read(file,format="ascii",names=("wave","flux"))
    wave = ARC["wave"].value
    flux = ARC["flux"].value
    wave = wave[::-1]
    flux = flux[::-1]
    #scale = np.max(ypt)/np.max(flux)
    wbin = np.where((wave >3000)&(wave<7200))
    wave = wave[wbin]
    flux = flux[wbin] #*scale

    
    
    #ypt = ypt/np.max(ypt)
    dispersion = 1
    offset = 5000
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(15,8))

    
    zipplot(wave,flux,ax,color="r")    
    #plt.plot(wave,flux)
    Xpt = offset + xpt*dispersion
    #fit = np.poly1d(np.polyfit(xpt,Xpt,6))
    #Xpt = fit(Xpt)
    

    pks = linepckr(Xpt,ypt,nopks=15,display=False,ax=ax)
    lines = ax.scatter(Xpt[pks],ypt[pks])
    [Lines] = ax.plot(Xpt,ypt)
    
    sliders = []
    fig.subplots_adjust(left=0.25, bottom=0.25)
    for i in range(order):
       ax_coeff = fig.add_axes([0.1+i*0.05,0.2,0.03,0.65])
       slider = Slider(ax=ax_coeff,label=f"Coeff {i+1}",valmin=int(-24/(i+1)),valmax=int(24/(i+1)),valinit = 0,orientation="vertical")
       sliders.append(slider)
    
    
    fig.subplots_adjust(left=0.25,bottom=0.25)
    axoffset = fig.add_axes([0.25,0.1,0.65,0.03])
    offset_sdr = Slider(ax=axoffset,label="offset",valmin=1000,valmax=10000,valinit=offset)
    '''
    fig.subplots_adjust(left=0.25, bottom=0.25)
    A_axdisp = fig.add_axes([0.1,0.25,0.03,0.65])
    A_disp_sdr = Slider(ax=A_axdisp,label="dispersion_A",valmin=-0.001,valmax=0.001,valinit=dispersion,
                      orientation="vertical")

    fig.subplots_adjust(left=0.25, bottom=0.25)
    B_axdisp = fig.add_axes([0.15,0.25,0.03,0.65])
    B_disp_sdr = Slider(ax=B_axdisp,label="dispersion_B",valmin=-5,valmax=5,valinit=dispersion,
                      orientation="vertical")
    #pks_ob = linepckr(wave,flux,ax=ax1,display=True)
    '''
    #plt.show(block=False)
    def update(val):
        coeffs = [s.val for s in sliders]
        new_xpt = np.polyval(coeffs[::-1],xpt)+offset_sdr.val
        #new_xpt = offset_sdr.val+ xpt*B_disp_sdr.val +(xpt**2)*A_disp_sdr.val
        #ax.plot(new_xpt,ypt)
        Lines.set_xdata(new_xpt)
        pks1 = linepckr(new_xpt,ypt,nopks=15,display=False,ax=ax)
        lines.set_offsets(np.column_stack((new_xpt[pks1], ypt[pks1])))
        #plt.clf()
        #plt.show(block=True)
        #ax.relim()
        #ax.autoscale_view()
        fig.canvas.draw_idle()
    
        



    for slider in sliders:
        slider.on_changed(update)
    offset_sdr.on_changed(update)
    #B_disp_sdr.on_changed(update)
    #A_disp_sdr.on_changed(update)
	
    #plt.clf()
    
    plt.show()
    coeffs = [s.val for s in sliders]
    new_xpt = np.polyval(coeffs[::-1],xpt)+offset_sdr.val
    #plt.close(fig)
    return new_xpt,coeffs,offset_sdr.val
    
#########################################################################################################
#########################################################################################################

########################################################################################################
##################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from astropy.table import Table

def line_slider3(xpt, ypt, file, order=2):
    xpt = xpt - xpt[len(xpt)//2]
    
    # Assuming `scpath` is defined earlier in your code
    file = os.path.join(scpath, "resources/arc/", file)
    print(file)
    
    # Read the spectral data from the file
    ARC = Table.read(file, format="ascii", names=("wave", "flux"))
    wave = ARC["wave"].value * 10
    flux = ARC["flux"].value
    wave = wave[::-1]
    flux = flux[::-1]
    
    # Consider wavelengths between 5000 and 7200 Angstroms
    wbin = np.where((wave > 5000) & (wave < 7200))
    wave = wave[wbin]
    flux = flux[wbin]
    
    # Normalize the spectrum
    ypt = ypt / np.max(ypt)
    
    # Initialize parameters
    offset = 5000
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(wave, flux, color="r")
    lines, = ax.plot([], [], marker='o')  # Placeholder for the plot line
    
    # Create sliders for coefficients
    sliders = []
    for i in range(order):
        ax_coeff = fig.add_axes([0.1 + i * 0.05, 0.02, 0.03, 0.03])
        slider = Slider(ax=ax_coeff, label=f"Coeff {i+1}", valmin=-10, valmax=10, valinit=0)
        sliders.append(slider)
    
    # Update function for sliders
    def update(val):
        coeffs = [s.val for s in sliders]
        new_xpt = np.polyval(coeffs[::-1], xpt) + offset
        lines.set_data(new_xpt, ypt)
        fig.canvas.draw_idle()
    
    for slider in sliders:
        slider.on_changed(update)
    
    plt.show()
    
    
    
#########################################################################################################
#########################################################################################################

def spectr_slider(data,vmin=None,vmax=None,xlim=1300,nb=10,cen=500):
    #print("data = ",data)
    if isinstance(data,ccdproc.CCDData):
        data = data.data
    elif isinstance(data,type([])):
        data=data
    mn = np.nanmean(data)
    stdd = np.nanstd(data)
    #print("mn = ",mn,"std = ",stdd)
    fig,ax = plt.subplots()
    if vmin==None:
        vmin = mn-2*stdd
    if vmax == None:
        vmax = mn+2*stdd
    Im = ax.imshow(data,cmap="gray",vmin = vmin ,vmax =vmax,origin="lower")
    
    fig.subplots_adjust(left=0.25, bottom=0.25)
    axvmin = fig.add_axes([0.25,0.1,0.65,0.03])
    vmin_sdr = Slider(ax=axvmin,label = "vmin",valmin=mn-4*stdd,valmax =mn,valinit=vmin)

    axvmax = fig.add_axes([0.1,0.25,0.03,0.65])
    vmax_sdr = Slider(ax=axvmax,label = "vmax",valmin=mn,valmax =mn+4*stdd,valinit=vmax,
                      orientation="vertical")
    def update(val):
        
        #ax.imshow(data,vmin=vmin_sdr.val,vmax = vmax_sdr.val)
        Im.set_clim(vmin = vmin_sdr.val,vmax = vmax_sdr.val)
        fig.canvas.draw_idle()
        
    vmin_sdr.on_changed(update)
    vmax_sdr.on_changed(update)
    
    
    
    
    
##########################################################################################
##########################################################################################

def cal_rmse(x,X,k=2,display=False):
    """
    Parameter
    
    x : 1D array, observed wavelength
    X : 1D array, reference wavelength
    k : no. of contraints
    
    Return
    
    reduced chi sq value, pairs of match wavelengths
    """
    
    
    a,xx,XX = closeval2(x,X)
    #print(f' the pairs are {a}')
    num=[]
    obswave = []
    refwave=[]
    
    for i in range(len(a)):
        diff =  ((a[i][0]-a[i][1])**2)/a[i][0]
        num.append(diff)
        obswave.append(a[i][0])
        refwave.append(a[i][1])

    mse = np.sum(num)
    rchisq = np.sqrt(mse)/(len(xx)-k)
    print("rchisq = ",rchisq)
    if display==True:
        plt.scatter(obswave,refwave)
        plt.scatter(refwave,refwave)
        plt.plot(refwave,refwave,linestyle="dashed",alpha=0.5,color="gray")
        plt.xlabel("observed")
        plt.ylabel("Reference")

    return rchisq,obswave,refwave


#############################################################################
#############################################################################   

def linefit2(wave,flux,sciflux,file,nopks=15,height=0.97,func="legd"):


    """
    Parameters:
    wave : 1D array, observed wavelength
    flux : 1D array, counts of the spectrum
    Refwave: 1D array, reference wavelength
    nopks : no, of peaks
    height : minumum height for the peaks to be found
    func : function for wavelength calibration
    
    Returns:
    
    fitted coefficent/function,matched wavelength,pairs of match wavelengths, new wavelength array
    """
    file = os.path.join(scpath, "resources/arc/", file)
    print(f' the arc file name is {file}')
    ARC = Table.read(file,format="ascii",names=("wave","flux"))
    Refwave = ARC["wave"].value
    flux_arc = ARC["flux"].value
    
    #wave = wave[::-1]   #3ops
    
    if wave[0] > wave[-1]:
        w_lower = wave[-1]
        w_upper = wave[0]
    else:
        w_lower = wave[0]
        w_upper = wave[-1]
    
    print(f' the lower wave {w_lower} and upper wave {w_upper}')
    refbin = np.where((Refwave > w_lower) & (Refwave <= w_upper))
    Refwave = Refwave[refbin]
    flux_arc = flux_arc[refbin]
    '''
    plt.plot(wave,flux)
    plt.title(" in input")
    plt.show()
    
    plt.plot(Refwave,flux_arc)
    plt.show()
    '''
    

    #if wave[0] <wave[-1]:
     #   wave = wave[::-1]
    #else:
     #   pass
    print("obs wave=",wave)
    WAVE = wave
    flux = flux[::-1]
    FLUX = flux#[::-1]
    #plt.plot(WAVE,FLUX)
    #plt.title(" the flux in linefit")
    #plt.show()
    #print(f' the len of wave {len(wave)} and flux {len(flux)} in linefit' )
    #flux = FLUX
    #wbin = np.where((wave > 3000) & (wave < 7500))
    #wave = wave[wbin]
    #flux = flux[wbin]
    #plt.plot(wave,flux)
    #plt.title("in linefit hehe")
    #plt.show()
    zipplot(wave,flux,color="r")
    pks = linepckr(wave,FLUX,nopks=nopks)
    pixel =np.arange(1,len(wave)+1)
    print(f' the len of pks {len(pks)} and wave : {len(wave)} \n and the pks is {pks}')
    xcxc = wave[pks]
    cxcx = flux[pks]
    print("peaks wave = ",xcxc)
    #zipplot(xcxc,cxcx)
    #zipplot(Refwave,flux_arc)
    
    
    #w_lower = wave[0]
    #w_upper = wave[-1]
    #nbin = np.where((Refwave > w_lower)&(Refwave < w_upper))
    
    #plt.show(block=True)
    print(f' the input for cal_rmse = selected Wave: {xcxc} and Refwave :{Refwave}')
    rmse,obswave,refwave = cal_rmse(wave[pks],Refwave)
    print("obswave=",obswave,"refwave=",refwave)
    xpix = np.where(np.isin(wave,obswave))
    PIXEL = pixel[xpix]
    print(f' the xpix = {xpix} and PIXEL = {PIXEL}')
    i=0
    
    if func=="poly":
    
        while rmse > 0.002:
            i += 1
            print(f"i={i}, len of. PIXEL={len(PIXEL)} ,pixel:{len(pixel)}, refwave = {len(refwave)}" )
            fit = np.polyfit(PIXEL,refwave,i)
            fpt = np.polyval(fit,PIXEL)
            rmse,obswave,refwave = cal_rmse(fpt,Refwave,k=int(len(fit)))
            print("done")
        Newwave = np.polyval(fpt,pixel)
        print("len of fpt=",len(fpt),"len of WAVE=",len(WAVE))
        #new_wave = np.interp(WAVE,np.arange(len(fpt)),fpt)
        #new_wave = np.interp(WAVE,pks,fpt)
        new_wave = np.polyval(fpt,pixel)
        
        
    if func=="interp1d":
        fit=None
        fpt=None
        while rmse > 0.01:
            i += 2
            print("i=",i)
            fit = interp1d(PIXEL,refwave,kind=2,fill_value='extrapolate')
            fpt = fit(PIXEL)
            print(f' i interp1d= fit:{fit} and fpt:{fpt}')
            rmse,obswave,refwave = cal_rmse(fpt,Refwave)
            #xpx = np.where(np.isin(wave,obswave))
    	 #PIXEL = pixel[xpx]
            print("done")
            Newwave = fit(wave)
        print("len of fpt=",len(fpt),"len of WAVE=",len(WAVE))
        #new_wave = np.interp(WAVE,np.arange(len(fpt)),fpt)
        new_wave = fit(pixel)
    
	
    if func=="spline":
        while rmse > 1e-3:
            i+=0.1
            print("i=",i)
            print(f' the obswave:{obswave} \n PIXEL:{PIXEL} \n and \n refwave {refwave}')
            
            fit = UnivariateSpline(PIXEL,refwave,ext=0,k=3,s=i) #bbox=[np.min(WAVE),np.max(WAVE)]
            fpt = fit(PIXEL)
            print(f' for i={i} | updated wave:{fpt}')
            rmse,obswave,refwave = cal_rmse(fpt,Refwave,k=3)
            print(f' the comparision wave:{wave} \n and \n new obswave {obswave}')
            #xpx = np.where(np.isin(wave,obswave))
            #print(f' the idx of the obswave {xpx}')
            print("done")
        Newwave = fit(PIXEL)
        new_wave = fit(pixel)
        
    if func=="spline3":
        while rmse > 1e-3:
            i+=0.1
            fit = CubicSpline(PIXEL,refwave)
            fpt = fit(PIXEL)
            print(f' for i={i} | updated wave:{fpt}')
            rmse,_,_= cal_rmse(fpt,refwave)
            print("done")
        Newwave = fit(PIXEL)
        new_wave = fit(pixel)
        
        
    if func=="legd":
        while rmse > 0.01:
            i+=1
            print("i =",i)
            fit = legd.fit(PIXEL,refwave,deg=i)
            fpt = fit(PIXEL)
            FPT = fit(pixel)
            #rmse,obswave,refwave= cal_rmse(fpt,refwave)
            pks = linepckr(FPT,FLUX,nopks=nopks,height=height)
            newwaveset = FPT[pks]
            rmse,obswave,refwave = cal_rmse(newwaveset,refwave,display=False)
            #PIXEL = pixel[pks]
            print(f' for i={i} | updated wave:{fpt} | pixel peaks {PIXEL}')
            
            print("done")
        Newwave = fit(PIXEL)
        new_wave = fit(pixel)
            

    '''
    ## check if the data points are within the limit
    
    if np.where(new_wave <2000):
        nbin = np.where(new_wave >2000)
        new_wave = new_wave[nbin]
        FLUx = sciflux[nbin]
    '''
    FLUx = sciflux[::-1]
    
    plt.figure()
    #plt.plot(refwave,refwave,linestyle="dashed",alpha=0.5,color="gray")
    #plt.scatter(Newwave,Newwave)
    #plt.title("After fitting the func")
    #plt.show()
    
    
    plt.plot(new_wave/1.043,FLUx)
    plt.title("new spectrum")
    plt.show()
    return new_wave,FLUx,fit

###############
def linepckr(X,Y,ax=None,color="k",label=None,nopks = None,height=0.97,distance=None,display=False):
    if nopks==None:
        nopks = 8
    else:
        nopks=nopks
    #print("nopks = ",nopks)
    if label==None:
        label=" "
    else:
        pass

    
    if distance == None:
        dist = 2
    else:
        dist = distance
    Y = Y/np.max(Y)
    #plt.scatter(X,Y)
    #plt.plot(X,Y)
    #plt.show()
    #print("heihgt before peaks=",height,"distance = ",dist)
    _,_,peaks = Find_peaks(X,Y,pthreshold=height)
    #peaks,_ = find_peaks(Y,height=0.01,distance=dist)
    #print("heihgt after peaks=",height,"distance = ",dist)
    #a = sorted(Y[peaks],reverse=True)
    print("peaks detected = ",len(peaks),"no. of lines = ",nopks)
    i = len(peaks)
    H=float(height)
    if nopks >= len(peaks):
        #print("height=",height)
        prev_len = len(peaks)
        
        while i < nopks:
            
            H -= 0.01
            #peaks, _ = find_peaks(Y, height=H, distance=dist)
            #print("new H = ",H)
            _,_,peaks = Find_peaks(X,Y,pthreshold=H)
            #print("new H = ",H)
            i = len(peaks)
            if i == prev_len:
                break
            prev_len = len(peaks)
        Height= H		
        nopks=i
        #a = sorted(Y[peaks],reverse=True)
        #height = a[nopks-1]
        print("H = ",H,"height of reverse = ",height," len of pks after refind :",len(peaks))
        #print("new nopks",nopks)
    else:
        print("if no change, nopks=",nopks)
        pass
        #a = sorted(Y[peaks],reverse=True)
        #height = a[nopks-1]
    #print("Height for {} pks={}".format(nopks,height))
    pixcent,wavcent,pks = Find_peaks(X,Y,pthreshold=Height)
    '''
    if display:
        if ax==None:
            fig,ax = plt.subplots(figsize=(10,7))
        else:
            pass
        for i,j in zip(X[pks],Y[pks]):
            ax.axvline(i,ymax=j,color=color,label=label)
            ax.text(x=i,y=j+0.01,s="{:.2f}".format(i),rotation=90,color=color)
            plt.show()
    '''

    return pixcent,wavcent,pks

def select_pts2(x, y, X):
    while True:
        print(f' idx \t pixel \t obswave \t refwave')
        for idx, (i, j, k) in enumerate(zip(x, y, X), 1):
            print(f' {idx} \t {i} \t {j} \t {k}')

        try:
            idx_input = int(input("Enter the idx number to remove (or 0 to finish): "))
            if idx_input == 0:
                break  # Exit the loop if 0 is entered
            elif 1 <= idx_input <= len(x):
                # Adjust index for zero-based indexing
                del x[idx_input - 1], y[idx_input - 1], X[idx_input - 1]
            else:
                print(f"Please enter a valid index between 1 and {len(x)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    return x, y, X
    
def select_pts(x, y, X):
    print(f' idx \t pixel \t obswave \t refwave')
    for idx, (i, j, k) in enumerate(zip(x, y, X), 1):
        print(f' {idx} \t {i} \t {j} \t {k}')

    try:
        # Take multiple indices input at once, separated by spaces or commas
        indices_input = input("Enter the idx numbers to remove, separated by spaces or commas: ")
        # Parse the input into a list of integers and adjust for zero-based indexing
        indices_to_remove = sorted([int(i) - 1 for i in indices_input.replace(',', ' ').split()], reverse=True)
        
        # Validate indices and remove elements
        if all(0 <= idx < len(x) for idx in indices_to_remove):
            x = [i for ix, i in enumerate(x) if ix not in indices_to_remove]
            y = [j for iy, j in enumerate(y) if iy not in indices_to_remove]
            X = [k for iX, k in enumerate(X) if iX not in indices_to_remove]
        else:
            print(f"Please enter valid indices between 1 and {len(x)}.")
            
    except ValueError:
        print("Invalid input. Please enter numbers only, separated by spaces or commas.")
    
    return np.array(x), np.array(y), np.array(X)


    
def linefit(wave,flux,sciflux,file,nopks=15,height=0.97,func="legd"):


    """
    Parameters:
    wave : 1D array, observed wavelength
    flux : 1D array, counts of the spectrum
    Refwave: 1D array, reference wavelength
    nopks : no, of peaks
    height : minumum height for the peaks to be found
    func : function for wavelength calibration
    
    Returns:
    
    fitted coefficent/function,matched wavelength,pairs of match wavelengths, new wavelength array
    """
    file = os.path.join(scpath, "resources/arc/", file)
    print(f' the arc file name is {file}')
    ARC = Table.read(file,format="ascii",names=("wave","flux"))
    Refwave = ARC["wave"].value
    flux_arc = ARC["flux"].value
    
    #wave = wave[::-1]   #3ops
    
    if wave[0] > wave[-1]:
        w_lower = wave[-1]
        w_upper = wave[0]
    else:
        w_lower = wave[0]
        w_upper = wave[-1]
    
    print(f' the lower wave {w_lower} and upper wave {w_upper}')
    refbin = np.where((Refwave > w_lower) & (Refwave <= w_upper))
    Refwave = Refwave[refbin]
    flux_arc = flux_arc[refbin]
    '''
    plt.plot(wave,flux)
    plt.title(" in input")
    plt.show()
    
    plt.plot(Refwave,flux_arc)
    plt.show()
    '''
    

    #if wave[0] <wave[-1]:
     #   wave = wave[::-1]
    #else:
     #   pass
    #print("obs wave=",wave)
    WAVE = wave
    flux = flux[::-1]   # reversing here bcoz i provided reversed array from identify task
    FLUX = flux#[::-1]
    #plt.plot(WAVE,FLUX)
    #plt.title(" the flux in linefit")
    #plt.show()
    #print(f' the len of wave {len(wave)} and flux {len(flux)} in linefit' )
    #flux = FLUX
    #wbin = np.where((wave > 3000) & (wave < 7500))
    #wave = wave[wbin]
    #flux = flux[wbin]
    #plt.plot(wave,flux)
    #plt.title("in linefit hehe")
    #plt.show()
    #zipplot(wave,flux,color="r")
    PIXEL,WAVPIXEL,pks = linepckr(wave,FLUX,nopks=nopks)
    pixel =np.arange(1,len(wave)+1)
    print(f' the len of pks {len(pks)} and wave : {len(wave)} \n and the pks is {pks}')
    #xcxc = wave[pks]
    #cxcx = flux[pks]
    print(f"centriod of peaks :{PIXEL} |peaks wave = {WAVPIXEL}")
    #zipplot(xcxc,cxcx)
    #zipplot(Refwave,flux_arc)
    
    
    #w_lower = wave[0]
    #w_upper = wave[-1]
    #nbin = np.where((Refwave > w_lower)&(Refwave < w_upper))
    
    #plt.show(block=True)
    #print(f' the input for cal_rmse = selected Wave: {xcxc} and Refwave :{Refwave}')
    rmse,obswave,refwave = cal_rmse(WAVPIXEL,Refwave)
    print(" Nearest pairs of obswave=",obswave,"refwave=",refwave)
    p2,o2,r2 = closeval3(PIXEL,WAVPIXEL,Refwave)
    PIXEL,obswave,refwave = select_pts(p2,o2,r2)
    print(f'before func fitting PIXEL= {PIXEL} \n obswave= {obswave} \n refwave= {refwave}')
    #xpix = np.where(np.isin(wave,obswave))
    #PIXEL = pixel[xpix]
    #print(f' the xpix = {xpix} and PIXEL = {PIXEL}')
    i=0
    
    if func=="poly":
    
        while rmse > 0.002:
            i += 1
            print(f"i={i}, len of. PIXEL={len(PIXEL)} ,pixel:{len(pixel)}, refwave = {len(refwave)}" )
            fit = np.polyfit(PIXEL,refwave,i)
            fpt = np.polyval(fit,PIXEL)
            rmse,obswave,refwave = cal_rmse(fpt,Refwave,k=int(len(fit)))
            print("done")
        Newwave = np.polyval(fpt,pixel)
        print("len of fpt=",len(fpt),"len of WAVE=",len(WAVE))
        #new_wave = np.interp(WAVE,np.arange(len(fpt)),fpt)
        #new_wave = np.interp(WAVE,pks,fpt)
        new_wave = np.polyval(fpt,pixel)
        
        
    if func=="interp1d":
        fit=None
        fpt=None
        while rmse > 0.01:
            i += 2
            print("i=",i)
            fit = interp1d(PIXEL,refwave,kind=2,fill_value='extrapolate')
            fpt = fit(PIXEL)
            print(f' i interp1d= fit:{fit} and fpt:{fpt}')
            rmse,obswave,refwave = cal_rmse(fpt,Refwave)
            #xpx = np.where(np.isin(wave,obswave))
    	 #PIXEL = pixel[xpx]
            print("done")
            Newwave = fit(wave)
        print("len of fpt=",len(fpt),"len of WAVE=",len(WAVE))
        #new_wave = np.interp(WAVE,np.arange(len(fpt)),fpt)
        new_wave = fit(pixel)
    
	
    if func=="spline":
        while rmse > 1e-3:
            i+=0.1
            print("i=",i)
            print(f' the obswave:{obswave} \n PIXEL:{PIXEL} \n and \n refwave {refwave}')
            
            fit = UnivariateSpline(PIXEL,refwave,ext=0,k=3,s=i) #bbox=[np.min(WAVE),np.max(WAVE)]
            fpt = fit(PIXEL)
            print(f' for i={i} | updated wave:{fpt}')
            rmse,obswave,refwave = cal_rmse(fpt,Refwave,k=3)
            print(f' the comparision wave:{wave} \n and \n new obswave {obswave}')
            #xpx = np.where(np.isin(wave,obswave))
            #print(f' the idx of the obswave {xpx}')
            print("done")
        Newwave = fit(PIXEL)
        new_wave = fit(pixel)
        
    if func=="spline3":
        while rmse > 1e-3:
            i+=0.1
            fit = CubicSpline(PIXEL,refwave)
            fpt = fit(PIXEL)
            print(f' for i={i} | updated wave:{fpt}')
            rmse,_,_= cal_rmse(fpt,refwave)
            print("done")
        Newwave = fit(PIXEL)
        new_wave = fit(pixel)
        
        
    if func=="legd":
        while rmse > 0.005:
            i+=1
            print(f"i ={i} | len of PIXEL {len(PIXEL)} & refwave {len(refwave)}")
            #PXL,WPIX,refwave = closeval3(PIXEL,WAVPIXEL,Refwave)
            PXL = np.array(PIXEL)
            Pixel,PXL = X_NORM(pixel,PIXEL)
            #print(f' the new PIXEL set {PXL} for refwave {refwave}')
            fit = legd.fit(PXL,refwave,deg=i)
            fpt = fit(PXL)
            FPT = fit(Pixel)
            #rmse,obswave,refwave= cal_rmse(fpt,refwave)
            pixcent,wavecent,pks = linepckr(FPT,FLUX,nopks=nopks)
            newwaveset = wavecent
            print(f' the peak finded in new waveset: {wavecent}')
            rmse,_,_ = cal_rmse(newwaveset,refwave,display=False)
            #PIXEL = pixel[pks]
            print(f' for i={i} | updated wave:{fpt} | \n refwave: {refwave} | \npixel peaks {PXL}')
            
            print("done")
        Newwave = fit(PXL)
        new_wave = fit(Pixel)
            

    '''
    ## check if the data points are within the limit
    sciflux = sciflux[::-1]
    if np.where((new_wave <2000) & (new_wave > 8000)):
        nbin = np.where((new_wave >3000) & (new_wave  < 7500))
        new_wave = new_wave[nbin]
        FLUx = sciflux[nbin]
    else:
        FLUx = sciflux
    #'''
    FLUx = sciflux[::-1]
    
    if new_wave[0] > new_wave[-1]:
        new_wave = new_wave[::-1]
    else:
        pass
        
    plt.figure()
    #plt.plot(refwave,refwave,linestyle="dashed",alpha=0.5,color="gray")
    #plt.scatter(Newwave,Newwave)
    #plt.title("After fitting the func")
    #plt.show()
    
    
    plt.plot(new_wave/1.043,FLUx)
    plt.title("new spectrum")
    plt.show()
    return new_wave,FLUx,fit

#############################################################################
#############################################################################   

def X_NORM(X_set,x_set):

    # normalize the PIXEL array in -1 to 1 range
    # x = (2*x - (xmax-xmin))/(xmax+xmin)
    X_norm =[]
    x_norm =[]
    
    Xmax = np.max(X_set)
    Xmin = np.min(X_set)
    #print(f' the X-max {Xmax} | X-min {Xmin}')
    for X in X_set:
        Norm_set = (2*X - (Xmax - Xmin))/(Xmax + Xmin)
        X_norm.append(Norm_set)
       
    for x in x_set:
        norm_set = (2*x - (Xmax - Xmin))/(Xmax + Xmin)
        #print(f' the norm value {norm_set}')
        x_norm.append(norm_set)
        
        
    return np.array(X_norm),np.array(x_norm)
    
    
def wave_pix(X,Y,px,ax=None,color="k",label=None):
    Y = Y/np.max(Y)
    if label==None:
        label=" "
    else:
        pass

    if ax==None:
        fig,ax = plt.subplots()
        ax.set_ylim(0,None)
    else:
        pass

    for x,y,z in zip(X,Y,px):
        ax.axvline(x=x,ymax = y,color=color,label=label)
        ax.text(x=x,y=y+0.01,s="{:.2f}".format(x),rotation=90,color="g",label="wave")
        ax.text(x=x+20,y=y+0.01,s="{:.2f}".format(z),rotation=90,color="r",label="pixel")
        
    plt.figtext(0.5,0.8,s="red for pixel \n green for wavelength")
    
    

###############################################################################
###############################################################################
from astropy.io import fits
def fitssave(NAME,WAVE,FLUX):
    wavelength = WAVE
    flux = FLUX

    # Create a FITS primary HDU
    primary_hdu = fits.PrimaryHDU()

    # Create a binary table HDU containing wavelength and flux columns
    cols = []
    cols.append(fits.Column(name='WAVELENGTH', format='D', array=wavelength))
    cols.append(fits.Column(name='FLUX', format='D', array=flux))
    col_defs = fits.ColDefs(cols)
    table_hdu = fits.BinTableHDU.from_columns(col_defs)

    # Create an HDU list containing both the primary and table HDUs
    hdul = fits.HDUList([primary_hdu, table_hdu])

    # Write the HDU list to a FITS file
    name = NAME+".fits"
    hdul.writeto(name, overwrite=True)


