import numpy as np
from astropy.io import fits
import os, sys, glob
import julian
from datetime import datetime
from astropy.nddata.utils import Cutout2D
import ccdproc
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.stats import SigmaClip, mad_std
from astropy import units as u
from astropy.io.fits import CompImageHDU


# Default CCD trim box (ADFOSC). Instrument-specific values should be
# supplied via an instrument config (see src/specred/instruments/) and
# passed explicitly to readfile()/Trim() rather than relying on these
# module-level defaults.
DEFAULT_TRIM = dict(cx=0, cy=2030, h=550, w=1640)

#########################
def parse_date_obs(s: str) -> datetime:
    """Try several common DATE-OBS formats."""
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",  # with microseconds, ISO
        "%Y-%m-%dT%H:%M:%S",     # no microseconds, ISO
        "%Y-%m-%d %H:%M:%S.%f",  # space instead of T, with microseconds
        "%Y-%m-%d %H:%M:%S",     # space instead of T, no microseconds
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # If we get here, nothing matched
    raise ValueError(f"Unrecognized DATE-OBS format: {s!r}")

def Trim(data,xmin,xmax,ymin,ymax):
    # trim the 2D image(2k*2k for ADFOSC) into the required size
    DATA = data[ymin:ymax,xmin:xmax]
    Data = ccdproc.CCDData(DATA,unit="adu")
    return Data

def check(file):
    if check is not None:
        name = file #sorted(glob.glob(file))
    else:
        w = input("enter the :")
        name = w # sorted(glob.glob(w))
    return name

def save_file1(foldername,file_name, data, header=None):
        """Helper function to save a file in the output folder."""
        full_path = os.path.join(foldername, file_name)
        
        compressed_hdu = CompImageHDU(data=data.data.astype(np.float32), header=header, compression_type='RICE_1')
        compressed_hdu.writeto(full_path,overwrite=True)
        #fits.writeto(full_path, data, header, overwrite=True)
        print(f"COmpressed Saved file: {full_path}")

def save_file(foldername,file_name, data, header=None):
    """Helper function to save a file in the output folder."""
    full_path = os.path.join(foldername, file_name)
    fits.writeto(full_path,data.data.astype(np.float32), header, overwrite=True)
    print(f"Saved file: {full_path}")
        
def readfile(Bias=None,Flat=None,Sci=None,Std=None,
             Lamp =None,Sci_name=None,Std_name=None,trim=False,sci_combine=False,
             trim_box=None):
    """
    Read raw bias/flat/science/standard/lamp frames, build calibration
    masters, and write bias-subtracted (and optionally trimmed) frames
    to an output folder.

    Parameters
    ----------
    trim_box : dict, optional
        {'cx':..., 'cy':..., 'h':..., 'w':...}. Defaults to
        io.DEFAULT_TRIM (ADFOSC) if not supplied.

    Notes
    -----
    All intermediate frame lists (bias, flat, science, standard, lamp)
    are local to this call -- nothing is accumulated in module-level
    state, so repeated calls (e.g. from a long-lived GUI session) do
    not leak frames from a previous run into a new master bias/flat.
    """
    trim_box = trim_box or DEFAULT_TRIM
    cx, cy, h, w = trim_box["cx"], trim_box["cy"], trim_box["h"], trim_box["w"]

    # local, per-call frame lists (previously module-level globals that
    # persisted across calls and silently accumulated frames)
    biaslist = []
    flatlist = []
    sclist = []
    # NOTE (bug fix, follow-up): `faltin` was also one of the original
    # module-level globals removed in the first pass, but was missed
    # then -- it's actively used (appended to below, and returned at the
    # end of this function), unlike lamplist/stdlist which turned out to
    # be dead/commented-out in the original code. Caught by re-scanning
    # for names used-but-never-bound before shipping.
    faltin = []
    masterbias = None

    if os.path.exists(Sci_name):
        # Find all folders matching the base_name pattern
        existing_folders = glob.glob(f"{Sci_name}_*")
        # Extract numbers from the folder names
        numbers = [int(folder.split('_')[-1]) for folder in existing_folders if folder.split('_')[-1].isdigit()]
        # Determine the next folder number
        next_number = max(numbers, default=0) + 1
        new_folder = f"{Sci_name}_{next_number}"
        os.makedirs(new_folder)
        foldername = str(new_folder)
    else:
        os.makedirs(Sci_name)
        foldername = str(Sci_name)
        
    #os.chdir(foldername)
    
    bias_name = check(Bias)
    flat_name = check(Flat)
    sc_name = check(Sci)
    std_name = check(Std)
    Lamp_name = check(Lamp)

    
    if Sci_name is not None:
        Sc_obj_name = Sci_name
    else:
        Sc_obj_name = str(input("enter the name of Science obj"))
        
    if Std_name is not None:
        std_obj_name = Std_name
    else:
        std_obj_name = str(input("enter the name of the standard star"))

    log_file = Sc_obj_name + datetime.now().strftime("%Y-%m-%d_%H") + ".txt"
    flatin = "flat_cor.csv"
    # Open the log file for writing
    directory = os.getcwd()
    #files = sorted(glob.glob("bias_13.fit"))
    with open(log_file, "w") as log:
        # Write a header to the log file
        log.write("FITS file list for directory " + directory + "\n\n")
        log.write("Filename \t File Type \t Date of Observation \t ExpTime \n ")
        log.write("--"*50 + "\n")

    def log_read():
        with open(log_file, "a") as log:
            log.write(obj_name + "\t" + obj_type + "\t" +str(obs_date) + "\t" +str(exptime) +"\n" )
    def log_line():
        with open(log_file,"a") as log:
            log.write("--"*50 + "\n")
    def fltin():
        with open(flatin,"w") as flt:
            flt.write(name)
            
    ##### for bias files
    if bias_name != str():
        blist = sorted(glob.glob(bias_name))
        print(" BIAS FILES :\n", blist)
        for files in blist:
            #print(f" for the bias file {files}")
            #data = ccdproc.CCDData.read(files,unit="adu")
            file = fits.open(files)
            data = file[0].data
            #print(f" biass data = {data}")
            header = file[0].header
            if trim==True:
                data = Trim(data,cx,cy,h,w)
            if "NAXIS3" in header:
                header.remove("NAXIS3")
            d = "OBJECT" in header
            if d == True:
                obj_name_1 = header["OBJECT"]
                print("1")
                a= files.split("_")
                b = a[0]
                if b==obj_name_1:
                    print("matched")
                    obj_name = obj_name_1
                else:
                    print("not matched")
                    obj_name = b
            else :
                print("2")
                a= files.split("_")
                b = a[0]
                obj_name = b

            obj_type = "BIAS"
            exptime = header["EXPTIME"]
            obs_date = header["DATE-OBS"]
            
            dt = parse_date_obs(header["DATE-OBS"])
            #print(dt)
            jd = julian.to_jd(dt, fmt="jd")
            #print("JD=",jd)
            header["JD"] = jd
            header["MJD"] = round(jd - 2400000.5, 5)

##            header['JD'] =  julian.to_jd(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"),fmt='jd')
##            jd =  julian.to_jd(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"),fmt='jd')
##            header["MJD"] = round((jd - 2400000.5),5)
            header["NAXIS"]= 3
            header["BUNIT"]="adu"
            biaslist.append(data)
            log_read()
        print("read all bias frames")
        if os.path.exists(os.path.join(foldername,"masterbias.fit"))==False:
            print(11)
            masterbias = ccdproc.combine(biaslist,method = 'median',sigma_clip=True,sigma_clip_low_thresh=5,sigma_clip_high_thresh=5,sigma_clip_func=np.ma.median,sigma_clip_dev_func=mad_std)
            print(12)
            #fits.writeto("masterbias.fit",masterbias, header,overwrite=True)
            save_file(foldername,"masterbias.fit",masterbias,header)
            print(" Masterbias file has been created")
        else :
            print("masterbias file already exist")
            masterbias = ccdproc.CCDData.read(os.path.join(foldername,"masterbias.fit"),unit="adu")
    else:
        print(" Bias files are not given")
        masterbias = None
  



    #### for flat (continuum) frames
    if os.path.exists("mflat.fits")==True:
        pass
    else:
        if flat_name != str():
            blist = sorted(glob.glob(flat_name))
            print(" Flat FILES :\n", blist)
            for files in blist:
                f = fits.open(files)
                #data = f[0].data
                
                data = f[0].data #ccdproc.CCDData.read(files,unit="adu")
                header = f[0].header
                if "NAXIS3" in header:
                    header.remove("NAXIS3")
           
                if trim==True:
                    data = Trim(data,cx,cy,h,w)
                #print(header)

                d = "OBJECT" in header
                if d == True:
                    obj_name_1 = header["OBJECT"]
                    print("1")
                    a= files.split("_")
                    b = a[0]
                    if b==obj_name_1:
                        print("matched")
                        obj_name = obj_name_1
                    else:
                        print("not matched")
                        obj_name = b
                else :
                    print("2")
                    a= files.split("_")
                    b = a[0]
                    obj_name = b

                obj_type = "Continuum"
                exptime = header["EXPTIME"]
                obs_date = header["DATE-OBS"]
                

                log_read()
                dt = parse_date_obs(header["DATE-OBS"])
                jd = julian.to_jd(dt, fmt="jd")
                header["JD"] = jd
                header["MJD"] = round(jd - 2400000.5, 5)

##                header['JD'] =  julian.to_jd(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"),fmt='jd')
##                jd =  julian.to_jd(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"),fmt='jd')
##                header["MJD"] = round((jd - 2400000.5),5)
                header["BUNIT"]="adu"
                if masterbias is not None:
                    data = ccdproc.subtract_bias(data,masterbias)
                    header["comment"] = "Bias Subtracted"
                    
                else:
                    print(" No bias Subtraction done")
                flatlist.append(data)
                
            masterflat = ccdproc.combine(flatlist,method = 'median',sigma_clip=True,sigma_clip_low_thresh=5,sigma_clip_high_thresh=5,sigma_clip_func=np.ma.median,sigma_clip_dev_func=mad_std)
            #fits.writeto("masterflat.fit",masterflat, header,overwrite=True)
            save_file(foldername,"masterflat.fit",masterflat, header)

        else:
            print(" flat files are not given")

        #### for science frames

    if sc_name != None:
        
        blist = sorted(glob.glob(sc_name))
        print(" SCIENCE FILES :\n", blist)
        for files in blist:
            file = fits.open(files)
            data= file[0].data
            #data = ccdproc.CCDData.read(files,unit="adu")
            header = file[0].header
            if "NAXIS3" in header:
                header.remove("NAXIS3")
            if trim:
                data = Trim(data,cx,cy,h,w)
            header["NAXIS"]=3
            
            header['JD'] =  julian.to_jd(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"),fmt='jd')
            jd =  julian.to_jd(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"),fmt='jd')
            header["MJD"] = round((jd - 2400000.5),5)
            header["BUNIT"]="adu"
            if masterbias is not None:
                data = ccdproc.subtract_bias(data,masterbias)
                header["comment"] = "Bias Subtracted"
            else :
                print("No bias Subtraction to science frames")
            sclist.append(data)
            red_data = data
            
            

            d = "OBJECT" in header
            if d == True:
                obj_name_1 = header["OBJECT"]
                print("1")
                a= files.split("_")
                b = a[0]
                if b==obj_name_1:
                    print("matched")
                    obj_name = obj_name_1
                else:
                    print("not matched")
                    obj_name = b
            else :
                print("2")
                a= files.split("_")
                b = a[0]
                obj_name = b

            obj_type = "Science"
            exptime = header["EXPTIME"]
            obs_date = header["DATE-OBS"]

            log_read()
            name = "bs_" + files
            faltin.append(name)
            if sci_combine:
                scdata=ccdproc.combine(sclist,method="median",sigma_clip=True,sigma_clip_low_thresh=5,sigma_clip_high_thresh=5,
                                                      sigma_clip_func=np.ma.median,sigma_clip_dev_func=mad_std)
            
                #fits.writeto(name,scdata,header,overwrite=True)
                save_file(foldername,name,scdata,header)
            else:
                save_file(foldername,name,red_data,header)
                

    else:
        print(" Science files not given")



    #### for standard source frames 

    if std_name != str():
        blist = sorted(glob.glob(std_name))
        print(" STD Source FILES :\n", blist)
        for files in blist:
            file = fits.open(files)
            data = file[0].data
            #data = ccdproc.CCDData.read(files,unit="adu")
            header = file[0].header
            if "NAXIS3" in header:
                header.remove("NAXIS3")
            if trim:
                data = Trim(data,cx,cy,h,w)
            header["NAXIS"]=3
            dt = parse_date_obs(header["DATE-OBS"])
            jd = julian.to_jd(dt, fmt="jd")
            header["JD"] = jd
            header["MJD"] = round(jd - 2400000.5, 5)
##            header['JD'] =  julian.to_jd(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"),fmt='jd')
##            jd =  julian.to_jd(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"),fmt='jd')
##            header["MJD"] = round((jd - 2400000.5),5)
            header["BUNIT"]="adu"
            if masterbias is not None:
                data = ccdproc.subtract_bias(data,masterbias)
                header["comment"] = "Bias Subtracted"
            else :
                print("No bias Subtraction to std source frames")
            #stdlist.append(data)
            name = "bs_" + files
            #fits.writeto(name,data,header,overwrite=True)
            save_file(foldername,name,data,header)
            faltin.append(name)

            d = "OBJECT" in header
            if d == True:
                obj_name_1 = header["OBJECT"]
                print("1")
                a= files.split("_")
                b = a[0]
                if b==obj_name_1:
                    print("matched")
                    obj_name = obj_name_1
                else:
                    print("not matched")
                    obj_name = b
            else :
                print("2")
                a= files.split("_")
                b = a[0]
                obj_name = b

            obj_type = "Std"
            exptime = header["EXPTIME"]
            obs_date = header["DATE-OBS"]

            log_read()

    else:
        print(" Std source files not given")


    #### for Lamp frames
    
    if Lamp_name != None:
        Lampin=[]
        blist = sorted(glob.glob(Lamp_name))
        print(" Lamp FILES :\n", blist)
        for files in blist:
            file = fits.open(files)
            data = file[0].data
            #data = ccdproc.CCDData.read(files,unit="adu")
            header = file[0].header
            if "NAXIS3" in header:
                header.remove("NAXIS3")
            if trim:
                data = Trim(data,cx,cy,h,w)
            dt = parse_date_obs(header["DATE-OBS"])
            jd = julian.to_jd(dt, fmt="jd")
            header["JD"] = jd
            header["MJD"] = round(jd - 2400000.5, 5)
##            header['JD'] =  julian.to_jd(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"),fmt='jd')
##            jd =  julian.to_jd(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"),fmt='jd')
##            header["MJD"] = round((jd - 2400000.5),5)
            header["BUNIT"]="adu"
            if masterbias is not None:
                data = ccdproc.subtract_bias(data,masterbias)
                header["comment"] = "Bias Subtracted"
            else :
                print("No bias Subtraction to Lamp frames")
            Lampin.append(data)
            name = "bs_" + files
            #fits.writeto(name,data,header,overwrite=True)
            save_file(foldername,name,data,header)
            
            #Lampin.append(name)
            
                    
            #lamplist.append(data)

            d = "OBJECT" in header
            if d == True:
                obj_name_1 = header["OBJECT"]
                print("1")
                a= files.split("_")
                b = a[0]
                if b==obj_name_1:
                    print("matched")
                    obj_name = obj_name_1
                else:
                    print("not matched")
                    obj_name = b
            else :
                print("2")
                a= files.split("_")
                b = a[0]
                obj_name = b

            obj_type = "arc lamp"
            exptime = header["EXPTIME"]
            obs_date = header["DATE-OBS"]

            log_read()
            
        masterlamp = ccdproc.combine(Lampin,method = 'median',sigma_clip=True,sigma_clip_low_thresh=5,
                                         sigma_clip_high_thresh=5,sigma_clip_func=np.ma.median,
                                         sigma_clip_dev_func=mad_std)
            
        Name = "com_"+files.split("_")[0]+".fit"
        #fits.writeto(Name,masterlamp, header,overwrite=True)
        save_file(foldername,Name,masterlamp, header)
        faltin.append(Name)
    else:
        print(" Lamp files not given")
    
    log_line()  
    
    #f.write("FITS file list for directory " + directory + "\n\n")
    
    return (foldername,Sc_obj_name, std_obj_name,faltin)


