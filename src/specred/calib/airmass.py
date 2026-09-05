
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.table import Table
from astropy.constants import c as cc
import ccdproc
from astropy import units as u
import os
from astropy.io import fits

"""
NOTE:
-----

instead of importing the RA and DEC of the object using astropy.coordinates we can call the SIMBAD query or 
NED query(for extragalatic objects only) and extract from the searched query.
### look for add.redshift2(objname_name) 
"""

def airmass(DATE, sourcename, header,teles="DOT"):
    if DATE is None:
        raise ValueError("Failed to obtain DATE-OBS from the input.")
    print(DATE)
    if sourcename is None:
        d = "OBJECT" in header
        if d:
            name = header["OBJECT"]
        else:
            print("Specify the object name")
            name = str(input("Enter the Object name (searchable in SIMBAD): "))
    else:
        name = sourcename
    print("The Object name is:", name)
    print("DATE =", DATE)
    try:
        obj = SkyCoord.from_name(name)
    except Exception as e:
        print(f" error in airmass function {str(e)}")

        try:
            obj = SkyCoord(f"{header['ra']} {header['dec']}",unit=u.deg)
        except Exception as e:
            print(" RA DEC issue or some issue")
            

    if teles == "DOT":
        site = EarthLocation(lat=29.3611*u.deg, lon=79.6844*u.deg, height=2420*u.m)
        utcoffset = -5.5*u.hour
        time = Time(DATE) + utcoffset
    elif teles == "HCT":
        site = EarthLocation(lat=32.7794*u.deg, lon=78.9642*u.deg, height=4500*u.m)
        time = Time(DATE)
        
    

    objaltaz = obj.transform_to(AltAz(obstime=time, location=site))
    #print(f"Object's Altitude = {objaltaz.alt:.3}")

    frame_night = AltAz(obstime=time, location=site)
    objaltazs_night = obj.transform_to(frame_night)

    objairmass = objaltazs_night.secz

    print("OBS_time:", header["DATE-OBS"], "\nUTC:", time, "\nAirmass:", objairmass)
    aIRMASS = objairmass.value

    return aIRMASS

def Airmass(file, sourcename=None,teles="DOT"):
    DATE = None
    airMASS = None
    print("TELESCOPE=", teles)
    ccddata_check = ccdproc.CCDData([0, 0], unit="adu")
    if isinstance(file, type(ccddata_check)):
        #print("1")
        header = file.header
        DATE = header["DATE-OBS"]
        hdr = header
        d = "AIRMASS" in header
        if d:
            if isinstance(hdr["AIRMASS"], (int, float)):
                print("airmass present")
                airMASS = hdr["AIRMASS"]
            else:
                print("adding")
                airMASS = airmass(DATE, sourcename, hdr,teles)
        else:
            print("adding")
            airMASS = airmass(DATE, sourcename, hdr,teles)
        
    elif isinstance(file, type('r')):
        #print("2")

        if os.path.exists(file) == True:
            file2 = file
            print(file2)

        elif os.path.exists(file+".fit") == True:
            file2 = file+".fit"
            print("exist fit")

        else:
            print("exist fits")
            file2 = file+".fits"
        print("filename=", file)
        File = fits.open(file2)

        header = File[0].header
        DATE = header["DATE-OBS"]
        hdr = header
        d = "AIRMASS" in header
        if d:
            if isinstance(hdr["AIRMASS"], (int, float)):
                print("airmass present")
                airMASS = hdr["AIRMASS"]
            else:
                print("adding")
                airMASS = airmass(DATE, sourcename, hdr,teles)
        else:
            airMASS = airmass(DATE, sourcename, hdr,teles)  # Assign airmass if not found in header

    print("airmass =", airMASS)
    return airMASS

def Airmass_fits(file, sourcename=None,teles="DOT",makenew=True):
    DATE = None
    airMASS = None
    print("TELESCOPE=", teles)
    ccddata_check = ccdproc.CCDData([0, 0], unit="adu")
    if isinstance(file, type(ccddata_check)):
        #print("1")
        header = file.header
        DATE = header["DATE-OBS"]
        hdr = header
        d = "AIRMASS" in header
        if d:
            if isinstance(hdr["AIRMASS"], (int, float)):
                print("airmass present")
                airMASS = hdr["AIRMASS"]
            else:
                print("adding")
                airMASS = airmass(DATE, sourcename, hdr,teles)
        else:
            print("adding")
            airMASS = airmass(DATE, sourcename, hdr,teles)
        
    elif isinstance(file, type('r')):
        #print("2")

        if os.path.exists(file) == True:
            file2 = file
            print(file2)

        elif os.path.exists(file+".fit") == True:
            file2 = file+".fit"
            print("exist fit")

        else:
            print("exist fits")
            file2 = file+".fits"
        print("filename=", file)
        File = fits.open(file2)

        header = File[0].header
        DATE = header["DATE-OBS"]
        hdr = header
        d = "AIRMASS" in header
        if d:
            if isinstance(hdr["AIRMASS"], (int, float)):
                print("airmass present")
                airMASS = hdr["AIRMASS"]
            else:
                print("adding")
                airMASS = airmass(DATE, sourcename, hdr,teles)
        else:
            airMASS = airmass(DATE, sourcename, hdr,teles)  # Assign airmass if not found in header
        
        if makenew:
            header["AIRMASS"] = airMASS
            fits.writeto(file2,File[0].data,header,overwrite=True)

    print("airmass =", airMASS)
    return airMASS


