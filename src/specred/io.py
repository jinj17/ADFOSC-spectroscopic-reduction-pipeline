"""ADFOSC basic CCD preparation: file discovery, trimming, bias subtraction.

Compatible with Python 3.8 and the original GUI call:
readfile(Bias=..., Flat=..., Sci=..., Lamp=..., Std=...,
         Sci_name=..., Std_name=..., trim=...)

Products are written in a new science-named output folder. Returned names of
bias-subtracted science/standard and combined arc frames are relative to that
folder, matching the downstream flat_2.flat_correction interface.
"""
from pathlib import Path
from datetime import datetime
import glob
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.time import Time
import astropy.units as u
import ccdproc

# Original ADFOSC crop: Python slices [550:1640, 0:2030].
DEFAULT_TRIM = (0, 2030, 550, 1640)  # xmin, xmax, ymin, ymax


def parse_date_obs(value):
    """Parse a DATE-OBS string with or without fractional seconds."""
    return datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))


def Trim(data, xmin=0, xmax=2030, ymin=550, ymax=1640):
    """Return a trimmed CCDData preserving useful FITS metadata."""
    ccd = data if isinstance(data, CCDData) else CCDData(np.asarray(data), unit=u.adu)
    ny, nx = ccd.shape
    if not (0 <= xmin < xmax <= nx and 0 <= ymin < ymax <= ny):
        raise ValueError('Trim bounds {} exceed image shape {}.'.format((xmin, xmax, ymin, ymax), ccd.shape))
    return CCDData(ccd.data[ymin:ymax, xmin:xmax].copy(), unit=ccd.unit, meta=ccd.header.copy())


def _files(pattern):
    if not pattern:
        return []
    matches = sorted(Path(p) for p in glob.glob(str(pattern)))
    return [p for p in matches if p.is_file()]


def _read_ccd(path, trim=False, trim_bounds=DEFAULT_TRIM):
    with fits.open(str(path), memmap=False) as hdul:
        data = hdul[0].data
        if data is None or data.ndim != 2:
            raise ValueError('Expected 2D primary image: {}'.format(path))
        hdr = hdul[0].header.copy()
        ccd = CCDData(np.asarray(data, dtype=np.float64), unit=u.adu, meta=hdr)
    if trim:
        ccd = Trim(ccd, *trim_bounds)
    hdr = ccd.header
    hdr['BUNIT'] = 'adu'
    if hdr.get('DATE-OBS'):
        try:
            date = parse_date_obs(hdr['DATE-OBS'])
            jd = float(Time(date, scale='utc').jd)
            hdr['JD'] = jd
            hdr['MJD'] = round(jd - 2400000.5, 5)
        except (ValueError, TypeError) as exc:
            print('Warning: DATE-OBS not parsed for {}: {}'.format(path.name, exc))
    return ccd


def _combine(frames, label):
    if not frames:
        raise ValueError('No {} frames supplied.'.format(label))
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError('Inconsistent {} frame shapes: {}'.format(label, shapes))
    if len(frames) == 1:
        return frames[0].copy()
    return ccdproc.combine(frames, method='median', sigma_clip=True,
                           sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                           sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std)


def _write(path, ccd, comment=None):
    header = ccd.header.copy()
    header['BUNIT'] = 'adu'
    if comment:
        header.add_history(comment)
    fits.writeto(str(path), np.asarray(ccd.data, dtype=np.float32), header=header, overwrite=True)
    print('Saved: {}'.format(path))


def _output_folder(parent, name):
    base = Path(parent) / name
    if not base.exists():
        base.mkdir()
        return base
    index = 1
    while (Path(parent) / '{}_{}'.format(name, index)).exists():
        index += 1
    folder = Path(parent) / '{}_{}'.format(name, index)
    folder.mkdir()
    return folder


def readfile(Bias=None, Flat=None, Sci=None, Std=None, Lamp=None,
             Sci_name=None, Std_name=None, trim=False, sci_combine=False,
             trim_bounds=DEFAULT_TRIM):
    """Prepare ADFOSC frames and return (output_folder, science, standard, flat_inputs).

    Expects the GUI to have selected the working directory (usually via chdir).
    Outputs masterbias.fit, masterflat.fit, bs_<source>.fit(s), and com_<lamp>.fit.
    `flat_inputs` lists filenames relative to output_folder, as in the old API.
    """
    if not Sci_name or not str(Sci_name).strip():
        raise ValueError('Science object name is required.')
    if not Std_name or not str(Std_name).strip():
        raise ValueError('Standard star name is required.')
    sci_name, std_name = str(Sci_name).strip(), str(Std_name).strip()
    if Path(sci_name).name != sci_name or Path(std_name).name != std_name:
        raise ValueError('Object names must not contain directory separators.')

    matches = {kind: _files(pattern) for kind, pattern in
               [('BIAS', Bias), ('FLAT', Flat), ('SCIENCE', Sci), ('STANDARD', Std), ('ARC', Lamp)]}
    for kind in ('BIAS', 'FLAT', 'SCIENCE', 'STANDARD', 'ARC'):
        if not matches[kind]:
            raise FileNotFoundError('No {} files matched pattern {!r}'.format(kind, {'BIAS': Bias, 'FLAT': Flat, 'SCIENCE': Sci, 'STANDARD': Std, 'ARC': Lamp}[kind]))
    # Check before writing any output; the flat and science frames must match.
    folder = _output_folder(Path.cwd(), sci_name)
    log_path = folder / 'fileinfo_{}.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
    product_names = []
    records = []

    def load(kind):
        frames = []
        for path in matches[kind]:
            frame = _read_ccd(path, trim=trim, trim_bounds=trim_bounds)
            frames.append(frame)
            records.append((path.name, kind, frame.header.get('DATE-OBS', ''), frame.header.get('EXPTIME', '')))
        return frames

    bias_frames = load('BIAS')
    masterbias = _combine(bias_frames, 'bias')
    _write(folder / 'masterbias.fit', masterbias, 'Median combined bias frames')

    def subtract(frames):
        output = []
        for frame in frames:
            if frame.shape != masterbias.shape:
                raise ValueError('Frame shape {} does not match bias shape {}'.format(frame.shape, masterbias.shape))
            result = ccdproc.subtract_bias(frame, masterbias)
            result.header.add_history('Master bias subtracted')
            output.append(result)
        return output

    flat_frames = subtract(load('FLAT'))
    masterflat = _combine(flat_frames, 'flat')
    _write(folder / 'masterflat.fit', masterflat, 'Bias-subtracted median continuum flat')

    for kind in ('SCIENCE', 'STANDARD'):
        inputs = matches[kind]
        corrected = subtract(load(kind))
        if kind == 'SCIENCE' and sci_combine:
            merged = _combine(corrected, 'science')
            name = 'bs_{}_combined.fit'.format(sci_name)
            _write(folder / name, merged, 'Combined bias-subtracted science frames')
            product_names.append(name)
        else:
            for original, frame in zip(inputs, corrected):
                name = 'bs_' + original.name
                _write(folder / name, frame)
                product_names.append(name)

    arcs = subtract(load('ARC'))
    for original, frame in zip(matches['ARC'], arcs):
        _write(folder / ('bs_' + original.name), frame)
    combined_arc = _combine(arcs, 'arc')
    arc_stem = matches['ARC'][0].name.split('_')[0]
    arc_name = 'com_{}.fit'.format(arc_stem)
    _write(folder / arc_name, combined_arc, 'Combined bias-subtracted arc frames')
    product_names.append(arc_name)

    with log_path.open('w', encoding='utf-8') as fh:
        fh.write('ADFOSC input frames | working directory: {}\n'.format(Path.cwd()))
        fh.write('Filename\tType\tDATE-OBS\tEXPTIME\n')
        for record in records:
            fh.write('{}\t{}\t{}\t{}\n'.format(*record))
    print('Prepared {} inputs in {}'.format(len(product_names), folder))
    # Relative output folder retains the GUI's existing os.chdir(foldername) behaviour.
    return folder.name, sci_name, std_name, product_names
