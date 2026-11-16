# specred

A long-slit spectroscopy reduction pipeline (built around ADFOSC data)
covering the full workflow from raw frames to a flux-calibrated 1D
spectrum, with a PyQt GUI wrapper on top of a GUI-independent library.

**Pipeline stages:**

1. Bias combination & subtraction, frame trimming — `specred.io`
2. Flat-field response + flat correction + cosmic-ray removal — `specred.calib.flat`
3. Aperture tracing, boxcar extraction, sky subtraction — `specred.extraction.aperture`
4. Wavelength calibration — `specred.calib.wavelength`
5. Flux calibration (airmass correction, sensitivity function) — `specred.calib.flux`
6. Final spectrum plotting/saving — `specred.output`


## Installation

```bash
git clone <this-repo-url>
cd specred
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gui]"
```

This installs the `specred` library and the `specred-gui` console
script. If you only want to script the reduction (no GUI), drop the
`[gui]` extra and install without PyQt5.

## Required reference data (not included)

The pipeline needs two kinds of reference tables

- Arc-lamp line lists (e.g. `HgAr_neon.dat`) → `src/specred/resources/arc/`
- Standard-star flux tables (e.g. from IRAF's `onedstds`) → `src/specred/resources/onedstar/<catalog>/`

See `src/specred/resources/README.md` for the expected layout. Add
your own files here (and note their source/license) before publishing
a "batteries included" release, or document how users should obtain
them.

## Configuration

A reduction run is described by a YAML config (see
`examples/fileinfo_example.yaml`):

```yaml
workdir: "./data/"
bias_pattern: "bias*fit"
continuum_lamp: "continuum*fit"
arc_lamp:
  pattern: "arclamp*fit"
  name: "HgAr_neon.dat"
standard_star:
  pattern: "Feige66*fit"
  name: "Feige66"
  combine: "mean"
science_obj:
  pattern: "science*fit"
  name: "simbad name"
  xlim: "1500"
  combine: "mean"
slit: '1.5"'
grism: '676R-420gr/mm'
trim: "True"
spatial_axis: "0"
```

Copy `examples/fileinfo_example.yaml`, point `workdir` at your data,
and load it from the GUI's "File Selection" tab (Browse... button), or
pass the parsed values directly to `specred.io.readfile(...)` in a
script.

Instrument-specific constants (currently only ADFOSC) live in
`src/specred/instruments/adfosc.yaml`. To support a different
instrument, add a new YAML file there with your own trim box and
wavelength-solution guesses — no code changes needed.

## Running the GUI

```bash
specred-gui
# or, without installing:
python -m gui.main_window
```

The GUI tabs follow the pipeline order: File Selection → Flat
Correction → Aperture Extraction → Spectral Calibration → Flux
Calibration → Output.

## Repository layout

```
specred_pipeline/
├── pyproject.toml
├── requirements.txt
├── README.md / CHANGELOG.md / LICENSE
├── src/specred/            # GUI-independent reduction library
│   ├── io.py                bias/flat/science/std/lamp reading, trimming
│   ├── output.py             final spectrum plotting/saving
│   ├── utils.py               shared helpers (peak-finding, line-picking, etc.)
│   ├── calib/
│   │   ├── flat.py             flat response + flat-fielding + cosmic rays
│   │   ├── wavelength.py        wavelength calibration
│   │   ├── flux.py               flux calibration
│   │   └── airmass.py
│   ├── extraction/
│   │   └── aperture.py          tracing + boxcar extraction + sky subtraction
│   ├── instruments/
│   │   ├── adfosc.yaml           instrument-specific constants
│   │   └── __init__.py            config loader
│   └── resources/                arc line lists & standard star tables (add your own)
├── gui/                     # PyQt front-end (thin; calls into src/specred)
│   ├── main_window.py
│          
├── examples/
│   └── fileinfo_example.yaml
└── tests/                   # add unit tests for the library modules here
```

### Update V1.1
1. Dispersion line trace approach is changed.
2. Upgrade in the GUI interface
3. Changes in the final file saving format.

