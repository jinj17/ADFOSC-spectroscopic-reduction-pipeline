"""
Instrument configuration loading.

Previously, instrument-specific numbers (CCD trim box, slit/grism ->
wavelength-solution guesses) were hardcoded inside readfile_v2.py and
the GUI's wavelength-calibration dialog. They now live in per-instrument
YAML files in this package (e.g. adfosc.yaml), loaded through the
helpers below.
"""
import os
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_instrument(name: str) -> dict:
    """
    Load an instrument config by name (matches a `<name>.yaml` file in
    this folder, case-insensitive). Raises FileNotFoundError with a
    helpful message if the instrument isn't defined yet.
    """
    path = os.path.join(_THIS_DIR, f"{name.lower()}.yaml")
    if not os.path.exists(path):
        available = [
            f[:-5] for f in os.listdir(_THIS_DIR) if f.endswith(".yaml")
        ]
        raise FileNotFoundError(
            f"No instrument config found for '{name}'. "
            f"Available: {available}. Add a new YAML file here to "
            f"support another instrument."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def get_trim_box(instrument_cfg: dict) -> dict:
    """Return {'cx','cy','h','w'} from a loaded instrument config."""
    return instrument_cfg["trim_box"]


def get_wavelength_guess(instrument_cfg: dict, slit: str, grism: str):
    """
    Return (dispersion, offset) initial guess for the given slit/grism
    combination, or (None, None) if not defined for this instrument.
    """
    for entry in instrument_cfg.get("wavelength_solution_guess", []):
        if entry["slit"] == slit and entry["grism"] == grism:
            return entry["dispersion"], entry["offset"]
    return None, None
