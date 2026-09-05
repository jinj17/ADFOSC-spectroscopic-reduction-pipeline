"""
Main GUI window for the spectroscopic reduction pipeline.

This is a thin PyQt front-end: each tab collects parameters from the
user and calls into the GUI-independent `specred` library
(specred.io, specred.calib.*, specred.extraction, specred.output).
The library itself can be scripted headlessly without this GUI.
"""
import sys
import os
import numpy as np
from astropy.io import fits
import ccdproc
from specutils import Spectrum1D
from astropy import units as u
from astropy.nddata import StdDevUncertainty
import matplotlib
import matplotlib.pylab as pllt
import yaml
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Button
from collections import defaultdict
from astropy.convolution import convolve, Box1DKernel
from astropy.table import Table
# NOTE (bug fix): find_peaks (used in the sensitivity-preview plot) was
# called without ever being imported. It was wrapped in a bare
# try/except so it failed silently -- the peak markers on that plot
# never actually appeared.
from scipy.signal import find_peaks

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QLineEdit, QComboBox,
    QFileDialog, QTabWidget, QTextEdit, QCheckBox, QSpinBox,
    QDoubleSpinBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QStyledItemDelegate, QItemEditorFactory, QItemEditorCreatorBase,
    QDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator

# NOTE (bug fix): the original computed `scpath` via
# os.path.dirname(os.path.abspath(__name__)) and sys.path.append'd it to
# import loose script files. `__name__` is a *module name string*, not a
# file path, so this resolved against the current working directory
# rather than this file's real location -- it happened to work only
# when launched from the right directory. Now that this is an
# installed/importable package, no sys.path hacking is needed at all.
from specred import utils as add
from specred.calib import flux as fluxcal
from specred.calib import airmass
from specred.calib import wavelength as identify
from specred.calib import flat as flat_2
from specred import io as readfile_v2
from specred import output as final
from specred.extraction import aperture as aperture
from specred.instruments import load_instrument, get_trim_box

from gui.dialogs.wavelength_dialog import InteractiveWavelengthDialog

# NOTE (bug fix): the GUI body has several defaults that build paths under
# "resources/" (extinction table, standard-star tables, resource-path
# field). The original `script_dir` was computed the same buggy way as
# `scpath` used to be (os.path.dirname(os.path.abspath(__name__)),
# resolved against the CWD rather than any real file location). Since
# resources/ now lives inside the installed specred package
# (src/specred/resources/), point script_dir there directly so these
# defaults work regardless of which directory specred-gui is launched from.
import specred
script_dir = os.path.dirname(os.path.abspath(specred.__file__))

# Global spatial axis variable
sxs = 0
SXS = 0

#=============================================================================================================================#

class SpectroscopicReductionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectroscopic Reduction Pipeline")
        self.setGeometry(10, 10, 1000, 1000)
        
        # Initialize variables
        self.init_variables()
        
        # Setup UI
        self.setup_ui()
        
        
    def init_variables(self):
        """Initialize all class variables"""
        # Which instruments/<name>.yaml config to use for trim box,
        # wavelength-solution guesses, etc. Only "adfosc" exists today;
        # expose this as a dropdown in the UI if/when a second
        # instrument config is added.
        self.instrument_name = "adfosc"

        # File lists
        self.fileinfo =[]
        self.sci_lst = []
        self.std_lst = []
        self.arclst = []
        self.flatin = []
        
        # Extracted data
        self.SCIdata = []
        self.STDdata = []
        self.ARCdata = []
        self.sci_results = []
        self.std_results = []
        self.sci_airmass = []
        self.std_airmass = []
        
        # Combined spectra
        self.comb_SCI = None
        self.comb_STD = None
        
        # Calibrated spectra
        self.W_sci = None
        self.W_std = None
        self.final_spec = None
        
        # Other
        self.sciname = ""
        self.stdname = ""
        self.lst = []
        self.standardstar = None
        self.slit=""
        self.grism=""
        # Preview tracking
        self.current_preview_index = 0
        self.current_preview_type = None
        
    def setup_ui(self):
        """Setup the main UI components"""
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.setup_file_selection_tab()
        self.setup_flat_correction_tab()
        self.setup_aperture_extraction_tab()
        self.setup_spectral_calibration_tab()
        self.setup_flux_calibration_tab()
        self.setup_output_tab()
        self.main_layout.addWidget(self.tabs)
        
        # Console output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.main_layout.addWidget(self.console)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def log_message(self, message):
        """Add message to console and status bar"""
        self.console.append(message)
        self.statusBar().showMessage(message)
        QApplication.processEvents()  # Update GUI immediately


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#   Read Files and do basic reduction     #
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

        
    def setup_file_selection_tab(self):
        """Tab for selecting input files"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File patterns group
        file_group = QGroupBox("File Patterns")
        file_layout = QVBoxLayout(file_group)

        # YAML file
        self.workdir = None
        yaml_layout = QHBoxLayout()
        yaml_layout.addWidget(QLabel("YAML file:"))
        # NOTE: was hardcoded to a personal path (/opt/test_spec_code/fileinfo.yaml).
        # Left blank by default -- use the "Browse" button, or set your own
        # default here if you always keep configs in the same place.
        self.yaml_pattern = QLineEdit("")
        yaml_layout.addWidget(self.yaml_pattern)
        yaml_browse_button = QPushButton("Browse...")
        yaml_browse_button.clicked.connect(self.browse_yaml_file)
        yaml_layout.addWidget(yaml_browse_button)
        file_layout.addLayout(yaml_layout)
        # Read yaml file button
        yread_button = QPushButton("Read YAML file")
        yread_button.clicked.connect(self.yread_files)
        file_layout.addWidget(yread_button)
        
        # Bias files
        bias_layout = QHBoxLayout()
        bias_layout.addWidget(QLabel("Bias:"))
        self.bias_pattern = QLineEdit("bias*fit")
        bias_layout.addWidget(self.bias_pattern)
        file_layout.addLayout(bias_layout)

        # Flat files
        flat_layout = QHBoxLayout()
        flat_layout.addWidget(QLabel("Flat:"))
        self.flat_pattern = QLineEdit("Lamp3*676R*fit")
        flat_layout.addWidget(self.flat_pattern)
        file_layout.addLayout(flat_layout)
        
        # Science files
        sci_layout = QHBoxLayout()
        sci_layout.addWidget(QLabel("Science:"))
        self.sci_pattern = QLineEdit("J08191752*fit")
        #self.sci_pattern.setPlaceholderText("e.g., NGC1068_1arcsec_676R_00.fit as NGC1068*fit")
        sci_layout.addWidget(self.sci_pattern)
        file_layout.addLayout(sci_layout)
        
        # Lamp files
        lamp_layout = QHBoxLayout()
        lamp_layout.addWidget(QLabel("Lamp:"))
        self.lamp_pattern = QLineEdit("lamp*676R*fit")
        lamp_layout.addWidget(self.lamp_pattern)
        file_layout.addLayout(lamp_layout)
        
        # Standard star files
        std_layout = QHBoxLayout()
        std_layout.addWidget(QLabel("Standard:"))
        self.std_pattern = QLineEdit("Feige34*fit")
        #self.std_pattern.setPlaceholderText("e.g., GD71_1arcsec_676R_00.fit as GD71*fit")
        std_layout.addWidget(self.std_pattern)
        file_layout.addLayout(std_layout)
        
        # Object names
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Science Object Name:"))
        self.sci_name = QLineEdit("J08191752")
        #self.sci_name.setPlaceholderText("NGC1068")
        name_layout.addWidget(self.sci_name)
        name_layout.addWidget(QLabel("Standard Star Name:"))
        self.std_name = QLineEdit("Feige34")
        #self.std_name.setPlaceholderText("GD71")
        name_layout.addWidget(self.std_name)
        file_layout.addLayout(name_layout)
        
        # Working directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Working Directory:"))
        self.dir_path = QLineEdit()
        if self.workdir !=None:
            print("workdir ",self.workdir)
            self.dir_path.setText(self.workdir)
        else:
            #self.dir_path=QLineEdit("/disk1/ADFOSC_pp/RED_Spec/RED_CLAGN/20250130/")
            self.dir_path.setPlaceholderText("Select directory...")
            dir_button = QPushButton("Browse...")
            dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(self.dir_path)
        #dir_layout.addWidget(dir_button)
        file_layout.addLayout(dir_layout)
        ## resource directory
        res_name_layout = QHBoxLayout()
        res_name_layout.addWidget(QLabel("Resource path Name:"))
        self.resource_path = QLineEdit(os.path.join(script_dir,"resources"))
        res_name_layout.addWidget(self.resource_path)
        file_layout.addLayout(res_name_layout)
        print(f" the resource path = {self.resource_path.text()}")
        # Trim checkbox
        self.trim_check = QCheckBox("Trim images")
        self.trim_check.setChecked(True)
        file_layout.addWidget(self.trim_check)
        
        # Spatial axis
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("Spatial Axis:"))
        self.spatial_axis = QComboBox()
        self.spatial_axis.addItems(["0", "1"])
        self.spatial_axis.setCurrentIndex(0)
        axis_layout.addWidget(self.spatial_axis)

        axis_layout.addWidget(QLabel("Slit:"))
        self.slit = QComboBox()
        self.slit.addItems(['1"','1.5"','2"'])
        self.slit.setCurrentIndex(1)
        axis_layout.addWidget(self.slit)

        axis_layout.addWidget(QLabel("GRISM:"))
        self.grism = QComboBox()
        self.grism.addItems(['770R-300gr/mm','132R-600gr/mm','676R-420gr/mm'])
        self.grism.setCurrentIndex(2)
        axis_layout.addWidget(self.grism)
        
        file_layout.addLayout(axis_layout)
        
        # Read files button
        read_button = QPushButton("Read Files + Bias Correction")
        read_button.clicked.connect(self.read_files)
        file_layout.addWidget(read_button)
        
        layout.addWidget(file_group)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "File Selection")

    def select_directory(self):
        """Select working directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_path.setText(directory)
            
    def browse_yaml_file(self):
        """Open a file picker for the reduction config YAML (fileinfo.yaml)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reduction config YAML", "", "YAML files (*.yaml *.yml)"
        )
        if path:
            self.yaml_pattern.setText(path)

    def yread_files(self):
        yfile = str(self.yaml_pattern.text())
        print(" the yaml file ",yfile)
        with open(yfile,'r') as file:
            yamlfile = yaml.load(file,Loader=yaml.SafeLoader)

        self.workdir             = yamlfile["workdir"]
        self.biaspattern         = yamlfile["bias_pattern"]
        self.flatpattern         = yamlfile["continuum_lamp"]
        self.arcpattern          = yamlfile["arc_lamp"]["pattern"]
        self.arclampname         = yamlfile["arc_lamp"]["name"]
        self.standstarpattern    = yamlfile["standard_star"]["pattern"]
        self.standstar_name      = yamlfile["standard_star"]["name"]
        self.sciobjpattern       = yamlfile["science_obj"]["pattern"]
        self.sciobj_name         = yamlfile["science_obj"]["name"]
        std = yamlfile.get("standard_star", {})
        std_dispersion = std.get("dispersion", "510")
        std_xlim = std.get("xlim", "1500")

        self.dir_path.setText(self.workdir)
        self.bias_pattern.setText(self.biaspattern)
        self.flat_pattern.setText(self.flatpattern)
        self.lamp_pattern.setText(self.arcpattern)
        self.std_pattern.setText(self.standstarpattern)
        self.std_name.setText(self.standstar_name)
        self.sci_pattern.setText(self.sciobjpattern)
        self.sci_name.setText(self.sciobj_name)
        self.arc_file.setText(self.arclampname)
        self.std_name_edit.setText(self.standstar_name)

        if self.std_table.rowCount() == 0:
            self.std_table.setRowCount(1)
            self.std_table.setItem(0, 1, QTableWidgetItem(str(std_dispersion)))
            self.std_table.setItem(0, 2, QTableWidgetItem(str(std_xlim)))
        else:
            if not self.std_table.item(0, 1):
                self.std_table.setItem(0, 1, QTableWidgetItem())
            if not self.std_table.item(0, 2):
                self.std_table.setItem(0, 2, QTableWidgetItem())
            self.std_table.item(0, 1).setText(str(std_dispersion))
            self.std_table.item(0, 2).setText(str(std_xlim))
    
    def read_files(self):
        """Read input files based on patterns"""
        try:
            dirname = self.dir_path.text()
            if not dirname:
                raise ValueError("Please select a working directory")
            else:
                os.chdir(dirname)
            
            self.log_message("Reading files...")
            print(f" currently workin on {os.getcwd()}")
            # Call your readfile_v2.readfile function
            foldername, self.sciname, self.stdname, self.flatin = readfile_v2.readfile(
                Bias=self.bias_pattern.text(),
                Flat=self.flat_pattern.text(),
                Sci=self.sci_pattern.text(),
                Lamp=self.lamp_pattern.text(),
                Std=self.std_pattern.text(),
                Sci_name=self.sci_name.text(),
                Std_name=self.std_name.text(),
                trim=self.trim_check.isChecked(),
                # instrument-specific trim box now comes from
                # specred/instruments/<name>.yaml instead of being
                # hardcoded in specred.io
                trim_box=get_trim_box(load_instrument(self.instrument_name)),
            )
            
            # Update the file lists
            self.sci_lst = [item for item in self.flatin if f"{self.sciname}" in item]
            self.std_lst = [item for item in self.flatin if f"{self.stdname}" in item]
            self.arclst = [item for item in self.flatin if "LAMP" in item.upper()][0]
            
            self.log_message(f"Found {len(self.sci_lst)} science files")
            self.log_message(f"Found {len(self.std_lst)} standard star files")
            self.log_message(f"Arc file: {self.arclst}")
            
            # Update the spatial axis
            global sxs, SXS
            sxs = int(self.spatial_axis.currentText())
            SXS = sxs
            
            self.log_message("File reading completed successfully")
            os.chdir(foldername)
            
            # Enable next tab
            self.tabs.setTabEnabled(1, True)
            
        except Exception as e:
            self.log_message(f"Error reading files: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to read files: {str(e)}")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#    Flat and Cosmic-ray correction       #
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def setup_flat_correction_tab(self):
        """Tab for flat field correction"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Flat correction group
        flat_group = QGroupBox("Flat Field Correction")
        flat_layout = QVBoxLayout(flat_group)
        
        # Master flat options
        master_layout = QHBoxLayout()
        master_layout.addWidget(QLabel("Master Flat Output:"))
        self.master_flat_name = QLineEdit("masterflat.fit")
        master_layout.addWidget(self.master_flat_name)
        flat_layout.addLayout(master_layout)
        
        # Flat correction options
        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel("Smoothing:"))
        self.smooth_check = QCheckBox()
        self.smooth_check.setChecked(True)
        options_layout.addWidget(self.smooth_check)
        
        options_layout.addWidget(QLabel("Display:"))
        self.display_flat_check = QCheckBox()
        options_layout.addWidget(self.display_flat_check)
        flat_layout.addLayout(options_layout)
        
        # Corrected flat output
        corrected_layout = QHBoxLayout()
        corrected_layout.addWidget(QLabel("Corrected Flat Output:"))
        self.corrected_flat_name = QLineEdit("nflat.fits")
        corrected_layout.addWidget(self.corrected_flat_name)
        flat_layout.addLayout(corrected_layout)

        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel("Cosmic-ray correction:"))
        self.cosmic_corr = QCheckBox()
        self.cosmic_corr.setChecked(True)
        options_layout.addWidget(self.cosmic_corr)
        flat_layout.addLayout(options_layout)
        
        # Perform flat correction button
        flat_button = QPushButton("Perform Flat Correction (Flat Correction and Cosmic-ray removal)")
        flat_button.clicked.connect(self.perform_flat_correction)
        flat_layout.addWidget(flat_button)
        
        layout.addWidget(flat_group)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Flat Correction")

    
    def perform_flat_correction(self):
        """Perform flat field correction"""
        try:
            if not hasattr(self, 'flatin') or not self.flatin:
                raise ValueError("Please read files first")
            
            self.log_message("Performing flat field correction...")
            
            # Call your flat_2 functions
            flat_2.lamp_response(
                self.master_flat_name.text(),
                func=True,
                smooth=self.smooth_check.isChecked(),
                Saxis=sxs,
                display=self.display_flat_check.isChecked()
                
            )
            
            self.lst = flat_2.flat_correction(self.flatin, self.corrected_flat_name.text(),cosmic_correction = self.cosmic_corr.isChecked())
            self.log_message(f"Flat correction completed. Output: {self.lst}")
            
            # Update file lists after flat correction
            self.sci_lst = [item for item in self.lst if f"{self.sciname}" in item]
            self.std_lst = [item for item in self.lst if f"{self.stdname}" in item]
            self.arclst = [item for item in self.lst if "LAMP" in item.upper()][0]
            
            self.log_message(f"Updated science files: {len(self.sci_lst)}")
            self.log_message(f"Updated standard files: {len(self.std_lst)}")
            self.log_message(f"Updated arc file: {self.arclst}")
            SLIT = self.slit.currentText()
            GRISM = self.grism.currentText()
            print(f" the selected config is SLIT:{SLIT} and {GRISM}")
            
            # Enable next tab
            self.tabs.setTabEnabled(2, True)
            
        except Exception as e:
            self.log_message(f"Error in flat correction: {str(e)}")
            QMessageBox.critical(self, "Error", f"Flat correction failed: {str(e)}")
            

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#         APERTURE EXTRACTION             #
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def setup_aperture_extraction_tab(self):
        """Tab for aperture extraction with preview and parameter input"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 1. Preview Section
        preview_group = QGroupBox("Image Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Preview figure
        self.preview_figure = Figure()
        self.preview_canvas = FigureCanvas(self.preview_figure)
        preview_layout.addWidget(self.preview_canvas)
        
        # Preview controls
        preview_controls = QHBoxLayout()
        
        # File selection
        refresh_btn = QPushButton("Refresh Files")
        refresh_btn.clicked.connect(self.update_file_lists)
        preview_controls.addWidget(refresh_btn)

        self.preview_file_combo = QComboBox()
        preview_controls.addWidget(QLabel("Select File:"))
        preview_controls.addWidget(self.preview_file_combo)
        
        # Guess value input
        preview_controls.addWidget(QLabel("Guess Value:"))
        self.preview_guess_input = QSpinBox()
        self.preview_guess_input.setRange(0, 2048)
        preview_controls.addWidget(self.preview_guess_input)
        
        # X Limit input
        preview_controls.addWidget(QLabel("X Limit:"))
        self.preview_xlim_input = QSpinBox()
        self.preview_xlim_input.setRange(0, 2048)
        self.preview_xlim_input.setValue(1500)
        preview_controls.addWidget(self.preview_xlim_input)
        
        # Preview button
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.preview_selected_image)
        preview_controls.addWidget(preview_btn)
        
        preview_layout.addLayout(preview_controls)
        layout.addWidget(preview_group)
        
        # 2. Science Object Section
        sci_group = QGroupBox("Science Object Extraction")
        sci_layout = QVBoxLayout(sci_group)
        
        # Science files table with Combine ID column
        self.sci_table = QTableWidget(0, 8)  # 8 columns
        self.sci_table.setHorizontalHeaderLabels(["File", "Guess Value", "X Limit", "Nbin","Ap width","Sky sep", "Method", "Combine ID"])   #2207
        self.sci_table.horizontalHeader().setStretchLastSection(True)
        sci_layout.addWidget(self.sci_table)
        
        # Science controls
        sci_controls = QHBoxLayout()
        
        # Combine Group selection
        sci_controls.addWidget(QLabel("Combine Group:"))
        self.sci_combine_group = QComboBox()
        self.sci_combine_group.addItems(["All", "0", "1", "2", "3"])
        sci_controls.addWidget(self.sci_combine_group)
        
        # Extract Science button
        extract_sci_btn = QPushButton("Extract Science")
        extract_sci_btn.clicked.connect(self.extract_science)
        sci_controls.addWidget(extract_sci_btn)
        
        # Combine Science Group button
        combine_sci_btn = QPushButton("Combine Science Group")
        combine_sci_btn.clicked.connect(lambda: self.combine_spectra('science', group=self.sci_combine_group.currentText()))
        sci_controls.addWidget(combine_sci_btn)
        
        # Combine Method selection
        sci_controls.addWidget(QLabel("Combine Method:"))
        self.sci_combine_method = QComboBox()
        self.sci_combine_method.addItems(["mean", "invvar", "median"])
        self.sci_combine_method.setCurrentIndex(0)
        sci_controls.addWidget(self.sci_combine_method)
        
        sci_layout.addLayout(sci_controls)
        layout.addWidget(sci_group)
        
        # 3. Standard Star Section (similar structure)
        std_group = QGroupBox("Standard Star Extraction")
        std_layout = QVBoxLayout(std_group)
        
        self.std_table = QTableWidget(0, 6)  # 6 columns
        self.std_table.setHorizontalHeaderLabels(["File", "Guess Value", "X Limit", "Nbin", "Method", "Combine ID"])
        self.std_table.horizontalHeader().setStretchLastSection(True)
        std_layout.addWidget(self.std_table)
        
        # Standard controls
        std_controls = QHBoxLayout()
        
        std_controls.addWidget(QLabel("Combine Group:"))
        self.std_combine_group = QComboBox()
        self.std_combine_group.addItems(["All", "0", "1", "2", "3"])
        std_controls.addWidget(self.std_combine_group)
        
        extract_std_btn = QPushButton("Extract Standard")
        extract_std_btn.clicked.connect(self.extract_standard)
        std_controls.addWidget(extract_std_btn)
        
        combine_std_btn = QPushButton("Combine Standard Group")
        combine_std_btn.clicked.connect(lambda: self.combine_spectra('standard', group=self.std_combine_group.currentText()))
        std_controls.addWidget(combine_std_btn)
        
        std_controls.addWidget(QLabel("Combine Method:"))
        self.std_combine_method = QComboBox()
        self.std_combine_method.addItems(["mean", "invvar", "median"])
        self.std_combine_method.setCurrentIndex(0)
        std_controls.addWidget(self.std_combine_method)
        
        std_layout.addLayout(std_controls)
        layout.addWidget(std_group)
        
        self.tabs.addTab(tab, "Aperture Extraction")


    def aperture_extract(self, sci_lst, otype="src", arcfile=None, sourcename=None, 
                        initial_guess=500, method="sum", xlim=1500, nbin=7,apw=6,skysep=6,
                        display_trace=True, display_extract=True):
        """Wrapper for aperture extraction function with GUI integration"""
        AP_data = []
        source_am = []
        ARC_data = []
        
        for idx, sciff in enumerate(np.unique(sci_lst)):
            self.log_message(f'Processing file {sciff}')
            self.log_message(f'Using guess value: {initial_guess}')
            
            scif = ccdproc.CCDData.read(sciff, unit="adu")
            scif = ccdproc.CCDData(
                np.nan_to_num(scif.data, copy=True, nan=0.0, posinf=0.0, neginf=0.0),
                unit="adu",
                header=scif.header
            )
            if otype=="src":
                self.sci_mjd = round((float(scif.header["JD"]) - 2400000.5),5)      ## taking MJD value while saving the final csv file
                print("sci mjd = ",int(self.sci_mjd))
            
            Airmass = airmass.Airmass(sciff, sourcename)
            source_am.append(Airmass)
            name = f"A_"+sciff
            
            # Trace function
            sci_tr = aperture.trfunc(
                scif,
                nbins=int(nbin),
                window=20,
                guess=initial_guess,
                Saxis=sxs,
                Xlim=int(xlim),
                display=display_trace,
                otype=otype,
                filename=name
            )
            #print("done JJ 0")
            # Boxcar extraction
            sci_ex, sky_sci = aperture.boxfunc(
                scif,
                sci_tr,
                Saxis=sxs,
                display=display_extract,
                otype=otype,
                apwidth=apw,
                skysep=skysep,
                filename=name,
                airmass=Airmass,
                method=method
            )
            #print("done JJ 1")
            #print(f"{name} \n hdr {scif.header}")
            Sci_ex = Spectrum1D(
                flux=(sci_ex.flux - sky_sci.flux),
                spectral_axis=sci_ex.spectral_axis,
                uncertainty=StdDevUncertainty(sci_ex.uncertainty.array),
                meta={'header':scif.header}
            )
            #print("done JJ 2")
            AP_data.append(Sci_ex)
            scif.header["airmass"] = Airmass
            name = name.replace(".fit", "").replace(".fits", "")
            fits.writeto(name+".fits", Sci_ex.flux.value, scif.header, overwrite=True)
            self.newher = scif.header
            #print("done JJ 3")
            if otype == "src" and idx == 0:
                if arcfile is None:
                    self.log_message("Warning: No arcfile provided for aperture extraction")
                else:
                    arcf = ccdproc.CCDData.read(arcfile, unit="adu")
                    arcf = ccdproc.CCDData(
                        np.nan_to_num(arcf.data, copy=True, nan=0.0, posinf=0.0, neginf=0.0),
                        unit="adu",
                        header=scif.header
                    )
                    sciarc_ex, _ = aperture.BoxcarExtract(
                        arcf,
                        sci_tr,
                        Saxis=sxs,
                        filename=self.lst[2],
                        log_file=None,
                        method=method
                    )
                    ARC_data.append(sciarc_ex)
                    arcname = f"A_ap_arc_"+arcfile
                    fits.writeto(arcname, sciarc_ex.flux.value, arcf.header, overwrite=True)
        #print("done JJ 4")
        if otype == "src":
            return AP_data, source_am, ARC_data
        else:
            return AP_data, source_am

        
    def preview_selected_image(self):
        """Preview the selected image with current parameters"""
        try:
            filename = self.preview_file_combo.currentText()
            if not filename:
                return
                
            guess = self.preview_guess_input.value()
            xlim = self.preview_xlim_input.value()
            
            # Read and display the image
            with fits.open(filename) as hdul:
                data = hdul[0].data if hdul[0].data is not None else hdul[1].data
                
            self.preview_figure.clear()
            ax = self.preview_figure.add_subplot(111)
            
            # Display with reasonable contrast
            vmin = np.nanpercentile(data, 5)
            vmax = np.nanpercentile(data, 95)
            
            ax.imshow(data, origin='lower', aspect='auto', 
                     cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(os.path.basename(filename))
            
            # Show guess line
            ax.axhline(y=guess, color='r', linestyle='--', alpha=0.7)
            ax.text(100, guess+80, f'Guess: {guess}', color='r',fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Show x limit
            if xlim > 0:
                ax.axvline(x=xlim, color='b', linestyle=':', alpha=0.7)
                ax.text(xlim+10, 10, f'X Limit: {xlim}', color='b',
                       bbox=dict(facecolor='white', alpha=0.7))
            
            self.preview_canvas.draw()
            #### 2607
            fig_apwidth,ax_apwidth = pllt.subplots(figsize=(7,4))
            line = np.sum(data[:,500:550],axis=1)
            ax_apwidth.plot(line,color="r")
            ax_apwidth.set_xlabel("dispersion axis")
            pllt.show()
        except Exception as e:
            self.log_message(f"Error previewing image: {str(e)}")


    def update_file_lists(self):
        """Update file lists with Combine ID selection"""
        self.preview_file_combo.clear()
        all_files = self.sci_lst + self.std_lst + ([self.arclst] if self.arclst else [])
        self.preview_file_combo.addItems([os.path.basename(f) for f in all_files])
        
        # Science table
        self.sci_table.setRowCount(len(self.sci_lst))
        
        for i, filename in enumerate(self.sci_lst):
            # File name
            file_item = QTableWidgetItem(os.path.basename(filename))
            file_item.setFlags(file_item.flags() ^ Qt.ItemIsEditable)
            self.sci_table.setItem(i, 0, file_item)
            
            # Guess value
            guess_item = QTableWidgetItem("445")
            self.sci_table.setItem(i, 1, guess_item)
            
            # X Limit
            xlim_item = QTableWidgetItem("1500")
            self.sci_table.setItem(i, 2, xlim_item)
            
            # Nbin (assuming you want this editable)
            nbin_item = QTableWidgetItem("10")
            self.sci_table.setItem(i, 3, nbin_item)

            apw_item = QTableWidgetItem("6")
            self.sci_table.setItem(i, 4, apw_item)

            skysep_item = QTableWidgetItem("6")
            self.sci_table.setItem(i, 5, skysep_item)
            
            # Method combo
            method_combo = QComboBox()
            method_combo.addItems(["sum", "average"])
            self.sci_table.setCellWidget(i, 6, method_combo)
            
            # Combine ID combo
            combine_combo = QComboBox()
            combine_combo.addItems(["0", "1", "2", "3"])
            combine_combo.setCurrentIndex(0)
            self.sci_table.setCellWidget(i, 7, combine_combo)
        
        # Standard table (similar)
        self.std_table.setRowCount(len(self.std_lst))
        for i, filename in enumerate(self.std_lst):
            # File name
            file_item = QTableWidgetItem(os.path.basename(filename))
            file_item.setFlags(file_item.flags() ^ Qt.ItemIsEditable)
            self.std_table.setItem(i, 0, file_item)
            
            # Guess value
            guess_item = QTableWidgetItem("490")
            self.std_table.setItem(i, 1, guess_item)
            
            # X Limit
            xlim_item = QTableWidgetItem("1500")
            self.std_table.setItem(i, 2, xlim_item)
            
            # Nbin
            nbin_item = QTableWidgetItem("10")
            self.std_table.setItem(i, 3, nbin_item)
            
            # Method combo
            method_combo = QComboBox()
            method_combo.addItems(["sum", "average"])
            self.std_table.setCellWidget(i, 4, method_combo)
            
            # Combine ID combo
            combine_combo = QComboBox()
            combine_combo.addItems(["0", "1", "2", "3"])
            combine_combo.setCurrentIndex(0)
            self.std_table.setCellWidget(i, 5, combine_combo)
        
        self.sci_table.resizeColumnsToContents()
        self.std_table.resizeColumnsToContents()

    def extract_science(self):
        """Perform aperture extraction for science objects"""
        try:
            if not hasattr(self, 'sci_lst') or not self.sci_lst:
                raise ValueError("No science files available")
            
            self.log_message("Performing science object extraction...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            self.sci_results = []  # Initialize SCIdata
            self.sci_airmass = []
            
            for row in range(self.sci_table.rowCount()):
                filename = self.sci_lst[row]
                guess = int(self.sci_table.item(row, 1).text())
                xlim = int(self.sci_table.item(row, 2).text())
                nb = int(self.sci_table.item(row, 3).text())
                apw = int(self.sci_table.item(row,4).text())
                skysep = int(self.sci_table.item(row,5).text())
                method = self.sci_table.cellWidget(row, 6).currentText()
                
                self.log_message(f"Processing {os.path.basename(filename)}")
                self.log_message(f"Using guess={guess}, xlim={xlim}, method={method}")
                sci_data, airmass, arc_data = self.aperture_extract(
                    [filename],
                    otype="src",
                    arcfile=self.arclst,
                    sourcename=self.sciname,
                    initial_guess=guess,
                    method=method,
                    nbin=nb,
                    xlim=xlim,
                    apw =apw,
                    skysep = skysep,
                    display_trace=True,
                    display_extract=True
                )
                
                self.sci_results.extend(sci_data)
                self.sci_airmass.append(airmass[0])
                
                if row == 0:
                    self.ARCdata = arc_data
            self.SCIdata = self.sci_results
            self.log_message(f"Extracted {len(self.SCIdata)} science spectra")
            
        except Exception as e:
            error_msg = f"Error in science extraction: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
        finally:
            QApplication.restoreOverrideCursor()

    def extract_standard(self):
        """Perform aperture extraction for standard stars"""
        try:
            if not hasattr(self, 'std_lst') or not self.std_lst:
                raise ValueError("No standard files available")
            
            self.log_message("Performing standard star extraction...")
            print(f" the file are {self.std_lst} and len ={self.std_table.rowCount()}")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            self.STDdata = []  # Initialize STDdata
            self.std_airmass = []
            #print(f" Table = ,{self.std_table.items()}" )

            #print(f" method = ,{self.std_table.cellWidget(0, 4).currentText()},,{self.std_table.cellWidget(1, 4).currentText()}" )
            for row in range(self.std_table.rowCount()):
                filename = self.std_lst[row]
                self.log_message(f"for  {filename} {row}")
                #print(f" guess = ,{self.std_table.item(0,1).text()},,{self.std_table.item(1,1).text()}" )
                guess = int(self.std_table.item(row, 1).text())
                #print(f" xlim = ,{self.std_table.item(0,2).text()},,{self.std_table.item(1,2).text()}" )
                xlim = int(self.std_table.item(row, 2).text())
                #print(f" nb = ,{self.std_table.item(0,3).text()},,{self.std_table.item(1,3).text()}" )
                nbb = int(self.std_table.item(row, 3).text())
                #print(f"{row}= {guess} | {xlim} | {nbb}")
                method = self.std_table.cellWidget(row, 4).currentText()
                #print(f"{row}= {method}")
                self.log_message(f"Processing {os.path.basename(filename)}")
                #print("done hehe 0")
                std_data, airmass = self.aperture_extract(
                    [filename],
                    otype="std",
                    sourcename=self.stdname,
                    initial_guess=guess,
                    method=str(method),
                    xlim=xlim,
                    nbin=nbb,
                    display_trace=True,
                    display_extract=True
                )
                #print("done hehe 1")
                #print(f"std_data = {std_data} \n airmass ={airmass}")
                self.STDdata.extend(std_data)
                #print("done hehe 2")
                self.std_airmass.append(airmass[0])
                #print("done hehe 3")
            self.log_message(f"Extracted {len(self.STDdata)} standard spectra")
            
        except Exception as e:
            error_msg = f"Error in standard extraction: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
        finally:
            QApplication.restoreOverrideCursor()


    def combine_spectra(self, spec_type, group="All"):
        """Combine spectra with group filtering and proper storage"""
        try:
            # Get reference to appropriate data and table
            if spec_type == 'science':
                data = getattr(self, 'SCIdata', [])
                table = self.sci_table
                method = self.sci_combine_method.currentText()
                prefix = 'SCI'
                combineID = 7
            else:
                data = getattr(self, 'STDdata', [])
                table = self.std_table
                method = self.std_combine_method.currentText()
                prefix = 'STD'
                combineID = 5

            if not data:
                raise ValueError(f"Please extract {spec_type} spectra first")

            # Initialize combined spectra storage if it doesn't exist
            if not hasattr(self, 'combined_spectra'):
                self.combined_spectra = {'SCI': {}, 'STD': {}}

            # Group spectra by Combine ID
            grouped_data = {}
            for i, spec in enumerate(data):
                if i < table.rowCount():
                    try:
                        combine_id = table.cellWidget(i, int(combineID)).currentText()  # Combine ID is column 5 for std and 7 for sci    #2207
                        print("combined_id",combine_id)
                        #print(combine_id.text())
                        if combine_id not in grouped_data:
                            grouped_data[combine_id] = []
                        grouped_data[combine_id].append(spec)
                    except Exception as e:
                        self.log_message(f"Error processing row {i}: {str(e)}")
                        continue

            # Process selected group(s)
            if group == "All":
                # Process all groups
                for gid, specs in grouped_data.items():
                    combined = self.combine_spectrum1d_stack(specs, method=method)
                    self.combined_spectra[prefix][gid] = combined
                    print(f"combining for {gid}")
                    self._save_combined_spectrum(combined, spec_type, gid)
            else:
                # Process specific group
                if group not in grouped_data:
                    raise ValueError(f"No {spec_type} spectra in group {group}")
                combined = self.combine_spectrum1d_stack(grouped_data[group], method=method)
                self.combined_spectra[prefix][group] = combined
                self._save_combined_spectrum(combined, spec_type, group)

            # For backward compatibility, set the first group as default comb_SCI/comb_STD
            if spec_type == 'science' and self.combined_spectra['SCI']:
                first_group = next(iter(self.combined_spectra['SCI']))
                setattr(self, 'comb_SCI', self.combined_spectra['SCI'][first_group])
            elif spec_type == 'standard' and self.combined_spectra['STD']:
                first_group = next(iter(self.combined_spectra['STD']))
                setattr(self, 'comb_STD', self.combined_spectra['STD'][first_group])

            self.log_message(f"Combined {len(data)} {spec_type} spectra (Group {group})")
            self.plot_combined_spectrum(spec_type, group)

        except Exception as e:
            error_msg = f"Error combining {spec_type} group {group}: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _save_combined_spectrum(self, spectrum, spec_type, group_id):
        """Save combined spectrum to FITS file with proper naming and headers, 
        including only the files that were combined for this specific group"""
        try:
            # Determine base name based on spectrum type
            if spec_type == 'science':
                base_name = self.sciname
                table = self.sci_table
                file_list = self.sci_lst
                combineID = 7
            else:
                base_name = self.stdname
                table = self.std_table
                file_list = self.std_lst
                combineID=5

            # Create output filename
            output_name = f"combined_{base_name}_group{group_id}.fits"
            
            # Get headers from spectrum meta if available
            headers = spectrum.meta.get('headers', [])
            
            # Create primary HDU with flux data
            primary_hdu = fits.PrimaryHDU(spectrum.flux.value)
            
            # If we have headers, use the first one as the base header
            if headers:
                primary_hdu.header = headers[0].copy()
            
            # Get the actual files that were combined for this group
            combined_files = []
            for row in range(table.rowCount()):
                # Check if this file belongs to our group
                current_group = table.cellWidget(row, int(combineID)).currentText()  # Combine ID is column 5
                if current_group == group_id or (group_id == "All" and current_group != ""):
                    combined_files.append(file_list[row])
            
            # Add combination information to header
            primary_hdu.header['COMBINE'] = (True, 'File is a combination of spectra')
            primary_hdu.header['COMBMETH'] = (self.sci_combine_method.currentText() if spec_type == 'science' 
                                            else self.std_combine_method.currentText(), 'Combination method')
            primary_hdu.header['COMBGRP'] = (group_id, 'Combination group ID')
            primary_hdu.header['NCOMBINE'] = (len(combined_files), 'Number of spectra combined')
            
            # Add list of input files that were actually combined
            for i, filename in enumerate(combined_files):
                primary_hdu.header[f'INFIL{i+1:03d}'] = (os.path.basename(filename), f'Input file {i+1}')
            
            # Create HDU list
            hdus = [primary_hdu]
            
            # Add uncertainty if available
            if spectrum.uncertainty is not None:
                uncert_hdu = fits.ImageHDU(spectrum.uncertainty.array, name='UNCERTAINTY')
                hdus.append(uncert_hdu)
            
            # Add wavelength if available
            if hasattr(spectrum, 'spectral_axis'):
                wave_hdu = fits.ImageHDU(spectrum.spectral_axis.value, name='WAVELENGTH')
                hdus.append(wave_hdu)
            
            # Write to file
            hdulist = fits.HDUList(hdus)
            hdulist.writeto(output_name, overwrite=True)
            self.log_message(f"Saved combined spectrum: {output_name} (Group {group_id}, {len(combined_files)} files)")

        except Exception as e:
            self.log_message(f"Error saving combined spectrum: {str(e)}")
            raise

    def combine_spectrum1d_stack(self, SCIdata, method="mean"):
        """Combine a list of Spectrum1D objects into one final spectrum"""
        flux_list = []
        fluxerr_list = []
        headers=[]
        #print(f"the input files for combined {SCIdata} ")
        for spec in SCIdata:
            flux_list.append(spec.flux.value)
            if spec.uncertainty is not None:
                fluxerr_list.append(spec.uncertainty.array)
            else:
                raise ValueError("Spectrum is missing uncertainty.")
            if hasattr(spec,'meta') and 'header' in spec.meta:
                headers.append(spec.meta["header"])

        flux_stack = np.array(flux_list)
        fluxerr_stack = np.array(fluxerr_list)

        # Call combine_spectra method
        combined_flux, combined_fluxerr = self.combine_flux_arrays(flux_stack, fluxerr_stack, method=method)

        # Use the spectral axis from the first spectrum
        spectral_axis = SCIdata[0].spectral_axis
        #print(f" spectral_axis for combined file {spectral_axis}")
        # Build final Spectrum1D object
        combined_spec = Spectrum1D(
            flux=combined_flux * SCIdata[0].flux.unit,
            uncertainty=StdDevUncertainty(combined_fluxerr),
            spectral_axis=spectral_axis,
            meta={"headers":headers}
        )
        return combined_spec
    
    def combine_flux_arrays(self, flux_stack, fluxerr_stack, method="invvar"):
        """Combine flux arrays using specified method"""
        mask = ~np.isfinite(flux_stack) | ~np.isfinite(fluxerr_stack)
        flux_stack = np.where(mask, np.nan, flux_stack)
        fluxerr_stack = np.where(mask, np.nan, fluxerr_stack)
        
        if method == "invvar":
            weights = 1.0 / np.square(fluxerr_stack)
            weights[~np.isfinite(weights)] = 0.0

            weighted_flux = np.nansum(flux_stack * weights, axis=0)
            sum_weights = np.nansum(weights, axis=0)

            combined_flux = np.divide(weighted_flux, sum_weights, 
                                    out=np.full_like(weighted_flux, np.nan), 
                                    where=sum_weights>0)
            combined_fluxerr = np.sqrt(1.0 / sum_weights)
        
        elif method == "mean":
            # For mean, use error propagation for weighted mean
            weights = 1.0 / np.square(fluxerr_stack)
            weights[~np.isfinite(weights)] = 0.0
            
            weighted_flux = np.nansum(flux_stack * weights, axis=0)
            sum_weights = np.nansum(weights, axis=0)
            
            combined_flux = np.divide(weighted_flux, sum_weights,
                                    out=np.full_like(weighted_flux, np.nan),
                                    where=sum_weights>0)
            combined_fluxerr = np.sqrt(np.nansum(weights**2 * fluxerr_stack**2, axis=0)) / sum_weights
        
        elif method == "median":
            # For median, use median absolute deviation scaled by input errors
            combined_flux = np.nanmedian(flux_stack, axis=0)
            
            # Calculate weighted median absolute deviation
            weights = 1.0 / fluxerr_stack
            weights[~np.isfinite(weights)] = 0.0
            
            # Calculate weighted MAD
            abs_dev = np.abs(flux_stack - combined_flux)
            weighted_abs_dev = weights * abs_dev
            mad = 1.4826 * np.nanmedian(weighted_abs_dev, axis=0)
            
            # Scale by number of observations
            n_obs = np.sum(~np.isnan(flux_stack), axis=0)
            combined_fluxerr = mad / np.sqrt(n_obs)
        
        else:
            raise ValueError(f"Unsupported method: {method}")

        return combined_flux, combined_fluxerr

    def plot_combined_spectrum(self, spec_type, group="All"):
        """Plot combined spectrum with group info"""
        attr_name = f'comb_{spec_type.upper()}_group{group}'
        if hasattr(self, attr_name):
            combined_spec = getattr(self, attr_name)
            self.preview_figure.clear()
            ax = self.preview_figure.add_subplot(111)
            ax.plot(combined_spec.spectral_axis, combined_spec.flux)
            ax.set_title(f'Combined {spec_type.capitalize()} (Group {group})')
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Counts')
            self.preview_canvas.draw()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#        WAVELENGTH CORRECTION            #
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def setup_spectral_calibration_tab(self):
        """Tab for spectral calibration with interactive steps"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Spectral calibration group
        spec_group = QGroupBox("Spectral Calibration")
        spec_layout = QVBoxLayout(spec_group)
        
        # ===== 1. Arc Line File Selection =====
        arc_layout = QHBoxLayout()
        arc_layout.addWidget(QLabel("Arc Line File:"))

        # User input for filename
        self.arc_file = QLineEdit()
        self.arc_file.setPlaceholderText("e.g., HgNe.dat")

        # "Go" button to confirm selection
        go_button = QPushButton("Go")
        go_button.clicked.connect(self.handle_arc_selection)

        # Browse button (hidden by default)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.select_arc_file)
        self.browse_button.hide()  # Initially hidden
        
        arc_layout.addWidget(self.arc_file)
        arc_layout.addWidget(go_button)
        arc_layout.addWidget(self.browse_button)
        spec_layout.addLayout(arc_layout)
        # ===== 2. Initial Wavelength Estimation =====
        init_group = QGroupBox("Step 1: Initial Wavelength Solution")
        init_layout = QVBoxLayout(init_group)
        
        # Add line slider button
        self.estimate_btn = QPushButton("Estimate Offset/Dispersion")
        self.estimate_btn.clicked.connect(self.estimate_wavelength_solution)
        init_layout.addWidget(self.estimate_btn)
        
        # Display estimated values
        est_layout = QHBoxLayout()
        est_layout.addWidget(QLabel("Estimated Offset:"))
        self.offset_display = QLabel("Not calculated")
        est_layout.addWidget(self.offset_display)
        
        est_layout.addWidget(QLabel("Estimated Dispersion:"))
        self.dispersion_display = QLabel("Not calculated")
        est_layout.addWidget(self.dispersion_display)
        init_layout.addLayout(est_layout)
        
        # Confirmation button
        self.confirm_initial_btn = QPushButton("Confirm Initial Solution")
        self.confirm_initial_btn.setEnabled(False)
        self.confirm_initial_btn.clicked.connect(self.confirm_initial_solution)
        init_layout.addWidget(self.confirm_initial_btn)
        
        spec_layout.addWidget(init_group)
        
        # ===== 3. Line Identification =====
        id_group = QGroupBox("Step 2: Line Identification")
        id_layout = QVBoxLayout(id_group)
        
        # Parameters
        id_params_layout = QHBoxLayout()
        id_params_layout.addWidget(QLabel("Func"))
        self.line_func = QComboBox()
        self.line_func.addItems(["spline","linear","slinear","cubic","legendre","polyfit","zero",'quadratic'])
        self.line_func.setCurrentIndex(0)
        id_params_layout.addWidget(self.line_func)

        id_params_layout.addWidget(QLabel("Deg:"))
        self.deg_val = QSpinBox()
        self.deg_val.setRange(1, 100)
        self.deg_val.setValue(3)
        id_params_layout.addWidget(self.deg_val)

        id_params_layout.addWidget(QLabel("Search Width:"))
        self.search_width = QSpinBox()
        self.search_width.setRange(1, 100)
        self.search_width.setValue(10)
        id_params_layout.addWidget(self.search_width)
        
        id_params_layout.addWidget(QLabel("Threshold:"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.1, 1.0)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.97)
        id_params_layout.addWidget(self.threshold)
        id_layout.addLayout(id_params_layout)
        
        # Refinement button
        self.identify_btn = QPushButton("Identify Lines")
        self.identify_btn.setEnabled(False)
        self.identify_btn.clicked.connect(self.identify_lines)
        id_layout.addWidget(self.identify_btn)
        
        # Confirmation button
        self.confirm_identify_btn = QPushButton("Confirm Identified Lines")
        self.confirm_identify_btn.setEnabled(False)
        self.confirm_identify_btn.clicked.connect(self.confirm_line_identification)
        id_layout.addWidget(self.confirm_identify_btn)
        
        spec_layout.addWidget(id_group)
        
        # ===== 4. Refinement Steps =====
        refine_group = QGroupBox("Step 3: Refinement")
        refine_layout = QVBoxLayout(refine_group)
        
        # First refinement
        refine1_layout = QHBoxLayout()
        refine1_layout.addWidget(QLabel("Width:"))
        self.refine_width = QSpinBox()
        self.refine_width.setRange(1, 50)
        self.refine_width.setValue(10)
        refine1_layout.addWidget(self.refine_width)
        
        refine1_layout.addWidget(QLabel("Threshold:"))
        self.refine_threshold = QDoubleSpinBox()
        self.refine_threshold.setRange(0.1, 1.0)
        self.refine_threshold.setSingleStep(0.01)
        self.refine_threshold.setValue(0.94)
        refine1_layout.addWidget(self.refine_threshold)
        
        self.refine1_btn = QPushButton("First Refinement")
        self.refine1_btn.setEnabled(False)
        self.refine1_btn.clicked.connect(self.first_refinement)
        refine1_layout.addWidget(self.refine1_btn)
        refine_layout.addLayout(refine1_layout)
        
        # Final refinement
        refine2_layout = QHBoxLayout()
        refine2_layout.addWidget(QLabel("Width:"))
        self.final_refine_width = QSpinBox()
        self.final_refine_width.setRange(1, 50)
        self.final_refine_width.setValue(10)
        refine2_layout.addWidget(self.final_refine_width)
        
        refine2_layout.addWidget(QLabel("Threshold:"))
        self.final_refine_threshold = QDoubleSpinBox()
        self.final_refine_threshold.setRange(0.1, 1.0)
        self.final_refine_threshold.setSingleStep(0.01)
        self.final_refine_threshold.setValue(0.9)
        refine2_layout.addWidget(self.final_refine_threshold)
        
        self.refine2_btn = QPushButton("Final Refinement")
        self.refine2_btn.setEnabled(False)
        self.refine2_btn.clicked.connect(self.final_refinement)
        refine2_layout.addWidget(self.refine2_btn)
        refine_layout.addLayout(refine2_layout)
        
        # Final confirmation
        self.confirm_final_btn = QPushButton("Confirm Final Solution")
        self.confirm_final_btn.setEnabled(False)
        self.confirm_final_btn.clicked.connect(self.confirm_final_solution)
        refine_layout.addWidget(self.confirm_final_btn)
        
        spec_layout.addWidget(refine_group)
        # ===== 5. Full Calibration Button =====
        cal_button = QPushButton("Complete Spectral Calibration")
        cal_button.clicked.connect(self.perform_spectral_calibration)
        spec_layout.addWidget(cal_button)
        
        layout.addWidget(spec_group)
        self.tabs.addTab(tab, "Spectral Calibration")


    def handle_arc_selection(self):
        """Handle arc file selection with proper error handling"""
        try:
            # First try the default arc directory
            filename = self.arc_file.text().strip()
            if not filename:
                QMessageBox.warning(self, "Missing Input", "Please enter an arc filename!")
                return
            print(f"f the resource_path = {self.resource_path} | {self.resource_path.text()}")  
            default_path = os.path.join(self.resource_path.text(), "arc", str(filename))
            
            if os.path.exists(default_path):
                print(f"Using arc file: {default_path}")
                self.browse_button.hide()
                self.arc_file.setText(str(default_path))
                return default_path  # Success case
                
            # If not found, ask user if they want to browse
            reply = QMessageBox.question(
                self,
                "File Not Found",
                f"'{filename}' not found in arc directory.\nWould you like to browse for it?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.browse_button.show()
            else:
                self.arc_file.setFocus()  # Let user edit the filename
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
            return None

    def select_arc_file(self):
        """File dialog that preserves the arc filename display"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Arc Line File",
            "",  # Start in current directory
            "Data Files (*.dat *.txt);;All Files (*)"
        )
        
        if filename:
            # Store full path internally but only show filename
            self._actual_arc_path = str(filename)
            self.arc_file.setText(os.path.basename(filename))
            self.browse_button.hide()
            
    def get_arc_path(self):
        """Returns full path or raises exceptions"""
        filename = self.arc_file.text().strip()
        if not filename:
            raise ValueError("Please enter an arc filename!")
        print(f" the filenam provided {filename}")
        full_path = os.path.join(str(self.resource_path), "arc", filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError()
        
        return full_path


    def estimate_wavelength_solution(self):
        """Interactive estimation of offset/dispersion"""
        try:
            # Validate we have the required data
            if not hasattr(self, 'ARCdata'):
                raise ValueError("No arc data loaded")
            
            # Validate arc filename
            arcfile = self.arc_file.text().strip()
            if not arcfile:
                QMessageBox.warning(self, "Input Needed", "Please enter an arc filename first")
                return
            
            # Get full path to arc file
            try:
                full_path = self.get_arc_path()
            except Exception as e:
                QMessageBox.critical(self, "File Error", f"Cannot access arc file:\n{str(e)}")
                return
            
            # Get the spectral data
            sciarc_ex = self.ARCdata[0]
            SLIT = self.slit.currentText()
            GRISM = self.grism.currentText()
            print(f"arc spec : {sciarc_ex} SLIT {SLIT} GRISM {GRISM}")
            # Create and show the interactive dialog
            dialog = InteractiveWavelengthDialog(
                self,
                sciarc_ex.spectral_axis.value,
                sciarc_ex.flux.value,
                full_path,
                slit=SLIT,
                grism=GRISM
            )
            
            if dialog.exec_() == QDialog.Accepted:
                # Get the results
                offset, dispersion = dialog.get_values()
                
                # Store and display the results
                self.offset = offset
                self.dispersion = dispersion
                self.offset_display.setText(f"{self.offset:.2f} Å")
                self.dispersion_display.setText(f"{self.dispersion:.4f} Å/pix")
                self.confirm_initial_btn.setEnabled(True)
                
                self.log_message(
                    f"New wavelength solution: Offset={self.offset:.2f}Å, "
                    f"Dispersion={self.dispersion:.4f}Å/pix"
                )
            else:
                self.log_message("Wavelength estimation cancelled by user")
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Calibration Error",
                f"Failed to estimate wavelength solution:\n{str(e)}"
            )
            self.log_message(f"Wavelength estimation failed: {str(e)}")

    def confirm_initial_solution(self):
        """Confirm initial wavelength solution"""
        self.identify_btn.setEnabled(True)
        self.log_message(f"Initial solution confirmed: offset={self.offset:.2f}, disp={self.dispersion:.2f}")

    def identify_lines(self):
        """First line identification"""
        try:
            sciarc_ex = self.ARCdata[0]
            arcfile = self.arc_file.text()
            
            # Calculate initial wavelengths
            sci_xp = sciarc_ex.spectral_axis.value
            self.wave1 = add.XFIT(sci_xp, self.offset, self.dispersion)
            print(f"func for line identification {self.line_func.currentText()}")
            # First identification
            self.wave2 = identify.fit_arclines(
                self.wave1,
                sciarc_ex.flux.value,
                arcfile,
                autotol=self.search_width.value(),
                pthreshold=self.threshold.value(),
                func=self.line_func.currentText(),deg=self.deg_val.value()
            )
            
            self.confirm_identify_btn.setEnabled(True)
            self.refine1_btn.setEnabled(True)
            self.log_message("Line identification completed")
            
        except Exception as e:
            self.log_message(f"Error in line identification: {str(e)}")

    def confirm_line_identification(self):
        """Confirm identified lines"""
        self.log_message("Line identification confirmed")
        # Could add plotting here to visualize identified lines

    def first_refinement(self):
        """First refinement step"""
        try:
            sciarc_ex = self.ARCdata[0]
            arcfile = self.arc_file.text()
            
            self.wave3 = identify.fit_arclines(
                self.wave2,
                sciarc_ex.flux.value,
                arcfile,
                self.refine_width.value(),
                self.refine_threshold.value(),func=self.line_func.currentText(),deg=self.deg_val.value()
            )
            
            self.refine2_btn.setEnabled(True)
            self.log_message("First refinement completed")
            
        except Exception as e:
            self.log_message(f"Error in first refinement: {str(e)}")

    def final_refinement(self):
        """Final refinement step"""
        try:
            sciarc_ex = self.ARCdata[0]
            arcfile = self.arc_file.text()
            
            self.final_wave = identify.fit_arclines(
                self.wave3,
                sciarc_ex.flux.value,
                arcfile,
                self.final_refine_width.value(),
                self.final_refine_threshold.value(),func=self.line_func.currentText(),deg=self.deg_val.value(),saveplot=True,
                plotoutdir=os.getcwd()
            )
            
            self.confirm_final_btn.setEnabled(True)
            self.log_message("Final refinement completed")
            
        except Exception as e:
            self.log_message(f"Error in final refinement: {str(e)}")

    def confirm_final_solution(self):
        """Confirm final wavelength solution"""
        self.log_message("Final wavelength solution confirmed")

    def perform_spectral_calibration(self):
        """Perform spectral calibration for all groups and save to FITS"""
        try:
            if not hasattr(self, 'final_wave'):
                raise ValueError("Please complete all refinement steps")
                
            # Initialize storage for calibrated spectra
            self.W_sci_groups = {}
            self.W_std_groups = {}
            
            # Process all science groups
            for group_id, sci_spec in self.combined_spectra['SCI'].items():
                calibrated = Spectrum1D(
                    spectral_axis=self.final_wave * u.AA,
                    flux=sci_spec.flux,
                    uncertainty=sci_spec.uncertainty
                )
                self.W_sci_groups[group_id] = calibrated
                self._save_calibrated_spectrum(calibrated, 'science', group_id)
            
            # Process all standard groups
            for group_id, std_spec in self.combined_spectra['STD'].items():
                calibrated = Spectrum1D(
                    spectral_axis=self.final_wave * u.AA,
                    flux=std_spec.flux,
                    uncertainty=std_spec.uncertainty
                )
                self.W_std_groups[group_id] = calibrated
                self._save_calibrated_spectrum(calibrated, 'standard', group_id)
            
            self.log_message("Spectral calibration completed and saved for all groups")
            
        except Exception as e:
            error_msg = f"Error in spectral calibration: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _calibrate_single_spectrum(self, spectrum):
        """Apply wavelength calibration to a single spectrum"""
        return Spectrum1D(
            spectral_axis=self.final_wave * u.AA,
            flux=spectrum.flux,
            uncertainty=spectrum.uncertainty
        )

    def _save_calibrated_spectrum00(self, spectrum, spec_type, group_id):
        """Save wavelength-calibrated spectrum to FITS file"""
        try:
            # Determine base filename
            if spec_type == 'science':
                base_name = self.sciname
            else:
                base_name = self.stdname
            
            filename = f"{base_name}_group{group_id}_wcal.fits"
            
            # Create primary HDU with flux
            primary_hdu = fits.PrimaryHDU(spectrum.flux.value)
            
            # Create HDU list
            hdus = [primary_hdu]
            
            # Add uncertainty if available
            if spectrum.uncertainty is not None:
                uncert_hdu = fits.ImageHDU(spectrum.uncertainty.array, name='UNCERTAINTY')
                hdus.append(uncert_hdu)
            
            # Add wavelength array
            wave_hdu = fits.ImageHDU(spectrum.spectral_axis.value, name='WAVELENGTH')
            print(f" the wavelength solution {wave_hdu}")
            hdus.append(wave_hdu)
            
            # Add header with calibration info
            hdus[0].header['WCALTYPE'] = 'Final'
            hdus[0].header['CRVAL1'] = spectrum.spectral_axis[0].value
            hdus[0].header['CDELT1'] = np.mean(np.diff(spectrum.spectral_axis.value))
            
            # Write to file
            hdulist = fits.HDUList(hdus)
            hdulist.writeto(filename, overwrite=True)
            self.log_message(f"Saved wavelength-calibrated spectrum: {filename}")
        
        except Exception as e:
            self.log_message(f"Error saving calibrated spectrum: {str(e)}")
            raise
    def _save_calibrated_spectrum(self, spectrum, spec_type, group_id):
        """Save wavelength-calibrated spectrum to FITS file"""
        try:
            # Determine base filename
            if spec_type == 'science':
                base_name = self.sciname
            else:
                base_name = self.stdname
            
            filename = f"{base_name}_group{group_id}_wcal.fits"
            headers = spectrum.meta.get('headers', [])
            # Create primary HDU with flux
            primary_hdu = fits.PrimaryHDU(spectrum.flux.value)
            if headers:
                primary_hdu.header = headers[0].copy()
            # Create HDU list
            hdus = [primary_hdu]
            
            # Add uncertainty if available
            if spectrum.uncertainty is not None:
                uncert_hdu = fits.ImageHDU(spectrum.uncertainty.array, name='UNCERTAINTY')
                hdus.append(uncert_hdu)
            
            # Add wavelength array
            wave_hdu = fits.ImageHDU(spectrum.spectral_axis.value, name='WAVELENGTH')
            print(f" the wavelength solution {wave_hdu}")
            hdus.append(wave_hdu)
            
            # Add header with calibration info
            hdus[0].header['WCALTYPE'] = 'Final'
            hdus[0].header['CRVAL1'] = spectrum.spectral_axis[0].value
            hdus[0].header['CDELT1'] = np.mean(np.diff(spectrum.spectral_axis.value))
            
            # Write to file
            hdulist = fits.HDUList(hdus)
            hdulist.writeto(filename, overwrite=True)
            self.log_message(f"Saved wavelength-calibrated spectrum: {filename}")
        
        except Exception as e:
            self.log_message(f"Error saving calibrated spectrum: {str(e)}")
            raise

    def select_directory(self):
        """Select working directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_path.setText(directory)

    def setup_flux_calibration_tab(self):
        """Tab for flux calibration with standard star search and preview"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Flux calibration group
        flux_group = QGroupBox("Flux Calibration")
        flux_layout = QVBoxLayout(flux_group)
        
        # ===== 1. Extinction File Selection =====
        ext_layout = QHBoxLayout()
        ext_layout.addWidget(QLabel("Extinction File:"))
        
        # Set default path to extinction file in resources
        default_ext = os.path.join(script_dir,"resources", "extinction", "DOT_extinction.dat")
        self.ext_file = QLineEdit(default_ext)
        
        ext_button = QPushButton("Browse...")
        ext_button.clicked.connect(self.select_extinction_file)
        ext_layout.addWidget(self.ext_file)
        ext_layout.addWidget(ext_button)
        flux_layout.addLayout(ext_layout)
        
        # ===== 2. Standard Star Selection =====
        std_layout = QHBoxLayout()
        std_layout.addWidget(QLabel("Standard Star:"))
        
        # Standard star name (default from self.stdname)
        #self.std_name_edit = QLineEdit(getattr(self, 'stdname', ""))
        self.std_name_edit = QLineEdit()
        self.std_name_edit.setPlaceholderText("name of the standard star listed in iraf catalogue")
        std_layout.addWidget(self.std_name_edit)
        
        # Search button for standard star file
        search_btn = QPushButton("Search Standard")
        search_btn.clicked.connect(self.search_standard_star)
        std_layout.addWidget(search_btn)
        
        # Display standard file path (read-only)
        self.std_file_display = QLineEdit()
        self.std_file_display.setReadOnly(True)
        std_layout.addWidget(self.std_file_display)
        
        flux_layout.addLayout(std_layout)
        
        # ===== 3. Preview Sensitivity Curve =====
        preview_group = QGroupBox("Sensitivity Curve Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Preview figure
        self.sens_figure = Figure()
        self.sens_canvas = FigureCanvas(self.sens_figure)
        preview_layout.addWidget(self.sens_canvas)
        
        # Preview button
        self.preview_btn = QPushButton("Preview Sensitivity Curve")
        self.preview_btn.clicked.connect(self.preview_sensitivity)
        self.preview_btn.setEnabled(False)  # Disabled until standard is selected
        preview_layout.addWidget(self.preview_btn)
        
        flux_layout.addWidget(preview_group)
        
        # ===== 4. Calibration Parameters =====
        param_group = QGroupBox("Calibration Parameters")
        param_layout = QVBoxLayout(param_group)
        
        # Sensitivity function mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Sensitivity Mode:"))
        self.sens_mode = QComboBox()
        self.sens_mode.addItems(["linear", "spline"])
        mode_layout.addWidget(self.sens_mode)
        param_layout.addLayout(mode_layout)
        
        # Display options
        display_layout = QHBoxLayout()
        display_layout.addWidget(QLabel("Display Sensitivity:"))
        self.display_sens_check = QCheckBox()
        self.display_sens_check.setChecked(True)
        display_layout.addWidget(self.display_sens_check)
        
        display_layout.addWidget(QLabel("Display Final Spectrum:"))
        self.display_final_check = QCheckBox()
        self.display_final_check.setChecked(True)
        display_layout.addWidget(self.display_final_check)
        param_layout.addLayout(display_layout)
        
        flux_layout.addWidget(param_group)
        
        # ===== 5. Perform Flux Calibration =====
        self.flux_cal_btn = QPushButton("Perform Flux Calibration")
        self.flux_cal_btn.clicked.connect(self.perform_flux_calibration)
        self.flux_cal_btn.setEnabled(False)  # Disabled until preview
        flux_layout.addWidget(self.flux_cal_btn)
        
        layout.addWidget(flux_group)
        self.tabs.addTab(tab, "Flux Calibration")

    def setup_output_tab(self):
        """Tab for final output"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Output group
        out_group = QGroupBox("Final Output")
        out_layout = QVBoxLayout(out_group)
        
        # Output name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Output Name:"))
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("Enter output name (without extension)")
        name_layout.addWidget(self.output_name)
        out_layout.addLayout(name_layout)

        '''
        redshift_layout = QHBoxLayout()
        redshift_layout.addWidget(QLabel("Redshift:"))
        self.redshift = QLineEdit()
        self.redshift.setValue(0.0)
        self.redshift.setPlaceholderText("Enter redshift")
        redshift_layout.addWidget(self.redshift)
        out_layout.addLayout(redshift_layout)
        '''
        redshift_layout = QHBoxLayout()
        redshift_layout.addWidget(QLabel("Redshift:"))
        self.redshift = QDoubleSpinBox()
        self.redshift.setDecimals(3)
        self.redshift.setRange(0.000, 2)
        self.redshift.setSingleStep(0.001)
        self.redshift.setValue(0.000)
        redshift_layout.addWidget(self.redshift)
        out_layout.addLayout(redshift_layout)
        
        # Display checkbox
        display_layout = QHBoxLayout()
        display_layout.addWidget(QLabel("Display Final Spectrum:"))
        self.display_final_check = QCheckBox()
        self.display_final_check.setChecked(True)
        display_layout.addWidget(self.display_final_check)
        out_layout.addLayout(display_layout)
        
        # Save final spectrum button
        save_button = QPushButton("Save Final Spectrum")
        save_button.clicked.connect(self.save_final_spectrum)
        out_layout.addWidget(save_button)

        # Save final spectrum plot button
        save_button = QPushButton("Save Final Spectrum PLot")
        save_button.clicked.connect(self.save_final_plot)
        #save_button.clicked.connect(self.plot_final_spectrum)
        out_layout.addWidget(save_button)        
        # Plot area
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        out_layout.addWidget(self.canvas)
        
        layout.addWidget(out_group)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Output")

    def select_extinction_file(self):
        """Select extinction file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Select Extinction File")
        if filename:
            self.ext_file.setText(filename)
    
    def select_std_file(self):
        """Select standard star data file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Select Standard Star Data File")
        if filename:
            self.std_data_file.setText(filename)

    def search_standard_star(self):
        """Search for standard star file in resources/onedstar"""
        try:
            star_name = self.std_name_edit.text().strip().lower()
            print(f" the star name in the search standard star function{star_name}")
            if not star_name:
                raise ValueError("Please enter a standard star name")
            
            # Search in onedstar directory
            onedstar_dir = os.path.join(script_dir,"resources", "onedstar")
            std_file = None
            std_folder = add.stdfile(star_name.lower() + ".dat")
            std_file = os.path.join(std_folder, star_name.lower() + ".dat")
           
            if std_file:
                self.std_file_display.setText(std_file)
                self.preview_btn.setEnabled(True)
                self.log_message(f"Found standard star file: {std_file}")
            else:
                raise FileNotFoundError(f"No standard star file found for '{star_name}'")
                
        except Exception as e:
            self.log_message(f"Error searching standard star: {str(e)}")
            QMessageBox.warning(self, "Search Failed", str(e))
    def preview_sensitivity(self):
        """Preview the sensitivity curve before full calibration"""
        try:
            std_file = self.std_file_display.text()
            if not std_file:
                raise ValueError("No standard star file selected")

            # Check if we have grouped data or single spectrum
            if hasattr(self, 'W_std_groups'):
                # Use the first group's standard spectrum if available
                if not self.W_std_groups:
                    raise ValueError("No wavelength-calibrated standard spectra available")
                group_id = next(iter(self.W_std_groups))
                std_spec = self.W_std_groups[group_id]
                self.log_message(f"Using standard spectrum from group {group_id} for preview")
            elif hasattr(self, 'W_std'):
                # Fall back to single spectrum if no groups exist
                std_spec = self.W_std
                self.log_message("Using single standard spectrum for preview")
            else:
                raise ValueError("No wavelength-calibrated standard spectrum available")

            # Read standard star reference data
            star = fluxcal.stdstr(std_file)
            
            # Get extinction data
            Xfile = fluxcal.read_obs_extinction(self.ext_file.text())
            
            # Apply airmass correction
            if not self.std_lst:
                raise ValueError("No standard star observation files available")
            
            std_ccd = ccdproc.CCDData.read(self.std_lst[-1], unit="adu")
            cal_am = airmass.Airmass(std_ccd, self.std_name.text())
            CAL = fluxcal.airmass_cor(std_spec, cal_am, Xfile)

            # Create preview sensitivity function
            sens_func = fluxcal.standard_sensfunc(
                CAL,
                star,
                mode=self.sens_mode.currentText(),
                display=False  # We'll display our own preview
            )

            # Plot sensitivity curve
            self.sens_figure.clear()
            ax = self.sens_figure.add_subplot(111)
            
            ax.plot(sens_func.spectral_axis, sens_func.flux)
            ax.set_xlabel('Wavelength (Å)')
            ax.set_ylabel('Sensitivity')
            ax.set_title('Sensitivity Function Preview')
            
            # Add wavelength markers for important features
            try:
                peaks, _ = find_peaks(sens_func.flux.value, height=np.nanmedian(sens_func.flux.value)*1.5)
                for peak in peaks:
                    ax.axvline(x=sens_func.spectral_axis[peak].value, 
                              color='r', linestyle=':', alpha=0.3)
            except Exception:
                pass  # Skip if peak finding fails
                
            self.sens_canvas.draw()
            self.flux_cal_btn.setEnabled(True)
            self.log_message("Sensitivity curve preview generated")

        except Exception as e:
            self.log_message(f"Error in sensitivity preview: {str(e)}")
            QMessageBox.warning(self, "Preview Failed", str(e))

    def perform_flux_calibration(self):
        """Perform flux calibration using the saved wavelength-calibrated files"""
        try:
            # Process each science group
            for group_id in self.W_sci_groups:
                sci_file = f"{self.sciname}_group{group_id}_wcal.fits"
                std_file = f"{self.stdname}_group{group_id}_wcal.fits"
                
                if not os.path.exists(sci_file) or not os.path.exists(std_file):
                    self.log_message(f"Skipping group {group_id} - missing files")
                    continue
                    
                # Load the spectra
                sci_spec = fluxcal.load_calibrated_spectrum(sci_file)
                std_spec = fluxcal.load_calibrated_spectrum(std_file)
                
                # Perform flux calibration
                self.flux_calibrated = self._calibrate_flux(sci_spec, std_spec)
                
                # Save result
                output_name = f"{self.sciname}_group{group_id}_fluxcal.fits"
                self._save_flux_calibrated(self.flux_calibrated, output_name)
                
            self.log_message("Flux calibration completed for all groups")
            
        except Exception as e:
            error_msg = f"Error in flux calibration: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _calibrate_flux(self, sci_spec, std_spec):
        """
        Thin GUI wrapper: pull parameters from the widgets, delegate the
        actual calibration to specred.calib.flux.calibrate_flux (moved
        there so it's usable/testable without the GUI), and translate
        errors into log messages.
        """
        try:
            sci_am = airmass.Airmass(self.sci_lst[-1], self.sciname) if self.sci_lst else 1.0
            std_am = airmass.Airmass(self.std_lst[-1], self.stdname) if self.std_lst else 1.0

            return fluxcal.calibrate_flux(
                sci_spec, std_spec,
                extinction_file=self.ext_file.text(),
                sci_airmass=sci_am,
                std_airmass=std_am,
                std_file=self.std_file_display.text(),
                sens_mode=self.sens_mode.currentText(),
                display=self.display_sens_check.isChecked(),
            )
        except Exception as e:
            self.log_message(f"Error in flux calibration: {str(e)}")
            raise

    def _save_flux_calibrated(self, spectrum, filename):
        """Thin GUI wrapper around specred.calib.flux.save_flux_calibrated."""
        try:
            fluxcal.save_flux_calibrated(
                spectrum, filename,
                calsrc=self.std_file_display.text(),
                calmode=self.sens_mode.currentText(),
            )
            self.log_message(f"Saved flux-calibrated spectrum: {filename}")
        except Exception as e:
            self.log_message(f"Error saving flux-calibrated spectrum: {str(e)}")
            raise

    def save_final_plot(self):
        """ save plot of the final spectrum"""
        try:
            if not hasattr(self,'final_spec'):
                raise ValueError("no final spectrum available")

            output_name = self.output_name.text()
            Redshift = self.redshift.value()
            print(f" the redshift {Redshift}")
            if not output_name:
                output_name = self.sciname
            #plot_final_spectrum()
            final.save_plot(self.flux_calibrated,name=output_name,redshift=Redshift)

            self.log_message(f" save the plot of final spectrum")

        except Exception as e:
            error_msg = f"Error saving spectrum plot: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to save spectrum plot: {str(e)}")
            
    

    def save_final_spectrum(self):
        """Save the final calibrated spectrum"""
        try:
            if not hasattr(self, 'final_spec'):
                raise ValueError("No final spectrum to save")
            
            output_name = self.output_name.text()
            if not output_name:
                output_name = self.sciname
            
            # Save the spectrum
            print(f" the final spec for save_spectrum {self.final_spec}")
            print(f" the final spec  {self.flux_calibrated}")

            final.save_spectrum(self.flux_calibrated, output_name+"_MJD"+str(int(self.sci_mjd)))
            
            self.log_message(f"Final spectrum saved as {output_name}.csv")
            QMessageBox.information(self, "Success", f"Spectrum saved as {output_name}.csv")
            
        except Exception as e:
            error_msg = f"Error saving spectrum: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to save spectrum: {str(e)}")


    def plot_final_spectrum(self):
        """Plot the final calibrated spectrum"""
        # NOTE (bug fixes applied here):
        #   - guarded on 'final_spec' but used self.flux_calibrated -> guard
        #     now checks the attribute actually used.
        #   - `plt` was never imported (only matplotlib.pylab as pllt) -> the
        #     stray plt.show() call is removed; this widget draws into the
        #     embedded self.sens_canvas, it doesn't need a separate window.
        #   - `box_kernal` typo (undefined name) -> box_kernel.
        #   - convolve() on the uncertainty array was missing the kernel arg.
        #   - self.redshift is the QDoubleSpinBox widget, not a number ->
        #     use self.redshift.value().
        if hasattr(self, 'flux_calibrated'):
            redshift = self.redshift.value()
            self.sens_figure.clear()
            ax = self.sens_figure.add_subplot(111)

            ax.errorbar(self.flux_calibrated.spectral_axis.value/(1+redshift), self.flux_calibrated.flux.value,self.flux_calibrated.uncertainty.array,ecolor="g")
            ## smoothed curve
            box_kernel = Box1DKernel(5)
            newflx = convolve(self.flux_calibrated.flux.value,box_kernel)
            newflxerr = convolve(self.flux_calibrated.uncertainty.array,box_kernel)
            ax.errorbar(self.flux_calibrated.spectral_axis.value/(1+redshift),newflx,newflxerr,ecolor="gray")
            ax.axvline(6563,color="brown",alpha=0.7,ls="dashed",label="H alpha")
            ax.axvline(5007,color="brown",alpha=0.7,ls="dashed",label="OIII")
            ax.axvline(4861,color="brown",alpha=0.7,ls="dashed",label="H beta")
            ax.set_xlabel('Wavelength (Å)')
            ax.set_ylabel('Flux')
            ax.set_title('Final Calibrated Spectrum')
            self.sens_canvas.draw()

#******************************************************************************************************#

def main():
    """Entry point for the `specred-gui` console script."""
    app = QApplication(sys.argv)
    gui = SpectroscopicReductionGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
