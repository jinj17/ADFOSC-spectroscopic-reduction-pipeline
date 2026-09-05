"""
Interactive wavelength-calibration dialog.

Extracted from the original monolithic gui_spec_yaml.py so it can be
imported/tested independently of the main window, and so the
slit/grism -> wavelength-solution-guess table lives in an instrument
config file (specred/instruments/<name>.yaml) instead of being
hardcoded in the UI code.
"""
import numpy as np
from astropy.table import Table

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Button

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton

from specred.instruments import load_instrument, get_wavelength_guess

#=============================================================================================================================#

class InteractiveWavelengthDialog(QDialog):
    def __init__(self, parent=None, xpt=None, ypt=None, arc_file=None, slit=None, grism=None,
                 instrument="adfosc"):
        super().__init__(parent)
        self.setWindowTitle("Wavelength Calibration")
        self.setModal(True)
        
        # Store the final values
        self.final_offset = None
        self.final_dispersion = None
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(17, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Setup UI
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        
        # Add close button
        close_btn = QPushButton("Apply and Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
        # Initialize plot
        self.init_plot(xpt, ypt, arc_file, slit, grism, instrument)
    
    def init_plot(self, xpt, ypt, arc_file, slit, grism, instrument="adfosc"):
        """Initialize the interactive plot"""
        # Your plotting code here
        xpt = xpt - xpt[len(xpt) // 2]
        self.xpt = xpt  # Store for updates
        ARC = Table.read(arc_file, format="ascii", names=("wave", "flux"))
        wave = ARC["wave"].value[::-1]
        flux = ARC["flux"].value[::-1]
        
        wbin = np.where((wave > 3500) & (wave < 9000))
        wave = wave[wbin]
        flux = flux[wbin]
        
        ypt = ypt / np.nanmax(ypt)
        SLIT = slit
        GRISM = grism

        # NOTE: this lookup table used to be hardcoded here. It now lives
        # in specred/instruments/<instrument>.yaml so a different
        # instrument/telescope can be supported without touching GUI code.
        instrument_cfg = load_instrument(instrument)
        dispersion, offset = get_wavelength_guess(instrument_cfg, SLIT, GRISM)

        if dispersion is None or offset is None:
            raise ValueError(
                f"No wavelength-solution guess defined for slit={SLIT!r}, "
                f"grism={GRISM!r} in instruments/{instrument}.yaml. "
                f"Add an entry there for this slit/grism combination."
            )
        Xpt = offset + xpt * dispersion
        
        self.ax.set_ylim(0, None)
        #self.ax.plot(wave, flux, color="r", label="Reference Spectrum")
        #add.zipplot(wave,flux,self.ax,color="r",label="Reference Spectrum")
        for i, (x, y) in enumerate(zip(wave,flux)):
            # Draw vertical line
            self.ax.axvline(x=x, ymax=y, color="r", label="reference spectrum" if i == 0 else "")
            # Place text slightly above the line
            self.ax.text(x=x, y=y + 0.01, s="{:.2f}".format(x), rotation=90, color="g")
        self.lines = self.ax.plot(Xpt, ypt, label="Observed spectrum")[0]
        self.ax.legend()
        
        self.figure.subplots_adjust(left=0.25, bottom=0.25)
        
        # Create sliders
        axoffset = self.figure.add_axes([0.25, 0.1, 0.65, 0.03])
        self.offset_sdr = Slider(ax=axoffset, label="offset", valmin=1000, valmax=10000, valinit=offset)
        
        axdisp_A = self.figure.add_axes([0.1, 0.25, 0.03, 0.65])
        self.A_disp_sdr = Slider(ax=axdisp_A, label="dispersion", valmin=-10, valmax=10, valinit=dispersion,
                                orientation="vertical")
        
        axdisp_B = self.figure.add_axes([0.05, 0.25, 0.03, 0.65])
        self.B_disp_sdr = Slider(ax=axdisp_B, label="dispersion", valmin=-1, valmax=1, valinit=0.0,
                                orientation="vertical")

        ## Adding buttons for incrementing offset and dispersion
        increment_offset_button_ax = self.figure.add_axes([0.9, 0.01, 0.045, 0.03])
        self.increment_offset_button = Button(increment_offset_button_ax, 'Offset +', hovercolor='0.975')

        decrement_offset_button_ax = self.figure.add_axes([0.9, 0.04, 0.045, 0.03])
        self.decrement_offset_button = Button(decrement_offset_button_ax, 'Offset -', hovercolor='0.975')
        
        increment_disp_button_ax = self.figure.add_axes([0.03, 0.1, 0.045, 0.03])
        self.increment_disp_button = Button(increment_disp_button_ax, 'disp +', hovercolor='0.975')

        decrement_disp_button_ax = self.figure.add_axes([0.075, 0.1, 0.045, 0.03])
        self.decrement_disp_button = Button(decrement_disp_button_ax, 'Disp -', hovercolor='0.975')
        
        # Create buttons
        button_ax = self.figure.add_axes([0.8, 0.15, 0.1, 0.05])
        self.apply_button = Button(button_ax, 'Apply Values')
        
        # Connect events
        self.offset_sdr.on_changed(self.update_plot)
        self.A_disp_sdr.on_changed(self.update_plot)
        self.B_disp_sdr.on_changed(self.update_plot)
        self.apply_button.on_clicked(self.on_apply)

        self.increment_offset_button.on_clicked(self.increment_offset)
        self.increment_disp_button.on_clicked(self.increment_disp)
        self.decrement_offset_button.on_clicked(self.decrement_offset)
        self.decrement_disp_button.on_clicked(self.decrement_disp)
        
        self.canvas.draw()
    
    def update_plot(self, val):
        """Update the plot when sliders change"""
        new_xpt = self.offset_sdr.val + self.xpt * (self.A_disp_sdr.val + self.B_disp_sdr.val)
        self.lines.set_xdata(new_xpt)
        self.canvas.draw_idle()
    
    def on_apply(self, event):
        """Store final values when Apply is clicked"""
        self.final_offset = self.offset_sdr.val
        self.final_dispersion = self.A_disp_sdr.val + self.B_disp_sdr.val
        self.accept()
    
    def get_values(self):
        """Return the final values"""
        return self.final_offset, self.final_dispersion
    def increment_offset(self,event):
        current_offset = self.offset_sdr.val
        self.offset_sdr.set_val(current_offset + 2)  # increment by 100 units
    def decrement_offset(self,event):
        current_offset = self.offset_sdr.val
        self.offset_sdr.set_val(current_offset - 3)  # increment by 100 units

    def increment_disp(self,event):
        current_disp = self.A_disp_sdr.val + self.B_disp_sdr.val
        self.A_disp_sdr.set_val(self.A_disp_sdr.val + 0.005)  # increment A_disp by 0.1 units
    def decrement_disp(self,event):
        current_disp = self.A_disp_sdr.val + self.B_disp_sdr.val
        self.A_disp_sdr.set_val(self.A_disp_sdr.val - 0.005) 

   

    
def XFIT(X,offset,disp):
    new_X = offset +(X-X[len(X)//2])*disp
    return new_X

#=============================================================================================================================#

