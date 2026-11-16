from pathlib import Path
import sys
import os
import numpy as np
from astropy.io import fits
import ccdproc
from specutils import Spectrum1D
from astropy import units as u
from astropy.nddata import StdDevUncertainty
import matplotlib
import matplotlib.pyplot as pllt
import yaml
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks
from collections import defaultdict
from astropy.convolution import convolve, Box1DKernel

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QLineEdit, QComboBox,
    QFileDialog, QTabWidget, QTextEdit, QCheckBox, QSpinBox,
    QDoubleSpinBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QStyledItemDelegate,QItemEditorFactory, QItemEditorCreatorBase, QScrollArea, QFrame, QSplitter,QColorDialog, QActionGroup,
    QToolButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer,QSettings
from PyQt5.QtGui import QIntValidator
import pyqtgraph as pg

script_dir = os.path.dirname(os.path.abspath(__file__))

# Import reduction functions
import specred
script_dir = os.path.dirname(os.path.abspath(specred.__file__))

scpath = script_dir
print(f" the script path : {scpath}")
if not os.path.isdir(scpath):
    raise FileNotFoundError(f"Required pipeline folder not found: {scpath}")
sys.path.insert(0, scpath)
from specred import utils as add
from specred.calib import flux as fluxcal
from specred.calib import airmass
from specred.calib import wavelength as identify
from specred.calib import flat as flat_2
from specred import io as readfile_v2
from specred import output as final
from specred.extraction import aperture as aperture
from specred.instruments import load_instrument, get_trim_box


from astropy.table import Table

# Global spatial axis variable
sxs = 0
SXS = 0
from PyQt5.QtWidgets import QDialog

#=============================================================================================================================#

class _ConsoleStream(QObject):
    """Redirect Python print messages to the on-screen log safely via signals."""
    message = pyqtSignal(str)

    def __init__(self, original):
        super().__init__()
        self.original = original

    def write(self, value):
        if self.original:
            self.original.write(value)
        if value:
            self.message.emit(str(value))

    def flush(self):
        if self.original:
            self.original.flush()


class InteractiveWavelengthDialog(QDialog):
    def __init__(self, parent=None, xpt=None, ypt=None, arc_file=None,slit=None,grism=None):
        super().__init__(parent)
        self.setWindowTitle("Wavelength Calibration")
        self.setModal(True)
        
        # Store the final values
        self.final_offset = None
        self.final_dispersion = None
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(18, 8))
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
        self.init_plot(xpt, ypt, arc_file,slit,grism)
    
    def init_plot(self, xpt, ypt, arc_file,slit,grism):
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
        SLIT = slit #self.slit.currentText()
        GRISM = grism #self.grism.currentText()
        config = {
            ('1"', '676R-420gr/mm'):   (-3.38, 6300),
            ('1"', '132R-600gr/mm'):   (-3.00, 6326),
            ('1"', '770R-300gr/mm'):   (-7.00, 6400),
            ('1.5"', '676R-420gr/mm'): (-3.355, 6305),
            ('1.5"', '132R-600gr/mm'): (-2.27, 5800),
            ('1.5"', '770R-300gr/mm'): (-7.00, 6500),
            ('2"', '676R-420gr/mm'): (-3.35, 6326),
            ('2"', '132R-600gr/mm'): (-2.27, 5800),
            ('2"', '770R-300gr/mm'): (-7.00, 6500),
        }

        # Get the corresponding dispersion and offset values
        dispersion, offset = config.get((SLIT, GRISM), (None, None))

        if dispersion is None or offset is None:
            raise ValueError(f"Unknown SLIT-GRISM combination: {SLIT}, {GRISM}")
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

class SpectroscopicReductionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectroscopic Reduction Pipeline")
        self.resize(1750, 900)
        self.setMinimumSize(1000, 650)
        
        # Initialize variables
        self.init_variables()
        self.info_details = self._load_info_details()
        self.settings = QSettings('ADFOSC', 'SpectroscopicReduction')
        self.theme_name = str(self.settings.value('theme', 'Dark'))
        if self.theme_name not in self.THEMES:
            self.theme_name = 'Dark'
        #self.custom_accent = str(self.settings.value('accent', '') or '')
        #self.custom_plot_bg = str(self.settings.value('plot_bg', '') or '')
        # Setup UI
        self.setup_ui()
        self.apply_theme()
        self.setup_menu()

    def _load_info_details(self):
        path = Path(script_dir) / 'gui_info.yaml'
        try:
            with path.open(encoding='utf-8') as stream:
                data = yaml.safe_load(stream) or {}
            return data if isinstance(data, dict) else {}
        except (OSError, yaml.YAMLError) as exc:
            print(f'Could not load GUI help: {exc}')
            return {}

    def create_info_label(self, key, text, checkbox=False):
        """Wrap a label or existing checkbox with optional YAML-backed help."""
        entry = self.info_details.get(key, {})
        label = text if checkbox else QLabel(text)
        if not isinstance(entry, dict) or not entry.get('description'):
            return label
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        button = QToolButton(container)
        button.setText('ⓘ')
        button.setToolTip(str(entry['description']))
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedSize(15, 15)
        button.setStyleSheet('QToolButton {color:#60A5FA;background:transparent;border:none;font-size:14px;} QToolButton:hover {color:#93C5FD;}')
        button.clicked.connect(lambda: QMessageBox.information(container, str(entry.get('title', key)), str(entry['description'])))
        layout.addWidget(label)
        layout.addWidget(button)
        #layout.addStretch()
        container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        return container

    def _add_table_help(self, table, keys):
        """Clickable information icons in table headers, with hover tooltips."""
        for col, key in keys.items():
            entry = self.info_details.get(key, {})
            item = table.horizontalHeaderItem(col)
            if item is not None and isinstance(entry, dict) and entry.get('description'):
                item.setText(item.text()+' ⓘ')
                item.setToolTip(str(entry['description']))
        def show_info(col):
            entry = self.info_details.get(keys.get(col), {})
            if isinstance(entry, dict) and entry.get('description'):
                QMessageBox.information(self, str(entry.get('title', keys[col])), str(entry['description']))
        table.horizontalHeader().setSectionsClickable(True)
        table.horizontalHeader().sectionClicked.connect(show_info)

        
    # Keep colours centrally defined; presets work with Python 3.8 and Qt5.
    THEMES = {
        'Dark': dict(bg='#111923', surface='#141f2b', input='#172633', text='#e6edf5', muted='#98b4ca', border='#34485c', accent='#176c85', hover='#247e99', plot='#0d1823', plottext='#dce7ef'),
        #'Midnight Blue': dict(bg='#0b1220', surface='#14213a', input='#1c2d4b', text='#e6efff', muted='#9fb3d1', border='#344b70', accent='#3b82f6', hover='#2563eb', plot='#080f1d', plottext='#cfddf6'),
        #'Nord': dict(bg='#2e3440', surface='#3b4252', input='#434c5e', text='#eceff4', muted='#d8dee9', border='#566477', accent='#88c0d0', hover='#5e81ac', plot='#242933', plottext='#eceff4'),
        'Dracula': dict(bg='#282a36', surface='#343746', input='#44475a', text='#f8f8f2', muted='#bdc0d1', border='#62708b', accent='#bd93f9', hover='#9966db', plot='#1f202a', plottext='#f8f8f2'),
        'Catppuccin Mocha': dict(bg='#1e1e2e', surface='#313244', input='#45475a', text='#cdd6f4', muted='#a6adc8', border='#585b70', accent='#cba6f7', hover='#b4befe', plot='#181825', plottext='#cdd6f4'),
        #'Monokai': dict(bg='#272822', surface='#33342a', input='#414339', text='#f8f8f2', muted='#b7b9ac', border='#616354', accent='#a6e22e', hover='#82b31d', plot='#20211c', plottext='#f8f8f2'),
        'Graphite': dict(bg='#17191d', surface='#24272d', input='#30343c', text='#eef0f3', muted='#a8aeb8', border='#464c55', accent='#60a5fa', hover='#3b82f6', plot='#111317', plottext='#e6edf5'),
        #'Forest': dict(bg='#102019', surface='#1b3125', input='#264232', text='#e1f4e7', muted='#a4c5af', border='#3d644b', accent='#33b679', hover='#218c5b', plot='#0d1812', plottext='#d4f2dc'),
        'Solar Orange': dict(bg='#171512', surface='#29231c', input='#373025', text='#f8ead8', muted='#c6ae95', border='#594537', accent='#d5360c', hover='#ec6035', plot='#11100f', plottext='#f3e2c6'),
        #'Rosewood': dict(bg='#21151c', surface='#35222e', input='#492e3d', text='#fbe9f0', muted='#d6b0c0', border='#70485e', accent='#e879a5', hover='#cb517f', plot='#190e15', plottext='#f9dce8'),
        #'Solarized Dark': dict(bg='#002b36', surface='#073642', input='#104653', text='#eee8d5', muted='#93a1a1', border='#456773', accent='#268bd2', hover='#1b74ae', plot='#00212b', plottext='#93a1a1'),
        'Slate': dict(bg='#161b24', surface='#202735', input='#273143', text='#e2e8f0', muted='#a5b4c7', border='#3d4b60', accent='#7c3aed', hover='#9061f9', plot='#10151e', plottext='#d8deea'),
        'Light': dict(bg='#f3f6fa', surface='#ffffff', input='#ffffff', text='#1e293b', muted='#64748b', border='#cbd5e1', accent='#2563eb', hover='#1d4ed8', plot='#ffffff', plottext='#334155'),
        'Arctic': dict(bg='#edf5f9', surface='#ffffff', input='#f6fbfd', text='#173247', muted='#617f91', border='#bfdae5', accent='#0e7490', hover='#155e75', plot='#ffffff', plottext='#173247'),
        'Ivory': dict(bg='#f7f3e9', surface='#fffdf8', input='#fffaf0', text='#342b27', muted='#78695c', border='#ded2c1', accent='#b45309', hover='#92400e', plot='#fffefb', plottext='#46392f'),
        #'Sakura': dict(bg='#fff4f7', surface='#ffffff', input='#fff9fb', text='#4b2533', muted='#8f6476', border='#edcad8', accent='#be185d', hover='#9d174d', plot='#ffffff', plottext='#60374a'),
        #'Solarized Light': dict(bg='#fdf6e3', surface='#eee8d5', input='#fffaf0', text='#586e75', muted='#657b83', border='#c9c2ae', accent='#268bd2', hover='#1b74ae', plot='#fdf6e3', plottext='#586e75')
    }
    LIGHT_THEMES = ('Light', 'Arctic', 'Ivory')
    
    def setup_menu(self):
        theme_menu = self.menuBar().addMenu('Theme')
        group = QActionGroup(self)
        group.setExclusive(True)
        self.theme_actions = {}
        theme_menu.addSection('Dark themes')
        for name in self.THEMES:
            if name in self.LIGHT_THEMES:
                continue
            action = theme_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == self.theme_name)
            action.triggered.connect(lambda checked, key=name: self.set_theme(key))
            group.addAction(action)
            self.theme_actions[name] = action
        theme_menu.addSeparator()
        theme_menu.addSection('Light themes')
        for name in self.LIGHT_THEMES:
            action = theme_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == self.theme_name)
            action.triggered.connect(lambda checked, key=name: self.set_theme(key))
            group.addAction(action)
            self.theme_actions[name] = action
        theme_menu.addSeparator()
        #theme_menu.addAction('Choose accent colour…', self.choose_accent_color)
        #theme_menu.addAction('Choose plot background…', self.choose_plot_background)
        #theme_menu.addAction('Reset custom colours', self.reset_custom_colors)
        menu = self.menuBar().addMenu('Help')
        menu.addAction('About the pipeline', self.show_about)
        menu.addAction('Open YAML configuration…', self.select_yaml_file)

    def show_about(self):
        QMessageBox.about(self, 'About ADFOSC Spectroscopic Reduction',
            '<h3>ADFOSC Spectroscopic Reduction Pipeline</h3>'
            '<p>GUI wrapper for bias/flat reduction, spectral extraction, '
            'arc wavelength calibration, combination, and flux calibration.</p>'
            '<p>Built with Python, PyQt5 and Matplotlib. '
            'The reduction algorithms are provided by the local SCRIPTS directory.</p>')

    def set_theme(self, name):
        if name not in self.THEMES:
            return
        self.theme_name = name
        #self.custom_accent = ''
        #self.custom_plot_bg = ''
        self.settings.setValue('theme', name)
        #self.settings.remove('accent')
        #self.settings.remove('plot_bg')
        self.apply_theme()
        for key, action in self.theme_actions.items():
            action.setChecked(key == name)
            
    """
    def apply_theme(self):
        self.setStyleSheet('''
            QMainWindow, QWidget {background: #111923; color: #e6edf5; font-size: 12px;}
            QTabWidget::pane {border: 1px solid #314154; border-radius: 5px; background: #141f2b;}
            QTabBar::tab {background: #223144; color: #cfdeec; padding: 10px 14px; margin-right: 2px;}
            QTabBar::tab:selected {background: #176c85; color: white; font-weight: 600;}
            QTabBar::tab:hover {background: #344b62;}
            QGroupBox {border: 1px solid #34485c; border-radius: 7px; margin-top: 16px; padding: 12px; font-weight: 600;}
            QGroupBox::title {subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #8cdaea;}
            QLabel#appHeading {font-size: 19px; font-weight: 700; color: #f2f8fd;}
            QLabel#subtitle {color: #98b4ca; font-size: 11px;}
            QPushButton {background: #264154; border: 1px solid #426179; padding: 7px 12px; border-radius: 4px;}
            QPushButton:hover {background: #1c7190;}
            QPushButton:pressed {background: #12536d;}
            QPushButton:disabled {background: #202b36; color: #6b8193;}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {background: #172633; color: #edf5fa; border: 1px solid #41576c; border-radius: 4px; padding: 5px;}
            QTableWidget, QTextEdit {background: #0d1823; color: #dce7ef; alternate-background-color: #162536; gridline-color: #314457;}
            QHeaderView::section {background: #253b4e; color: #e1edf6; padding: 6px; border: none; border-right: 1px solid #43566a;}
            QScrollArea {border: none; background: #111923;}
            QStatusBar {background: #1c2d3d; color: #bce4ec;}
        ''')
    """
    def apply_theme(self):
        c = dict(self.THEMES[self.theme_name])
        self.colors = c
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{background-color: {c['bg']}; color: {c['text']}; font-size: 12px;}}
            QMenuBar, QMenu {{background-color: {c['surface']}; color: {c['text']};}}
            QMenu::item:selected {{background: {c['accent']}; color: white;}}
            QTabWidget::pane {{border: 1px solid {c['border']}; border-radius: 5px; background: {c['surface']};}}
            QTabBar::tab {{background: {c['surface']}; color: {c['text']}; padding: 11px 8px; min-width: 115px;}}
            QTabBar::tab:selected {{background: {c['accent']}; color: white; font-weight: 600;}}
            QTabBar::tab:hover:!selected {{background: {c['hover']}; color: white;}}
            QGroupBox {{border: 1px solid {c['border']}; border-radius: 7px; margin-top: 16px; padding: 12px; font-weight: 600;}}
            QGroupBox::title {{subcontrol-origin: margin; left: 12px; padding: 0 5px; color: {c['accent']};}}
            QLabel#appHeading {{font-size: 19px; font-weight: 700; color: {c['text']};}}
            QLabel#subtitle {{color: {c['muted']}; font-size: 11px;}}
            QPushButton {{background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 7px 12px; border-radius: 4px;}}
            QPushButton:hover {{background: {c['accent']}; color: white;}}
            QPushButton:pressed {{background: {c['hover']}; color: white;}}
            QPushButton:disabled {{background: {c['surface']}; color: {c['muted']};}}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{background: {c['input']}; color: {c['text']}; border: 1px solid {c['border']}; border-radius: 4px; padding: 5px;}}
            QTableWidget, QTextEdit {{background: {c['plot']}; color: {c['text']}; alternate-background-color: {c['surface']}; gridline-color: {c['border']};}}
            QHeaderView::section {{background: {c['surface']}; color: {c['text']}; padding: 6px; border: none; border-right: 1px solid {c['border']};}}
            QScrollArea {{border: none; background: {c['bg']};}}
            QStatusBar {{background: {c['surface']}; color: {c['muted']};}}
            QSplitter::handle {{background: {c['border']};}}
            QCheckBox::indicator {{border: 1px solid {c['border']};border-radius: 4px;}}
            QCheckBox::indicator:checked {{background-color: {c['accent']};border: 1px solid {c['accent']};}}
        """)
        if hasattr(self, 'viewer_plot'):
            self.viewer_plot.setBackground(c['plot'])
            for axis in ('left', 'bottom', 'right', 'top'):
                ax = self.viewer_plot.getAxis(axis)
                ax.setPen(pg.mkPen(c['plottext']))
                ax.setTextPen(pg.mkPen(c['plottext']))
            self.viewer_plot.setLabel('bottom', self.viewer_plot.getAxis('bottom').labelText or 'Dispersion pixel', color=c['plottext'])
            self.viewer_plot.setLabel('left', self.viewer_plot.getAxis('left').labelText or 'Spatial pixel / Flux', color=c['plottext'])
        if hasattr(self, 'tabs'):
            self.tabs.style().unpolish(self.tabs)
            self.tabs.style().polish(self.tabs)
            
    def init_variables(self):
        """Initialize all class variables"""
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
        self.combined_spectra = {'SCI': {}, 'STD': {}}
        
        # Calibrated spectra
        self.W_sci = None
        self.W_std = None
        self.final_spec = None
        self.flux_calibrated = None
        self.final_groups = {}
        self.W_sci_groups, self.W_std_groups = {}, {}
        
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
        """Fixed three-part workstation: left controls, right viewer and log."""
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(6, 6, 6, 5)
        self.main_layout.setSpacing(6)

        header = QLabel('ADFOSC  |  SPECTROSCOPIC REDUCTION')
        header.setObjectName('appHeading')
        self.main_layout.addWidget(header)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_layout.addWidget(self.main_splitter, 1)

        # LEFT: all task tabs and their input fields, scrollable individually.
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMinimumWidth(360)
        self.setup_file_selection_tab()
        self.setup_flat_correction_tab()
        self.setup_aperture_extraction_tab()
        self.setup_spectral_calibration_tab()
        self.setup_flux_calibration_tab()
        self.setup_output_tab()
        # Wrap each existing page safely: remove it from QTabWidget before
        # reparenting it into QScrollArea, preserving its original tab state.
        original_tabs = []
        while self.tabs.count():
            page = self.tabs.widget(0)
            name = self.tabs.tabText(0)
            enabled = self.tabs.isTabEnabled(0)
            self.tabs.removeTab(0)
            original_tabs.append((page, name, enabled))
        for page, name, enabled in original_tabs:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            index = self.tabs.addTab(scroll, name)
            self.tabs.setTabEnabled(index, enabled)
        # Vertical task navigation keeps every stage visible in the narrow left panel.
        #self.tabs.setTabPosition(QTabWidget.West)
        #self.tabs.setUsesScrollButtons(False)
        self.tabs.setStyleSheet('QTabBar::tab {min-width: 120px; padding: 7.5px 2px;}')
        self.main_splitter.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self._tab_changed)

        # RIGHT: interactive viewer above an always-visible console.
        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.setChildrenCollapsible(False)
        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(6, 3, 6, 3)
        viewer_header = QHBoxLayout()
        viewer_header.addWidget(QLabel('INTERACTIVE VIEWER'))
        self.viewer_title = QLabel('Select a file and click Preview')
        self.viewer_title.setObjectName('subtitle')
        viewer_header.addWidget(self.viewer_title, 1)
        self.fit_view_btn = QPushButton('Fit view')
        self.view_previous_btn = QPushButton('◀ Previous')
        self.view_next_btn = QPushButton('Next ▶')
        self.view_counter = QLabel('')
        viewer_header.addWidget(self.view_previous_btn)
        viewer_header.addWidget(self.view_counter)
        viewer_header.addWidget(self.view_next_btn)
        viewer_header.addWidget(self.fit_view_btn)
        viewer_layout.addLayout(viewer_header)
        self.viewer_plot = pg.PlotWidget()
        self.viewer_plot.setBackground('#0d1823')
        self.viewer_plot.setLabel('bottom', 'Dispersion pixel', color='#dce7ef')
        self.viewer_plot.setLabel('left', 'Spatial pixel / Flux', color='#dce7ef')
        self.viewer_plot.showGrid(x=True, y=True, alpha=0.13)
        self.viewer_plot.getViewBox().setAspectLocked(False)
        viewer_layout.addWidget(self.viewer_plot, 1)
        self.viewer_readout = QLabel('Move the pointer over the plot to see coordinates')
        viewer_layout.addWidget(self.viewer_readout)
        self._aperture_views = []
        self._aperture_view_index = -1
        self.view_previous_btn.clicked.connect(lambda: self._navigate_aperture(-1))
        self.view_next_btn.clicked.connect(lambda: self._navigate_aperture(1))
        self._update_view_navigation()
        self.fit_view_btn.clicked.connect(self.viewer_plot.autoRange)
        self.viewer_plot.scene().sigMouseMoved.connect(self._viewer_mouse_moved)
        self.viewer_plot.scene().sigMouseClicked.connect(self._viewer_mouse_clicked)
        self.right_splitter.addWidget(viewer_panel)

        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(6, 3, 6, 3)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel('PIPELINE LOG'))
        log_header.addStretch(1)
        clear_btn = QPushButton('Clear log')
        clear_btn.clicked.connect(lambda: self.console.clear())
        log_header.addWidget(clear_btn)
        log_layout.addLayout(log_header)
        self.console = QTextEdit()
        self.console.setObjectName('pipelineConsole')
        self.console.setReadOnly(True)
        self.console.setPlaceholderText('Reduction status, errors, and print output appear here…')
        log_layout.addWidget(self.console)
        self.right_splitter.addWidget(log_panel)
        self.main_splitter.addWidget(self.right_splitter)
##        self.main_splitter.setStretchFactor(0, 0)
##        self.main_splitter.setStretchFactor(1, 1)
##        self.right_splitter.setStretchFactor(0, 1)
##        self.right_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([750, 1000])
        self.right_splitter.setSizes([510, 250])
        self._viewer_array = None
        self._viewer_kind = None
        self.statusBar().showMessage('Ready')

        # Capture ordinary Python print() without changing scientific modules.
        self._original_stdout, self._original_stderr = sys.stdout, sys.stderr
        self._gui_stdout = _ConsoleStream(self._original_stdout)
        self._gui_stderr = _ConsoleStream(self._original_stderr)
        self._gui_stdout.message.connect(self._append_console)
        self._gui_stderr.message.connect(self._append_console)
        sys.stdout, sys.stderr = self._gui_stdout, self._gui_stderr

    def _append_console(self, message):
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(message)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def closeEvent(self, event):
        if sys.stdout is self._gui_stdout:
            sys.stdout = self._original_stdout
        if sys.stderr is self._gui_stderr:
            sys.stderr = self._original_stderr
        super().closeEvent(event)

    def _viewer_mouse_moved(self, scene_pos):
        vb = self.viewer_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return
        point = vb.mapSceneToView(scene_pos)
        x, y = point.x(), point.y()
        if self._viewer_kind == 'image' and self._viewer_array is not None:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < self._viewer_array.shape[0] and 0 <= ix < self._viewer_array.shape[1]:
                self.viewer_readout.setText('X: {}   Y: {}   Counts: {:.3f}'.format(ix, iy, self._viewer_array[iy, ix]))
                return
        if self._viewer_kind == 'curve':
            self.viewer_readout.setText('X: {:.2f}   Y: {:.5g}'.format(x, y))
            return
        self.viewer_readout.setText('X: {:.1f}   Y: {:.1f}'.format(x, y))

    def show_2d_view(self, data, title='', guess=None, xlim=None):
        """Display FITS with image row=spatial and column=dispersion."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError('Viewer expects a 2D image.')
        self.viewer_plot.clear()
        self.viewer_plot.setLabel('bottom', 'Dispersion pixel')
        self.viewer_plot.setLabel('left', 'Spatial pixel')
        self._viewer_array, self._viewer_kind = data, 'image'
        img = pg.ImageItem(axisOrder='row-major')
        self.viewer_plot.addItem(img)
        finite = data[np.isfinite(data)]
        levels = np.percentile(finite, [5, 95]) if finite.size else [0, 1]
        if levels[0] >= levels[1]:
            levels[1] = levels[0] + 1
        img.setImage(data, autoLevels=False, levels=levels)
        self._image_item = img
        if guess is not None:
            self.viewer_plot.addItem(pg.InfiniteLine(pos=float(guess), angle=0, pen=pg.mkPen('#ff9c5b', width=1.5)))
        if xlim is not None:
            self.viewer_plot.addItem(pg.InfiniteLine(pos=float(xlim), angle=90, pen=pg.mkPen('#56d5f5', width=1.2)))
        self.viewer_title.setText(str(title))
        self.viewer_plot.autoRange()

    def show_curve_view(self, x, y, title='', xlabel='Pixel', ylabel='Counts'):
        """Plot a 1D spectrum/response curve in the persistent viewer."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.viewer_plot.clear()
        self.viewer_plot.setLabel('bottom', xlabel)
        self.viewer_plot.setLabel('left', ylabel)
        self.viewer_plot.plot(x, y, pen=pg.mkPen('#50c7dd', width=1.3), connect='finite')
        self._viewer_array, self._viewer_kind = y, 'curve'
        self.viewer_title.setText(str(title))
        self.viewer_plot.autoRange()

    def _update_view_navigation(self):
        n = len(self._aperture_views)
        self.view_previous_btn.setEnabled(n > 1 and self._aperture_view_index > 0)
        self.view_next_btn.setEnabled(n > 1 and self._aperture_view_index < n-1)
        self.view_counter.setText('{} / {}'.format(self._aperture_view_index+1, n) if n else '')

    def _navigate_aperture(self, step):
        idx = self._aperture_view_index + step
        if 0 <= idx < len(self._aperture_views):
            self._show_aperture_record(idx)

    def _show_aperture_record(self, idx):
        record = self._aperture_views[idx]
        self._aperture_view_index = idx
        if record['type'] == 'diagnostic':
            self.viewer_plot.clear()
            self.viewer_plot.setLabel('bottom', record['xlabel'])
            self.viewer_plot.setLabel('left', record['ylabel'])
            self.viewer_title.setText(record['title'])
            self._viewer_kind, self._viewer_array = 'curve', None
            if self.viewer_plot.plotItem.legend is None:
                self.viewer_plot.addLegend(offset=(-10, 10))
            for x, y, name, color, scatter in record['series']:
                self.viewer_plot.plot(x, y, name=name, pen=None if scatter else pg.mkPen(color, width=1.5),
                                      symbol='o' if scatter else None, symbolSize=6, symbolBrush=color, connect='finite')
            for pos, label, color in record.get('lines', []):
                self.viewer_plot.addItem(pg.InfiniteLine(pos=pos, angle=90, pen=pg.mkPen(color, style=Qt.DashLine), label=label, labelOpts={'position': 0.7, 'color': color, 'rotateAxis': (1, 0)}))
            self.viewer_plot.autoRange()
        elif record['type'] == 'image':
            self.show_2d_view(record['data'], record['title'])
            x = np.arange(len(record['trace']))
            t = record['trace']
            pen = lambda color, width=1.3: pg.mkPen(color, width=width)
            def overlay(offset, color, width=1.3):
                self.viewer_plot.plot(x, t+offset, pen=pen(color, width), connect='finite')
            overlay(0, '#56e9e6', 2.0)
            if record.get('apwidth') is not None:
                a, gap, skyw = record['apwidth'], record['skysep'], record['skywidth']
                for offset in (-a, a): overlay(offset, '#56e9e6')
                for offset in (-a-gap, -a-gap-skyw, a+gap, a+gap+skyw):
                    overlay(offset, '#fbb96a')
            details = record.get('details')
            if details:
                pts = details['accepted'] & np.isfinite(details['bin_y'])
                item = pg.ScatterPlotItem(x=details['bin_x'][pts], y=details['bin_y'][pts],
                                          pen=None, brush=pg.mkBrush('#ffdf84'), size=5)
                self.viewer_plot.addItem(item)
        else:
            curves = record['curves']
            self.viewer_plot.clear()
            self.viewer_plot.setLabel('bottom', 'Dispersion pixel')
            self.viewer_plot.setLabel('left', 'Counts (ADU)')
            self._viewer_kind = 'curve'
            self._viewer_array = curves['net']
            self.viewer_title.setText(record['title'])
            x = np.arange(len(curves['net']))
            self.viewer_plot.addLegend(offset=(-10, 10)) if self.viewer_plot.plotItem.legend is None else None
            for key, color in [('raw', '#6ea8ff'), ('sky', '#f8ad62'), ('net', '#62e5c2')]:
                self.viewer_plot.plot(x, curves[key], pen=pg.mkPen(color, width=1.4),
                                      name=key, connect='finite')
            self.viewer_plot.autoRange()
        self._update_view_navigation()

    def _diagnostic_views(self, records):
        self._aperture_views = [dict(type='diagnostic', **record) for record in records]
        self._show_aperture_record(0)

    def _arc_views(self, wave, flux, details, title):
        d = details
        p = np.arange(len(flux))
        previous = self._aperture_view_index if self.viewer_title.text().startswith(title+' —') else 0
        summary = f"RMS {d['rms']:.4f} Å | {len(d['pixels'])} accepted, {len(d['rejected_pixels'])} rejected"
        accepted = [(x, f'{w:.2f}', '#f8ad62') for x,w in zip(d['pixels'],d['wavelengths'])]
        rejected = [(x, f'{w:.2f} rejected', '#ff6969') for x,w in zip(d['rejected_pixels'],d['rejected_wavelengths'])]
        self._diagnostic_views([
            dict(title=title+' — Identified lines | '+summary, xlabel='Pixel', ylabel='Arc counts', series=[(p, flux, 'Arc', '#50c7dd', False)], lines=accepted+rejected),
            dict(title=title+' — Calibrated arc | '+summary, xlabel='Wavelength (Å)', ylabel='Arc counts', series=[(wave, flux, 'Calibrated arc', '#50c7dd', False)], lines=[(w, f'{w:.2f}', '#f8ad62') for w in d['wavelengths']]),
            dict(title=title+' — Wavelength fit | '+summary, xlabel='Pixel', ylabel='Wavelength (Å)', series=[(p, wave, 'Chebyshev solution', '#50c7dd', False), (d['pixels'], d['wavelengths'], 'Accepted lines', '#f8ad62', True), (d['rejected_pixels'], d['rejected_wavelengths'], 'Rejected lines', '#ff6969', True)]),
            dict(title=title+' — Residuals | '+summary, xlabel='Reference wavelength (Å)', ylabel='Fitted − reference (Å)', series=[(d['residual_wave'], d['residual'], 'Accepted', '#f8ad62', True), (d['rejected_wavelengths'], d['rejected_residual'], 'Rejected', '#ff6969', True), (np.array([np.min(wave), np.max(wave)]), np.zeros(2), 'Zero', '#50c7dd', False)])])
        if previous > 0:
            self._show_aperture_record(previous)

    def _add_aperture_views(self, frame, trace, details, arrays, apwidth, skysep, skywidth, name):
        """Three switchable views; both 2D panels share the same source image."""
        self._aperture_views.extend([
            dict(type='image', title=name+' — Traced spectrum', data=frame, trace=trace,
                 details=details, apwidth=None),
            dict(type='image', title=name+' — Aperture and sky regions', data=frame, trace=trace,
                 apwidth=apwidth, skysep=skysep, skywidth=skywidth),
            dict(type='curve', title=name+' — Raw / Sky / Sky-subtracted', curves=arrays)
        ])
        self._show_aperture_record(len(self._aperture_views)-3)

    def log_message(self, message):
        """Add message to console and status bar"""
        self.console.append(message)
        self.statusBar().showMessage(message)
        # The Qt event loop updates the console naturally; do not re-enter it here.


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
        self.yaml_pattern = QLineEdit(os.path.join(script_dir, 'fileinfo.yaml'))
        yaml_layout.addWidget(self.yaml_pattern)
        yaml_browse = QPushButton('Browse YAML…')
        yaml_browse.clicked.connect(self.select_yaml_file)
        yaml_layout.addWidget(yaml_browse)
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
        dir_button = QPushButton('Browse…')
        dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(dir_button)
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
        file_layout.addWidget(self.create_info_label('trim', self.trim_check, checkbox=True))
        
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
        # The QWidget already owns its layout.
        self.tabs.addTab(tab, "File Selection")

    def select_directory(self):
        """Select working directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_path.setText(directory)
            
    def select_yaml_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose pipeline YAML', script_dir, 'YAML files (*.yaml *.yml);;All files (*)')
        if path:
            self.yaml_pattern.setText(path)

    def yread_files(self):
        yfile = str(self.yaml_pattern.text())
        print(" the yaml file ",yfile)
        if not os.path.isfile(yfile):
            QMessageBox.warning(self, 'YAML not found', f'Cannot find: {yfile}')
            return
        try:
            with open(yfile, 'r', encoding='utf-8') as file:
                yamlfile = yaml.safe_load(file)
            if not isinstance(yamlfile, dict):
                raise ValueError('YAML root must be a mapping.')
        except (OSError, yaml.YAMLError, ValueError) as exc:
            QMessageBox.critical(self, 'Invalid YAML', str(exc))
            return

        self.workdir             = yamlfile["workdir"]
        if not os.path.isabs(self.workdir):
            self.workdir = os.path.normpath(os.path.join(os.path.dirname(yfile), self.workdir))
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
        self.slit.setCurrentText(str(yamlfile.get('slit', self.slit.currentText())))
        self.grism.setCurrentText(str(yamlfile.get('grism', self.grism.currentText())))
        self.trim_check.setChecked(str(yamlfile.get('trim', True)).lower() in ('true', '1', 'yes'))
        self.spatial_axis.setCurrentText(str(yamlfile.get('spatial_axis', '0')))
        self.log_message(f'Loaded YAML settings: {os.path.basename(yfile)}')

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
                trim=self.trim_check.isChecked()
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
        options_layout.addWidget(self.create_info_label('flat_smoothing', 'Smoothing:'))
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
        options_layout.addWidget(self.create_info_label('cosmic_ray', 'Cosmic-ray correction:'))
        self.cosmic_corr = QCheckBox()
        self.cosmic_corr.setChecked(True)
        options_layout.addWidget(self.cosmic_corr)
        flat_layout.addLayout(options_layout)
        
        # Perform flat correction button
        flat_button = QPushButton("Perform Flat Correction (Flat Correction and Cosmic-ray removal)")
        flat_button.clicked.connect(self.perform_flat_correction)
        flat_layout.addWidget(flat_button)
        
        layout.addWidget(flat_group)
        # The QWidget already owns its layout.
        self.tabs.addTab(tab, "Flat Correction")

    
    def perform_flat_correction(self):
        """Normalize the continuum flat and show it in the integrated viewer."""
        try:
            if not getattr(self, 'flatin', None):
                raise ValueError('Read and bias-correct the input frames first.')
            master_path = self.master_flat_name.text().strip()
            normalized_path = self.corrected_flat_name.text().strip()
            if not master_path or not normalized_path:
                raise ValueError('Specify master-flat and normalized-flat filenames.')
            self.log_message('Preparing normalized continuum flat...')
            normalized, response, raw_response = flat_2.lamp_response(
                master_path, smooth=self.smooth_check.isChecked(), Saxis=sxs,
                display=False, output=normalized_path, return_response=True)
            self.log_message('Normalized flat saved: {}'.format(normalized_path))
            if self.display_flat_check.isChecked():
                # Do not open Matplotlib; the same PyQtGraph viewer handles all images.
                frame_to_show = normalized if sxs == 0 else normalized.T
                self.show_2d_view(frame_to_show, title='Normalized continuum flat: {}'.format(normalized_path))
            self.log_message('Applying flat correction to {} frames...'.format(len(self.flatin)))
            self.lst = flat_2.flat_correction(self.flatin, normalized_path,
                cosmic_correction=self.cosmic_corr.isChecked())
            self.sci_lst = [name for name in self.lst if self.sciname in name]
            self.std_lst = [name for name in self.lst if self.stdname in name]
            arcs = [name for name in self.lst if Path(name).name.lower().startswith('fc_com_')]
            if not arcs:
                arcs = [name for name in self.lst if 'lamp' in Path(name).name.lower()]
            self.arclst = arcs[0] if arcs else None
            self.log_message('Flat correction complete: {} science, {} standard, {} arc.'.format(
                len(self.sci_lst), len(self.std_lst), len(arcs)))
            self.tabs.setTabEnabled(2, True)
        except Exception as exc:
            self.log_message('Flat correction failed: {}'.format(exc))
            QMessageBox.critical(self, 'Flat correction', str(exc))


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
        # Preview is displayed in the persistent upper-right PyQtGraph viewer.
        
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
        preview_controls.addWidget(self.create_info_label('guess', 'Guess Value:'))
        self.preview_guess_input = QSpinBox()
        self.preview_guess_input.setRange(0, 2048)
        preview_controls.addWidget(self.preview_guess_input)
        
        # X Limit input
        preview_controls.addWidget(self.create_info_label('xlim', 'X Limit:'))
        self.preview_xlim_input = QSpinBox()
        self.preview_xlim_input.setRange(0, 2048)
        self.preview_xlim_input.setValue(1500)
        preview_controls.addWidget(self.preview_xlim_input)
        
        # Preview button
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.preview_selected_image)
        preview_controls.addWidget(preview_btn)
        
        preview_layout.addLayout(preview_controls)
        trace_controls = QHBoxLayout()
        trace_controls.addWidget(self.create_info_label('bin_width', 'Bin width:'))
        self.preview_bin_width = QSpinBox(); self.preview_bin_width.setRange(1, 300); self.preview_bin_width.setValue(20)
        trace_controls.addWidget(self.preview_bin_width)
        trace_controls.addWidget(self.create_info_label('ap_width', 'Aperture ±:'))
        self.preview_ap_width = QSpinBox(); self.preview_ap_width.setRange(1, 100); self.preview_ap_width.setValue(6)
        trace_controls.addWidget(self.preview_ap_width)
        trace_controls.addWidget(self.create_info_label('sky_gap', 'Sky gap:'))
        self.preview_sky_sep = QSpinBox(); self.preview_sky_sep.setRange(1, 100); self.preview_sky_sep.setValue(6)
        trace_controls.addWidget(self.preview_sky_sep)
        self.click_guess_btn = QPushButton('Pick Y on image')
        self.click_guess_btn.setCheckable(True)
        self.click_guess_btn.setToolTip('Enable, then left-click the spectrum on the image to choose the initial Y position')
        trace_controls.addWidget(self.click_guess_btn)
        preview_layout.addLayout(trace_controls)
        self.live_trace_checkbox = QCheckBox('Live trace and extraction-region preview')
        self.live_trace_checkbox.setChecked(True)
        preview_layout.addWidget(self.live_trace_checkbox)
        self.preview_status = QLabel('Select a file to inspect its trace')
        self.preview_status.setObjectName('subtitle')
        preview_layout.addWidget(self.preview_status)
        self._preview_data = None
        self._preview_path = None
        self._preview_internal_update = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(350)
        self._preview_timer.timeout.connect(self._update_live_trace)
        self.preview_file_combo.currentIndexChanged.connect(self._preview_file_changed)
        for control in (self.preview_guess_input, self.preview_xlim_input, self.preview_bin_width,
                        self.preview_ap_width, self.preview_sky_sep):
            control.valueChanged.connect(self._preview_parameter_changed)
        self.live_trace_checkbox.toggled.connect(self._preview_parameter_changed)
        self.click_guess_btn.toggled.connect(self._guess_mode_changed)
        layout.addWidget(preview_group)
        
        # Shared extraction settings; valid for both science and standard files.
        common = QGroupBox('Trace / Extraction configuration')
        common_layout = QVBoxLayout()
        common_row1 = QHBoxLayout()
        common_row1.addWidget(self.create_info_label('search_width', 'Search ±'))
        self.ap_search_width = QSpinBox()
        self.ap_search_width.setRange(3, 100)
        self.ap_search_width.setValue(12)
        common_row1.addWidget(self.ap_search_width)
        common_row1.addWidget(self.create_info_label('max_shift', 'Max shift'))
        self.ap_max_shift = QSpinBox()
        self.ap_max_shift.setRange(1, 100)
        self.ap_max_shift.setValue(8)
        common_row1.addWidget(self.ap_max_shift)
        common_row1.addWidget(self.create_info_label('sky_width', 'Sky width'))
        self.ap_sky_width = QSpinBox()
        self.ap_sky_width.setRange(2, 100)
        self.ap_sky_width.setValue(7)
        common_row1.addWidget(self.ap_sky_width)
        common_row2 = QHBoxLayout()
        common_row2.addWidget(self.create_info_label('sky_degree', 'Sky polynomial degree'))
        self.ap_sky_degree = QSpinBox()
        self.ap_sky_degree.setRange(0, 2)
        self.ap_sky_degree.setValue(0)
        common_row2.addWidget(self.ap_sky_degree)
        common_row2.addWidget(self.create_info_label('psf_sigma', 'PSF sigma (px)'))
        self.ap_profile_sigma = QDoubleSpinBox()
        self.ap_profile_sigma.setRange(0.5, 20.)
        self.ap_profile_sigma.setValue(2.5)
        self.ap_profile_sigma.setSingleStep(0.25)
        common_row2.addWidget(self.ap_profile_sigma)
        common_layout.addLayout(common_row1)
        common_layout.addLayout(common_row2)
        common.setLayout(common_layout)
        layout.addWidget(common)

        for control in (self.ap_search_width, self.ap_max_shift, self.ap_sky_width,
                        self.ap_sky_degree, self.ap_profile_sigma):
            control.valueChanged.connect(self._preview_parameter_changed)

        # 2. Science Object Section
        sci_group = QGroupBox("Science Object Extraction")
        sci_layout = QVBoxLayout(sci_group)
        
        # Science files table with Combine ID column
        self.sci_table = QTableWidget(0, 8)  # 8 columns
        self.sci_table.setHorizontalHeaderLabels(["File", "Guess Value", "X Limit", "Bin width","Ap width","Sky sep", "Method", "Combine ID"])   #2207
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
        sci_controls.addWidget(self.create_info_label('combine_method', 'Combine Method:'))
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
        self.std_table.setHorizontalHeaderLabels(["File", "Guess Value", "X Limit", "Bin width", "Method", "Combine ID"])
        self._add_table_help(self.std_table, {1:'guess', 2:'xlim', 3:'bin_width', 4:'extract_method', 5:'combine_id'})
        self.std_table.horizontalHeader().setStretchLastSection(True)
        std_layout.addWidget(self.std_table)
        self.sci_table.itemChanged.connect(lambda item: self._table_parameter_changed('science', item))
        self.std_table.itemChanged.connect(lambda item: self._table_parameter_changed('standard', item))
        
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
        
        std_controls.addWidget(self.create_info_label('combine_method', 'Combine Method:'))
        self.std_combine_method = QComboBox()
        self.std_combine_method.addItems(["mean", "invvar", "median"])
        self.std_combine_method.setCurrentIndex(0)
        std_controls.addWidget(self.std_combine_method)
        
        std_layout.addLayout(std_controls)
        layout.addWidget(std_group)
        
        self.tabs.addTab(tab, "Aperture Extraction")


    def aperture_extract(self, sci_lst, otype='src', arcfile=None, sourcename=None,
                         initial_guess=500, method='sum', xlim=1500, nbin=20, apw=6,
                         skysep=6, display_trace=True, display_extract=True,
                         bin_width=None, search_halfwidth=12, max_shift=8, skywidth=7,
                         skydeg=0, profile_sigma=2.5):
        """Extract and stage interactive viewer records without opening new figures."""
        AP_data, source_am, ARC_data = [], [], []
        for idx, sciff in enumerate(sci_lst):
            self.log_message('Extracting {}'.format(sciff))
            scif = ccdproc.CCDData.read(sciff, unit='adu')
            # Do not replace NaNs by zeros: masked/bad pixels must be excluded.
            jd = scif.header.get('JD')
            if jd is not None and otype == 'src':
                self.sci_mjd = round(float(jd)-2400000.5, 5)
            am = airmass.Airmass(sciff, sourcename)
            source_am.append(am)
            name = 'A_'+os.path.basename(sciff)
            trace, details = aperture.trfunc(scif, filename=name, nbins=int(nbin),
                window=search_halfwidth, guess=initial_guess, Saxis=sxs, Xlim=xlim,
                bin_width=(bin_width if bin_width is not None else int(nbin)),
                max_shift=max_shift, display=False, return_details=True)
            raw, sky, arrays = aperture.boxfunc(scif, trace, filename=name,
                airmass=am, apwidth=apw, skysep=skysep, skywidth=skywidth,
                skydeg=skydeg, Saxis=sxs, method=method, otype=otype,
                profile_sigma=profile_sigma, display=False, return_arrays=True)
            extracted = Spectrum1D(flux=(raw.flux-sky.flux), spectral_axis=raw.spectral_axis,
                uncertainty=StdDevUncertainty(raw.uncertainty.array), meta={'header': scif.header})
            AP_data.append(extracted)
            hdr = scif.header.copy()
            if am is not None: hdr['AIRMASS'] = float(am)
            fits.writeto(Path(name).with_suffix('.fits'), extracted.flux.value, hdr, overwrite=True)
            image = np.asarray(scif.data, dtype=float)
            if sxs == 1: image = image.T
            self._add_aperture_views(image, trace, details, arrays, apw, skysep, skywidth,
                                     os.path.basename(sciff))
            if otype == 'src' and idx == 0 and arcfile:
                arcf = ccdproc.CCDData.read(arcfile, unit='adu')
                arc_raw, _, _ = aperture.BoxcarExtract(arcf, trace, filename='arc_'+name,
                    Saxis=sxs, method='sum', otype='lamp', apwidth=apw,
                    skysep=skysep, skywidth=skywidth, display=False, return_arrays=True)
                ARC_data.append(arc_raw)
                fits.writeto('A_ap_arc_'+os.path.basename(arcfile), arc_raw.flux.value,
                             arcf.header, overwrite=True)
        if otype == 'src':
            return AP_data, source_am, ARC_data
        return AP_data, source_am

    def _tab_changed(self, index):
        if index == 2 and getattr(self, '_preview_data', None) is not None:
            self._schedule_trace_preview()
        elif self.tabs.tabText(index) == 'Output':
            self.plot_final_spectrum()

    def _guess_mode_changed(self, enabled):
        self.click_guess_btn.setText('Click spectrum in viewer…' if enabled else 'Pick Y on image')
        if enabled:
            self.preview_status.setText('Click on the 2D image to set the spatial Y guess')

    def _viewer_mouse_clicked(self, event):
        """Use left mouse click on a 2D preview to update the spatial guess."""
        if not self.click_guess_btn.isChecked() or event.button() != Qt.LeftButton:
            return
        if self.tabs.currentIndex() != 2 or self._viewer_kind != 'image':
            return
        vb = self.viewer_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        point = vb.mapSceneToView(event.scenePos())
        x, y = point.x(), point.y()
        if self._preview_data is None:
            return
        ny, nx = self._preview_data.shape
        if not (0 <= x < nx and 0 <= y < ny):
            return
        self.preview_guess_input.setValue(int(round(y)))
        self.click_guess_btn.setChecked(False)
        event.accept()

    def _preview_row(self):
        """Locate selected preview file in the science or standard table."""
        name = self.preview_file_combo.currentText()
        for group, files, table in (('science', self.sci_lst, self.sci_table),
                                    ('standard', self.std_lst, self.std_table)):
            for row, path in enumerate(files):
                if os.path.basename(path) == name:
                    return group, row, table
        return None, None, None

    def _preview_file_changed(self, *_):
        self._preview_data = None
        self._preview_path = None
        group, row, table = self._preview_row()
        if table is not None and row < table.rowCount():
            self._preview_internal_update = True
            try:
                for col, widget in ((1, self.preview_guess_input), (2, self.preview_xlim_input),
                                    (3, self.preview_bin_width)):
                    item = table.item(row, col)
                    if item is not None:
                        try: widget.setValue(int(item.text()))
                        except ValueError: pass
                if group == 'science':
                    for col, widget in ((4, self.preview_ap_width), (5, self.preview_sky_sep)):
                        item = table.item(row, col)
                        if item is not None:
                            try: widget.setValue(int(item.text()))
                            except ValueError: pass
            finally:
                self._preview_internal_update = False
        if self.preview_file_combo.currentText():
            self.preview_selected_image()

    def _table_parameter_changed(self, group, item):
        if self._preview_internal_update or item.column() not in (1, 2, 3, 4, 5):
            return
        selected_group, row, _ = self._preview_row()
        if selected_group != group or row != item.row():
            return
        mapping = {1: self.preview_guess_input, 2: self.preview_xlim_input,
                   3: self.preview_bin_width}
        if group == 'science':
            mapping.update({4: self.preview_ap_width, 5: self.preview_sky_sep})
        widget = mapping.get(item.column())
        if widget is None:
            return
        try: value = int(item.text())
        except ValueError: return
        self._preview_internal_update = True
        try: widget.setValue(value)
        finally: self._preview_internal_update = False
        self._schedule_trace_preview()

    def _preview_parameter_changed(self, *_):
        if self._preview_internal_update:
            return
        group, row, table = self._preview_row()
        if table is not None and row < table.rowCount():
            columns = {1: self.preview_guess_input, 2: self.preview_xlim_input,
                       3: self.preview_bin_width}
            if group == 'science':
                columns.update({4: self.preview_ap_width, 5: self.preview_sky_sep})
            self._preview_internal_update = True
            try:
                for col, widget in columns.items():
                    item = table.item(row, col)
                    if item is not None:
                        item.setText(str(widget.value()))
            finally:
                self._preview_internal_update = False
        self._schedule_trace_preview()

    def _schedule_trace_preview(self):
        if not hasattr(self, 'viewer_plot') or not self.preview_file_combo.currentText():
            return
        self._preview_timer.start()

    def _get_preview_image(self):
        name = self.preview_file_combo.currentText()
        if not name:
            return None
        # Resolve against the current reduction folder, where processed frames reside.
        full_path = name if os.path.isabs(name) else os.path.join(os.getcwd(), name)
        if self._preview_path != full_path:
            with fits.open(full_path, memmap=False) as hdul:
                data = next((hdu.data for hdu in hdul if hdu.data is not None), None)
                if data is None:
                    raise ValueError('FITS file contains no image')
                frame = np.array(data, dtype=float, copy=True)
            if frame.ndim != 2:
                raise ValueError('Expected a 2D FITS image')
            self._preview_data = frame if sxs == 0 else frame.T
            self._preview_path = full_path
        return self._preview_data

    def _update_live_trace(self):
        """Recompute only the trace; no spectral extraction or file writing."""
        if self.tabs.currentIndex() != 2:
            return
        try:
            image = self._get_preview_image()
            if image is None:
                return
            ny, nx = image.shape
            self.preview_guess_input.setMaximum(max(0, ny-1))
            self.preview_xlim_input.setMaximum(nx)
            xlim = max(6, min(nx, self.preview_xlim_input.value()))
            guess = self.preview_guess_input.value()
            self.show_2d_view(image, os.path.basename(self._preview_path), guess=guess, xlim=xlim)
            self.preview_status.setText('X = 0–{}, selected Y = {}. Trace not calculated.'.format(xlim-1, guess))
            if not self.live_trace_checkbox.isChecked():
                return
            trace, details = aperture.trace_sloped(image, guess=guess, Xlim=xlim,
                bin_width=self.preview_bin_width.value(),
                search_halfwidth=self.ap_search_width.value(), max_shift=self.ap_max_shift.value(),
                poly_order=1, Saxis=0, return_details=True)
            x = np.arange(len(trace))
            self.viewer_plot.plot(x, trace, pen=pg.mkPen('#58e5d6', width=2), connect='finite')
            a, gap, skyw = self.preview_ap_width.value(), self.preview_sky_sep.value(), self.ap_sky_width.value()
            for offset, color in ((-a, '#65c9f5'), (a, '#65c9f5'),
                                  (-a-gap, '#ffbf69'), (-a-gap-skyw, '#ffbf69'),
                                  (a+gap, '#ffbf69'), (a+gap+skyw, '#ffbf69')):
                self.viewer_plot.plot(x, trace+offset, pen=pg.mkPen(color, width=1), connect='finite')
            accepted = details['accepted'] & np.isfinite(details['bin_y'])
            if np.any(accepted):
                self.viewer_plot.addItem(pg.ScatterPlotItem(
                    x=details['bin_x'][accepted], y=details['bin_y'][accepted],
                    pen=None, brush=pg.mkBrush('#fff09b'), size=5))
            self.preview_status.setText('Trace OK | accepted bins: {} / {} | Y at centre: {:.2f}'.format(
                np.count_nonzero(accepted), len(accepted), trace[len(trace)//2]))
        except Exception as exc:
            self.preview_status.setText('Trace preview: {}'.format(exc))
            self.statusBar().showMessage('Trace preview: {}'.format(exc))

    def preview_selected_image(self):
        """Load image and show the live trace in the upper-right viewer."""
        try:
            self._get_preview_image()
            self._preview_timer.stop()
            self._update_live_trace()
        except Exception as exc:
            self.log_message('Preview error: {}'.format(exc))

    def update_file_lists(self):
        """Update file lists with Combine ID selection"""
        self._aperture_views = []
        self._aperture_view_index = -1
        self._update_view_navigation()
        self.preview_file_combo.blockSignals(True)
        self.preview_file_combo.clear()
        all_files = self.sci_lst + self.std_lst + ([self.arclst] if self.arclst else [])
        self.preview_file_combo.addItems([os.path.basename(f) for f in all_files])
        self.preview_file_combo.blockSignals(False)
        
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
            method_combo.addItems(["sum", "variance"])
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
            method_combo.addItems(["sum", "variance"])
            self.std_table.setCellWidget(i, 4, method_combo)
            
            # Combine ID combo
            combine_combo = QComboBox()
            combine_combo.addItems(["0", "1", "2", "3"])
            combine_combo.setCurrentIndex(0)
            self.std_table.setCellWidget(i, 5, combine_combo)
        
        self.sci_table.resizeColumnsToContents()
        self.std_table.resizeColumnsToContents()
        self._preview_file_changed()

    def extract_science(self):
        """Perform aperture extraction for science objects"""
        try:
            if not hasattr(self, 'sci_lst') or not self.sci_lst:
                raise ValueError("No science files available")
            
            self.log_message("Performing science object extraction...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            self.sci_results = []  # Initialize SCIdata
            self.sci_airmass = []
            self.combined_spectra['SCI'].clear()
            self.W_sci_groups, self.W_std_groups, self.final_groups = {}, {}, {}
            self.flux_calibrated = self.final_spec = None
            self.flux_cal_btn.setEnabled(False)
            
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
                    search_halfwidth=self.ap_search_width.value(),
                    max_shift=self.ap_max_shift.value(),
                    skywidth=self.ap_sky_width.value(),
                    skydeg=self.ap_sky_degree.value(),
                    profile_sigma=self.ap_profile_sigma.value(),
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
            self.combined_spectra['STD'].clear()
            self.W_sci_groups, self.W_std_groups, self.final_groups = {}, {}, {}
            self.flux_calibrated = self.final_spec = None
            self.flux_cal_btn.setEnabled(False)
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
                    apw=self.preview_ap_width.value(),
                    skysep=self.preview_sky_sep.value(),
                    search_halfwidth=self.ap_search_width.value(),
                    max_shift=self.ap_max_shift.value(),
                    skywidth=self.ap_sky_width.value(),
                    skydeg=self.ap_sky_degree.value(),
                    profile_sigma=self.ap_profile_sigma.value(),
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
            grouped_rows = {}
            for i, spec in enumerate(data):
                if i < table.rowCount():
                    try:
                        combine_id = table.cellWidget(i, int(combineID)).currentText()  # Combine ID is column 5 for std and 7 for sci    #2207
                        print("combined_id",combine_id)
                        #print(combine_id.text())
                        if combine_id not in grouped_data:
                            grouped_data[combine_id] = []
                        grouped_data[combine_id].append(spec)
                        grouped_rows.setdefault(combine_id, []).append(i)
                    except Exception as e:
                        self.log_message(f"Error processing row {i}: {str(e)}")
                        continue

            # Process selected group(s)
            if group == "All":
                # Process all groups
                self.combined_spectra[prefix].clear()
                for gid, specs in grouped_data.items():
                    combined = self.combine_spectrum1d_stack(specs, method=method)
                    self._store_group_metadata(combined, spec_type, gid, grouped_rows[gid], method)
                    self.combined_spectra[prefix][gid] = combined
                    print(f"combining for {gid}")
                    self._save_combined_spectrum(combined, spec_type, gid)
            else:
                # Process specific group
                if group not in grouped_data:
                    raise ValueError(f"No {spec_type} spectra in group {group}")
                combined = self.combine_spectrum1d_stack(grouped_data[group], method=method)
                self._store_group_metadata(combined, spec_type, group, grouped_rows[group], method)
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
            self.W_sci_groups, self.W_std_groups = {}, {}
            self.final_groups = {}
            self.flux_calibrated = self.final_spec = None
            self.flux_cal_btn.setEnabled(False)
            self.plot_combined_spectrum(spec_type, group)

        except Exception as e:
            error_msg = f"Error combining {spec_type} group {group}: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _store_group_metadata(self, spectrum, spec_type, group_id, rows, method):
        files, am = (self.sci_lst, self.sci_airmass) if spec_type == 'science' else (self.std_lst, self.std_airmass)
        values = np.asarray([am[i] for i in rows], dtype=float)
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError(f'Invalid airmass in {spec_type} group {group_id}')
        spectrum.meta.update(group_id=group_id, input_files=[str(files[i]) for i in rows], airmass=float(np.mean(values)), combine_method=method)

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
        mask = ~np.isfinite(flux_stack) | ~np.isfinite(fluxerr_stack) | (fluxerr_stack <= 0)
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
            combined_fluxerr = np.sqrt(np.divide(1.0, sum_weights,
                                                  out=np.full_like(sum_weights, np.nan), where=sum_weights > 0))
        
        elif method == "mean":
            valid = np.isfinite(flux_stack) & np.isfinite(fluxerr_stack) & (fluxerr_stack >= 0)
            n = valid.sum(axis=0)
            combined_flux = np.divide(np.nansum(np.where(valid, flux_stack, 0.0), axis=0), n,
                                      out=np.full(flux_stack.shape[1:], np.nan), where=n > 0)
            combined_fluxerr = np.divide(np.sqrt(np.sum(np.where(valid, fluxerr_stack**2, 0.0), axis=0)), n,
                                         out=np.full(flux_stack.shape[1:], np.nan), where=n > 0)

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

    def plot_combined_spectrum(self, spec_type, group='All'):
        key = 'SCI' if spec_type == 'science' else 'STD'
        groups = getattr(self, 'combined_spectra', {}).get(key, {})
        if not groups:
            return
        selected = list(groups) if group == 'All' else [group] if group in groups else []
        if selected:
            self._diagnostic_views([dict(title=f'Combined {spec_type} | group {gid}', xlabel='Pixel', ylabel=f'Flux ({groups[gid].flux.unit})', series=[(groups[gid].spectral_axis.value, groups[gid].flux.value, f'Group {gid}', '#50c7dd', False)]) for gid in selected])

    def setup_spectral_calibration_tab(self):
        """Tab for spectral calibration with interactive steps"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Spectral calibration group
        spec_group = QGroupBox("Spectral Calibration")
        spec_layout = QVBoxLayout(spec_group)
        
        # ===== 1. Arc Line File Selection =====
        arc_layout = QHBoxLayout()
        arc_layout.addWidget(self.create_info_label('arc_file', 'Arc Line File:'))

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
        self.line_func.addItems(['chebyshev'])
        self.line_func.setCurrentIndex(0)
        id_params_layout.addWidget(self.line_func)

        id_params_layout.addWidget(self.create_info_label('poly_degree', 'Deg:'))
        self.deg_val = QSpinBox()
        self.deg_val.setRange(1, 20)
        self.deg_val.setValue(3)
        id_params_layout.addWidget(self.deg_val)

        id_params_layout.addWidget(self.create_info_label('match_tolerance', 'Match tolerance (Å):'))
        self.search_width = QSpinBox()
        self.search_width.setRange(1, 100)
        self.search_width.setValue(10)
        id_params_layout.addWidget(self.search_width)
        
        id_params_layout.addWidget(self.create_info_label('peak_threshold', 'Peak percentile (fraction):'))
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
        refine1_layout.addWidget(self.create_info_label('match_tolerance', 'Tolerance (Å):'))
        self.refine_width = QSpinBox()
        self.refine_width.setRange(1, 50)
        self.refine_width.setValue(10)
        refine1_layout.addWidget(self.refine_width)
        
        refine1_layout.addWidget(self.create_info_label('peak_threshold', 'Peak percentile:'))
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
        refine2_layout.addWidget(QLabel("Tolerance (Å):"))
        self.final_refine_width = QSpinBox()
        self.final_refine_width.setRange(1, 50)
        self.final_refine_width.setValue(10)
        refine2_layout.addWidget(self.final_refine_width)
        
        refine2_layout.addWidget(QLabel("Peak percentile:"))
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
        self._arc_stage = 0
        self._arc_preview_timer = QTimer(self)
        self._arc_preview_timer.setSingleShot(True)
        self._arc_preview_timer.timeout.connect(self._refresh_arc_preview)
        for widget in (self.line_func, self.deg_val, self.search_width, self.threshold):
            signal = widget.currentTextChanged if isinstance(widget, QComboBox) else widget.valueChanged
            signal.connect(lambda *_: self._arc_parameters_changed(1))
        for widget in (self.refine_width, self.refine_threshold):
            widget.valueChanged.connect(lambda *_: self._arc_parameters_changed(2))
        for widget in (self.final_refine_width, self.final_refine_threshold):
            widget.valueChanged.connect(lambda *_: self._arc_parameters_changed(3))

    def _arc_parameters_changed(self, stage):
        if self._arc_stage >= stage:
            self._arc_stage = stage
            self._invalidate_arc(stage)
            self._arc_preview_timer.start(350)

    def _invalidate_arc(self, stage):
        self.W_sci_groups, self.W_std_groups = {}, {}
        self.final_groups = {}
        self.flux_calibrated = self.final_spec = None
        self.flux_cal_btn.setEnabled(False)
        for n, attr in enumerate(('wave2', 'wave3', 'final_wave'), 1):
            if n >= stage and hasattr(self, attr):
                delattr(self, attr)
        if stage <= 1:
            self.confirm_identify_btn.setEnabled(False)
            self.refine1_btn.setEnabled(False)
        if stage <= 2:
            self.refine2_btn.setEnabled(False)
        self.confirm_final_btn.setEnabled(False)

    def _refresh_arc_preview(self):
        if self._arc_stage and self.tabs.tabText(self.tabs.currentIndex()) == 'Spectral Calibration':
            (self.identify_lines, self.first_refinement, self.final_refinement)[self._arc_stage-1]()


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
                self._actual_arc_path = str(default_path)
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
        """Resolve the selected arc reference correctly on Linux or Windows."""
        name = self.arc_file.text().strip()
        if not name:
            raise ValueError('Please enter an arc reference filename.')
        candidates = [name]
        if getattr(self, '_actual_arc_path', None) and os.path.basename(self._actual_arc_path) == os.path.basename(name):
            candidates.insert(0, self._actual_arc_path)
        candidates.append(os.path.join(self.resource_path.text().strip(), 'arc', os.path.basename(name)))
        for path in candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)
        raise FileNotFoundError('Arc reference not found: ' + ', '.join(candidates))

    def estimate_wavelength_solution(self):
        """Interactive estimation of offset/dispersion"""
        try:
            # Validate we have the required data
            if not hasattr(self, 'ARCdata'):
                raise ValueError("No arc data loaded")
            
            # Validate arc filename
            arcfile = self.get_arc_path().strip()
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
        self._arc_preview_timer.stop()
        self._arc_stage = 0
        self._invalidate_arc(1)
        self.log_message(f"Initial solution confirmed: offset={self.offset:.2f}, disp={self.dispersion:.2f}")

    def identify_lines(self):
        """First line identification"""
        try:
            self._arc_preview_timer.stop()
            self._arc_stage = 1
            self._invalidate_arc(1)
            sciarc_ex = self.ARCdata[0]
            arcfile = self.get_arc_path()
            
            # Calculate initial wavelengths
            sci_xp = sciarc_ex.spectral_axis.value
            self.wave1 = add.XFIT(sci_xp, self.offset, self.dispersion)
            print(f"func for line identification {self.line_func.currentText()}")
            # First identification
            self.wave2, details = identify.fit_arclines(
                self.wave1,
                sciarc_ex.flux.value,
                arcfile,
                autotol=self.search_width.value(),
                pthreshold=self.threshold.value(),
                func=self.line_func.currentText(),deg=self.deg_val.value(), display=False, return_details=True
            )
            
            self._arc_views(self.wave2, sciarc_ex.flux.value, details, 'Arc line identification')
            self.confirm_identify_btn.setEnabled(True)
            self.refine1_btn.setEnabled(True)
            self.log_message("Line identification completed")
            
        except Exception as e:
            self.log_message(f"Error in line identification: {str(e)}")
            self.viewer_title.setText(f'Line identification failed: {e}')

    def confirm_line_identification(self):
        """Confirm identified lines"""
        self.log_message("Line identification confirmed")
        # Could add plotting here to visualize identified lines

    def first_refinement(self):
        """First refinement step"""
        try:
            self._arc_preview_timer.stop()
            self._arc_stage = 2
            self._invalidate_arc(2)
            sciarc_ex = self.ARCdata[0]
            arcfile = self.get_arc_path()
            
            self.wave3, details = identify.fit_arclines(
                self.wave2,
                sciarc_ex.flux.value,
                arcfile,
                self.refine_width.value(),
                self.refine_threshold.value(),func=self.line_func.currentText(),deg=self.deg_val.value(), display=False, return_details=True
            )
            
            self._arc_views(self.wave3, sciarc_ex.flux.value, details, 'First wavelength refinement')
            self.refine2_btn.setEnabled(True)
            self.log_message("First refinement completed")
            
        except Exception as e:
            self.log_message(f"Error in first refinement: {str(e)}")
            self.viewer_title.setText(f'First refinement failed: {e}')

    def final_refinement(self):
        """Final refinement step"""
        try:
            self._arc_preview_timer.stop()
            self._arc_stage = 3
            self._invalidate_arc(3)
            sciarc_ex = self.ARCdata[0]
            arcfile = self.get_arc_path()
            
            self.final_wave, details = identify.fit_arclines(
                self.wave3,
                sciarc_ex.flux.value,
                arcfile,
                self.final_refine_width.value(),
                self.final_refine_threshold.value(),func=self.line_func.currentText(),deg=self.deg_val.value(), display=False, return_details=True
            )
            
            self._arc_views(self.final_wave, sciarc_ex.flux.value, details, 'Final wavelength solution')
            self.final_wave_details = details
            self.confirm_final_btn.setEnabled(True)
            self.log_message("Final refinement completed")
            
        except Exception as e:
            self.log_message(f"Error in final refinement: {str(e)}")
            self.viewer_title.setText(f'Final refinement failed: {e}')

    def confirm_final_solution(self):
        """Confirm final wavelength solution"""
        if not hasattr(self, 'final_wave'):
            return
        try:
            identify._plot_calibration(self.final_wave, self.ARCdata[0].flux.value, self.final_wave_details, display=False, filename=os.path.join(os.getcwd(), 'final_wavecalib.png'))
        except Exception as exc:
            self.log_message(f'Could not save wavelength diagnostic: {exc}')
        self.log_message("Final wavelength solution confirmed")

    def perform_spectral_calibration(self):
        """Keep wavelength-calibrated groups in memory until final export."""
        try:
            if not hasattr(self, 'final_wave'):
                raise ValueError("Please complete all refinement steps")
                
            # Initialize storage for calibrated spectra
            self.W_sci_groups = {}
            self.W_std_groups = {}
            self.final_groups = {}
            self.flux_calibrated = self.final_spec = None
            self.flux_cal_btn.setEnabled(False)
            
            # Process all science groups
            for group_id, sci_spec in self.combined_spectra['SCI'].items():
                calibrated = self._calibrate_single_spectrum(sci_spec)
                self.W_sci_groups[group_id] = calibrated
            
            # Process all standard groups
            for group_id, std_spec in self.combined_spectra['STD'].items():
                calibrated = self._calibrate_single_spectrum(std_spec)
                self.W_std_groups[group_id] = calibrated
            
            self.log_message("Spectral calibration completed for all groups (in memory)")
            
        except Exception as e:
            error_msg = f"Error in spectral calibration: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _calibrate_single_spectrum(self, spectrum):
        """Apply wavelength calibration to a single spectrum"""
        meta = dict(spectrum.meta)
        meta.update(wcal_rms=self.final_wave_details['rms'], wcal_degree=self.final_wave_details['degree'], wcal_nlines=len(self.final_wave_details['pixels']))
        return Spectrum1D(
            spectral_axis=self.final_wave * u.AA,
            flux=spectrum.flux,
            uncertainty=spectrum.uncertainty,
            meta=meta
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
        ext_layout.addWidget(self.create_info_label('extinction', 'Extinction File:'))
        
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
        std_layout.addWidget(self.create_info_label('standard_star', 'Standard Star:'))
        
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
        # Sensitivity preview is displayed in the persistent upper-right viewer.
        
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
        mode_layout.addWidget(self.create_info_label('sensitivity_mode', 'Sensitivity Mode:'))
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
        self.display_flux_final_check = QCheckBox()
        self.display_flux_final_check.setChecked(True)
        display_layout.addWidget(self.display_flux_final_check)
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
        redshift_layout.addWidget(self.create_info_label('redshift', 'Redshift:'))
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
        cut_layout = QHBoxLayout()
        self.output_cut = QCheckBox('Wavelength cutoff (observed Å)')
        cut_layout.addWidget(self.create_info_label('wavelength_cutoff', self.output_cut, checkbox=True))
        self.output_low, self.output_high = QDoubleSpinBox(), QDoubleSpinBox()
        for widget, value in ((self.output_low, 4500), (self.output_high, 9000)):
            widget.setRange(0, 1000000)
            widget.setDecimals(1)
            widget.setValue(value)
            cut_layout.addWidget(widget)
        out_layout.addLayout(cut_layout)
        bin_layout = QHBoxLayout()
        bin_layout.addWidget(self.create_info_label('output_bin', 'Bin size (pixels; 1 = off):'))
        self.output_bin = QSpinBox()
        self.output_bin.setRange(1, 10000)
        self.output_bin.setValue(5)
        bin_layout.addWidget(self.output_bin)
        out_layout.addLayout(bin_layout)
        preview_btn = QPushButton('Display Spectrum')
        preview_btn.clicked.connect(self.plot_final_spectrum)
        out_layout.addWidget(preview_btn)
        for widget in (self.redshift, self.output_low, self.output_high, self.output_bin):
            widget.valueChanged.connect(self._refresh_output_preview)
        for widget in (self.output_cut, self.display_final_check):
            widget.toggled.connect(self._refresh_output_preview)
        
        # Save final spectrum button
        save_button = QPushButton("Save Final FITS Tables (All Groups)")
        save_button.clicked.connect(self.save_final_spectrum)
        out_layout.addWidget(save_button)

        # Save final spectrum plot button
        save_button = QPushButton("Save Final Spectrum Plots (All Groups)")
        save_button.clicked.connect(self.save_final_plot)
        #save_button.clicked.connect(self.plot_final_spectrum)
        out_layout.addWidget(save_button)        
        # Plot area
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        # Final-spectrum preview uses the persistent upper-right viewer.
        
        layout.addWidget(out_group)
        # The QWidget already owns its layout.
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
            star = fluxcal.stdstr(std_file, display=False)
            
            # Get extinction data
            Xfile = fluxcal.read_obs_extinction(self.ext_file.text())
            
            # Apply airmass correction
            cal_am = std_spec.meta['airmass']
            CAL = fluxcal.airmass_cor(std_spec, cal_am, Xfile)

            # Create preview sensitivity function
            sens_func = fluxcal.standard_sensfunc(
                CAL,
                star,
                mode=self.sens_mode.currentText(),
                display=False  # We'll display our own preview
            )

            # Show sensitivity directly in the persistent viewer.
            self._sensitivity_views(CAL, sens_func, star)
            self.flux_cal_btn.setEnabled(True)
            self.log_message("Sensitivity curve preview generated")

        except Exception as e:
            self.log_message(f"Error in sensitivity preview: {str(e)}")
            QMessageBox.warning(self, "Preview Failed", str(e))

    def perform_flux_calibration(self):
        """Calibrate each science group in memory; export only in Output."""
        try:
            self.final_groups = {}
            self.flux_calibrated = self.final_spec = None
            if not self.W_sci_groups or not self.W_std_groups:
                raise ValueError('Complete wavelength calibration for science and standard groups first.')
            mapping = {}
            for gid in self.W_sci_groups:
                if gid in self.W_std_groups:
                    mapping[gid] = gid
                elif len(self.W_std_groups) == 1:
                    mapping[gid] = next(iter(self.W_std_groups))
                else:
                    raise ValueError(f'No matching standard for science group {gid}; assign matching Combine IDs.')
            results, records = {}, []
            for gid, std_gid in mapping.items():
                sci_spec, std_spec = self.W_sci_groups[gid], self.W_std_groups[std_gid]
                spec, sensitivity, diagnostics = self._calibrate_flux(sci_spec, std_spec)
                results[gid] = dict(spectrum=spec, sensitivity=sensitivity, meta=dict(sci_spec.meta), std_meta=dict(std_spec.meta), std_group=std_gid, standard=self.std_file_display.text(), mode=self.sens_mode.currentText(), extinction=self.ext_file.text())
                self.log_message(f'Calibrated science group {gid} using standard group {std_gid}')
                if self.display_sens_check.isChecked():
                    records.extend(dict(record, title=f'Group {gid} | '+record['title']) for record in diagnostics)
                if self.display_flux_final_check.isChecked():
                    records.append(dict(title=f'Flux-calibrated science | group {gid}', xlabel='Wavelength (Å)', ylabel=f'Flux ({spec.flux.unit})', series=[(spec.spectral_axis.value, spec.flux.value, f'Group {gid}', '#50c7dd', False)]))
            self.final_groups = results
            self.flux_calibrated = self.final_spec = next(iter(results.values()))['spectrum']
            if records:
                self._diagnostic_views(records)
            self.log_message('Flux calibration completed for all groups. Save final FITS tables from Output.')
            
        except Exception as e:
            error_msg = f"Error in flux calibration: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
    def _calibrate_flux(self, sci_spec, std_spec):
        """Perform flux calibration for a single science-standard pair"""
        try:
            # Get extinction data
            extinction_file = self.ext_file.text()
            if not os.path.exists(extinction_file):
                raise FileNotFoundError(f"Extinction file not found: {extinction_file}")
            
            Xfile = fluxcal.read_obs_extinction(extinction_file)
            
            # Apply airmass correction
            sci_am = sci_spec.meta['airmass']
            std_am = std_spec.meta['airmass']
            
            CAL = fluxcal.airmass_cor(std_spec, std_am, Xfile)
            SCI = fluxcal.airmass_cor(sci_spec, sci_am, Xfile)
            
            # Get standard star data
            std_file = self.std_file_display.text()
            if not std_file:
                raise ValueError("No standard star file selected")
            
            star = fluxcal.stdstr(std_file, display=False)
            
            # Create sensitivity function
            sens_func = fluxcal.standard_sensfunc(
                CAL,
                star,
                mode=self.sens_mode.currentText(),
                display=False
            )
            
            # Apply sensitivity function
            flux_calibrated = fluxcal.apply_sensfunc(SCI, sens_func)
            return flux_calibrated, sens_func, self._sensitivity_views(CAL, sens_func, star, show=False)
        
        except Exception as e:
            self.log_message(f"Error in flux calibration: {str(e)}")
            raise
    
    def _sensitivity_views(self, observed, sensitivity, star, show=True):
        wave = observed.spectral_axis.value
        raw = observed.flux.value
        calibrated = fluxcal.apply_sensfunc(observed, sensitivity).flux.value
        def normalize(values):
            scale = np.nanmedian(np.abs(values[np.isfinite(values)]))
            return values / scale if np.isfinite(scale) and scale > 0 else values * np.nan
        records = [
            dict(title='Observed × sensitivity — standard star comparison', xlabel='Wavelength (Å)', ylabel='Flux (independently normalized)', series=[(wave, normalize(raw), 'Observed', '#50c7dd', False), (wave, normalize(calibrated), 'Observed × sensitivity', '#f8ad62', False)]),
            dict(title='Calibrated standard vs reference', xlabel='Wavelength (Å)', ylabel=f'Flux ({star["flux"].unit})', series=[(wave, calibrated, 'Observed × sensitivity', '#50c7dd', False), (np.asarray(star['wave']), np.asarray(star['flux']), 'Reference standard', '#f8ad62', True)]),
            dict(title='Sensitivity function', xlabel='Wavelength (Å)', ylabel=f'Sensitivity ({sensitivity.flux.unit})', series=[(sensitivity.spectral_axis.value, sensitivity.flux.value, 'Sensitivity', '#50c7dd', False)])]
        if show:
            self._diagnostic_views(records)
        return records

    def _load_calibrated_spectrum(self, filename):
        """Load a wavelength-calibrated spectrum from FITS"""
        with fits.open(filename) as hdul:
            flux = hdul[0].data
            wave = hdul['WAVELENGTH'].data
            uncertainty = hdul['UNCERTAINTY'].data if 'UNCERTAINTY' in hdul else None
            
        return Spectrum1D(
            flux=flux * u.Unit('count'),
            spectral_axis=wave * u.AA,
            uncertainty=StdDevUncertainty(uncertainty) if uncertainty is not None else None
        )

    def _save_flux_calibrated(self, spectrum, filename):
        """Save flux-calibrated spectrum to FITS file"""
        try:
            # Create primary HDU with flux
            primary_hdu = fits.PrimaryHDU(spectrum.flux.value)
            primary_hdu.header['BUNIT'] = 'erg/s/cm^2/Angstrom'
            
            # Create HDU list
            hdus = [primary_hdu]
            
            # Add uncertainty if available
            if spectrum.uncertainty is not None:
                uncert_hdu = fits.ImageHDU(spectrum.uncertainty.array, name='UNCERTAINTY')
                hdus.append(uncert_hdu)
            
            # Add wavelength array
            wave_hdu = fits.ImageHDU(spectrum.spectral_axis.value, name='WAVELENGTH')
            wave_hdu.header['BUNIT'] = 'Angstrom'
            hdus.append(wave_hdu)
            
            # Add calibration metadata
            primary_hdu.header['CALSRC'] = self.std_file_display.text()
            primary_hdu.header['CALMODE'] = self.sens_mode.currentText()
            primary_hdu.header['CRVAL1'] = spectrum.spectral_axis[0].value
            primary_hdu.header['CDELT1'] = np.mean(np.diff(spectrum.spectral_axis.value))
            
            # Write to file
            hdulist = fits.HDUList(hdus)
            hdulist.writeto(filename, overwrite=True)
            self.log_message(f"Saved flux-calibrated spectrum: {filename}")
            
        except Exception as e:
            self.log_message(f"Error saving flux-calibrated spectrum: {str(e)}")
            raise
    def save_final_plot(self):
        """ save plot of the final spectrum"""
        try:
            if not self.final_groups:
                raise ValueError("no final spectrum available")

            output_name = self.output_name.text()
            Redshift = self.redshift.value()
            print(f" the redshift {Redshift}")
            if not output_name:
                output_name = self.sciname
            #plot_final_spectrum()
            for gid, result in self.final_groups.items():
                final.save_plot(result['spectrum'], name=f'{output_name}_group{gid}', wbin=self.output_cut.isChecked(), lw=self.output_low.value(), uw=self.output_high.value(), redshift=Redshift, binsize=self.output_bin.value(), display=False)

            self.log_message(f" save the plot of final spectrum")

        except Exception as e:
            error_msg = f"Error saving spectrum plot: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to save spectrum plot: {str(e)}")
            
    

    def save_final_spectrum(self):
        """Export one primary-header + binary-table FITS file per science group."""
        try:
            if not self.final_groups:
                raise ValueError("No final spectrum to save")
            
            output_name = self.output_name.text()
            if not output_name:
                output_name = self.sciname
            
            files = []
            for gid, result in self.final_groups.items():
                filename = f'{output_name}_group{gid}_final.fits'
                self._write_final_group(filename, gid, result)
                files.append(filename)
                self.log_message(f'Saved final spectrum: {filename}')
            QMessageBox.information(self, 'Success', '\n'.join(files))
            
        except Exception as e:
            error_msg = f"Error saving spectrum: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to save spectrum: {str(e)}")

    def _write_final_group(self, filename, gid, result):
        spec, sens, meta = result['spectrum'], result['sensitivity'], result['meta']
        wave = spec.spectral_axis.to_value(u.AA)
        flux = spec.flux.value
        if spec.uncertainty is None:
            raise ValueError(f'Group {gid} has no propagated flux uncertainty.')
        err = spec.uncertainty.represent_as(StdDevUncertainty).quantity.to_value(spec.flux.unit)
        sw = sens.spectral_axis.to_value(u.AA)
        order = np.argsort(sw)
        response = np.interp(wave, sw[order], sens.flux.value[order])
        headers = meta.get('headers', [])
        header = fits.Header(headers[0]) if headers else fits.Header()
        # The primary HDU is metadata only; raw image WCS must not describe the table.
        for key in list(header):
            if key in ('BUNIT', 'CHECKSUM', 'DATASUM', 'BSCALE', 'BZERO', 'BLANK', 'WCSAXES') or key.startswith(('CRPIX', 'CRVAL', 'CDELT', 'CTYPE', 'CUNIT', 'CROTA', 'CD1_', 'CD2_', 'PC1_', 'PC2_')):
                del header[key]
        primary = fits.PrimaryHDU(header=header)
        h = primary.header
        h['COMBGRP'] = (str(gid), 'Science combination group ID')
        h['STDGROUP'] = (str(result['std_group']), 'Standard combination group ID')
        h['COMBMETH'] = meta['combine_method']
        h['NCOMBINE'] = len(meta['input_files'])
        h['AIRMASS'] = (meta['airmass'], 'Mean science group airmass')
        h['STDAM'] = (result['std_meta']['airmass'], 'Mean standard group airmass')
        h['CALSRC'] = os.path.basename(result['standard'])
        h['CALMODE'] = result['mode']
        h['EXTFILE'] = os.path.basename(result['extinction'])
        h['WCALTYPE'] = 'Final'
        h['WCALFUNC'] = 'Chebyshev'
        h['WCALDEG'] = meta['wcal_degree']
        h['WCALRMS'] = (meta['wcal_rms'], 'Wavelength-fit RMS in Angstrom')
        h['WCALNPTS'] = meta['wcal_nlines']
        h['REDSHIFT'] = (self.redshift.value(), 'Preview redshift; stored wavelength observed')
        for i, path in enumerate(meta['input_files'], 1):
            h[f'INFIL{i:03d}'] = os.path.basename(path)
        for i, path in enumerate(result['std_meta']['input_files'], 1):
            h[f'STFIL{i:03d}'] = os.path.basename(path)
        h.add_history('Flux calibrated using group mean airmass and interpolated standard sensitivity.')
        h.add_history('Table contains native observed wavelengths and unbinned flux; preview settings do not modify data.')
        columns = [fits.Column(name='WAVELENGTH', format='D', unit='Angstrom', array=wave),
                   fits.Column(name='FLUX', format='D', unit=spec.flux.unit.to_string('fits'), array=flux),
                   fits.Column(name='FLUX_ERROR', format='D', unit=spec.flux.unit.to_string('fits'), array=err),
                   fits.Column(name='SENSITIVITY', format='D', unit=sens.flux.unit.to_string('fits'), array=response)]
        table = fits.BinTableHDU.from_columns(columns, name='SPECTRUM')
        fits.HDUList([primary, table]).writeto(filename, overwrite=True, checksum=True)


    def _refresh_output_preview(self, *_):
        if self.tabs.tabText(self.tabs.currentIndex()) == 'Output':
            self.plot_final_spectrum()

    def plot_final_spectrum(self):
        """Show the final calibrated spectrum in the upper-right viewer."""
        if self.final_groups:
            if not self.display_final_check.isChecked():
                self.viewer_plot.clear()
                self._aperture_views = []
                self._aperture_view_index = -1
                self._update_view_navigation()
                self.viewer_title.setText('Final spectrum hidden')
                return
            previous = self._aperture_views[self._aperture_view_index].get('group_id') if 0 <= self._aperture_view_index < len(self._aperture_views) else None
            records = []
            try:
                for gid, result in self.final_groups.items():
                    spec = result['spectrum']
                    wave, flux, bw, bf = final.preview_arrays(spec, self.output_cut.isChecked(), self.output_low.value(), self.output_high.value(), self.redshift.value(), self.output_bin.value())
                    series = [(wave, flux, 'Observed spectrum', '#50c7dd', False)]
                    if self.output_bin.value() > 1:
                        series.append((bw, bf, f'Binned ({self.output_bin.value()} px)', '#f8ad62', False))
                    lines = [(w, label, '#80c980') for w, label in ((4861, 'Hβ'), (5007, '[O III]'), (6563, 'Hα')) if np.nanmin(wave) <= w <= np.nanmax(wave)] if self.redshift.value() else []
                    records.append(dict(group_id=gid, title=f'Final spectrum | Group {gid} — z = {self.redshift.value():.3f}', xlabel='Rest wavelength (Å)' if self.redshift.value() else 'Observed wavelength (Å)', ylabel=f'Flux ({spec.flux.unit})', series=series, lines=lines))
                self._diagnostic_views(records)
                if previous in self.final_groups:
                    self._show_aperture_record(list(self.final_groups).index(previous))
            except ValueError as e:
                self.viewer_title.setText(str(e))
        else:
            self.viewer_plot.clear()
            self._aperture_views = []
            self._aperture_view_index = -1
            self._update_view_navigation()
            self.viewer_title.setText('No final calibrated groups available')

#******************************************************************************************************#

if __name__ == "__main__":
    #resource_path = "disk1/ADFOSC_pp/T_script/resources"
    app = QApplication(sys.argv)
    gui = SpectroscopicReductionGUI()
    gui.show()
    sys.exit(app.exec_())
