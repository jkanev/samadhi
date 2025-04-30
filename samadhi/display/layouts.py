# -*- coding:utf-8 -*-
#!/usr/bin/python3
from PyQt6.QtWidgets import QPushButton
from PyQt6 import QtWidgets, QtGui, QtCore
from .dancingdots import OpenGLDancingDots
from .radiantripples import OpenGLRadiantRipples
import time

class ColourButton(QPushButton):
    """
    Helper class. A button with a connected QColor dialog, that has a signal .valueChanged and a .value() function
    that returns three ints (the rgb value of the colour)
    """

    _value = (0, 0, 0)     # the current colour, a tuple of three ints (rgb 0/255)
    valueChanged = QtCore.pyqtSignal((int, int, int), name='valueChanged')   # signal when user selects colour

    def value(self):
        """
        :return: The current colour
        """
        return self._value

    def setValue(self, colour):
        """ Sets the current colour and changes the button background
        :param colour: The colour to set, a tuple of three ints
        """
        self._value = colour
        self.setStyleSheet('background-color: #{:02x}{:02x}{:02x}'.format(*self._value))

    def __init__(self):
        """ Connecting the the base class' signal
        """
        super().__init__()
        self.clicked.connect(self.select_colour)

    def select_colour(self):
        """ Callback when clicked; calls the colour dialog and stores the selected value
        :return: void
        """
        self.setValue(QtWidgets.QColorDialog.getColor(initial=QtGui.QColor(*self._value)).getRgb()[:3])
        self.valueChanged.emit(self._value[0], self._value[1], self._value[2])    # seperately, tuples aren't supported



class DancingDotsLayout(QtWidgets.QGridLayout):

    _showing_ddots = False
    _ddots_wdg = None       # opengl widget with the dots
    _settings_wdg = None     # widget with all settings
    _no_settings_wdg = None    # empty settins with a "show" button
    _settings = {}     # dictionary with settings widgets: 'string' → widget

    def __init__(self, parent, get_data):
        super().__init__(parent)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        parent.setSizePolicy(sizePolicy)

        # add setting layout
        self._settings_wdg = QtWidgets.QWidget()
        self._no_settings_wdg = QtWidgets.QWidget()
        settingslayout = QtWidgets.QGridLayout(self._settings_wdg)
        no_settingslayout =  QtWidgets.QGridLayout(self._no_settings_wdg)

        # show/hide buttons
        show_settings_btn = QtWidgets.QPushButton("☰")
        show_settings_btn.setStyleSheet("padding: 0.3em; font-size: 12pt;")
        show_settings_btn.clicked.connect(self.show_settings)
        hide_settings_btn = QtWidgets.QPushButton("Hide Settings")
        hide_settings_btn.clicked.connect(self.hide_settings)
        settingslayout.addWidget(hide_settings_btn, 0, 0, 1, 4)
        no_settingslayout.addWidget(show_settings_btn, 0, 0, 1, 1)
        settingslayout.addItem(QtWidgets.QSpacerItem(10, 20,
                                                       QtWidgets.QSizePolicy.Policy.Minimum,
                                                       QtWidgets.QSizePolicy.Policy.Expanding),
                                 1, 0, 1, 4)
        no_settingslayout.addItem(QtWidgets.QSpacerItem(10, 20,
                                                           QtWidgets.QSizePolicy.Policy.Minimum,
                                                           QtWidgets.QSizePolicy.Policy.Expanding),
                                    1, 0, 1, 1)

        # controls for time smoothing and power
        # spacer
        settingslayout.addItem(QtWidgets.QSpacerItem(10, 20,
                                                     QtWidgets.QSizePolicy.Policy.Minimum,
                                                     QtWidgets.QSizePolicy.Policy.Expanding),
                               2, 0, 1, 4)

        # label
        settingslayout.addWidget(QtWidgets.QLabel("General"), 3, 0, 1, 4)

        # circular frequency
        spin_timesmoothing = QtWidgets.QDoubleSpinBox()
        spin_timesmoothing.setRange(0.0, 1.0)
        spin_timesmoothing.setSingleStep(0.02)
        spin_timesmoothing.valueChanged.connect(self.update_ddots_display)
        settingslayout.addWidget(spin_timesmoothing, 4, 0, 1, 1)
        settingslayout.addWidget(QtWidgets.QLabel("Time smoothing"), 4, 1, 1, 1)
        self._settings['timesmoothing'] = spin_timesmoothing

        # rotation
        spin_power = QtWidgets.QDoubleSpinBox()
        spin_power.setRange(0.0, 5.0)
        spin_power.setSingleStep(0.5)
        spin_power.valueChanged.connect(self.update_ddots_display)
        settingslayout.addWidget(spin_power, 4, 2, 1, 1)
        settingslayout.addWidget(QtWidgets.QLabel("Power/Softmax"), 4, 3, 1, 1)
        self._settings['power'] = spin_power

        # controls for repr. frequencies
        headings = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        for n in range(0, 5):

            # spacer
            settingslayout.addItem(QtWidgets.QSpacerItem(10, 20,
                                                         QtWidgets.QSizePolicy.Policy.Minimum,
                                                         QtWidgets.QSizePolicy.Policy.Expanding),
                                   4*n+5, 0, 1, 4)

            # label
            settingslayout.addWidget(QtWidgets.QLabel(headings[n]), 4*n+6, 0, 1, 4)

            # circular frequency
            spin_freq = QtWidgets.QSpinBox()
            spin_freq.setRange(0, 50)
            spin_freq.valueChanged.connect(self.update_ddots_display)
            settingslayout.addWidget(spin_freq, 4*n+7, 0, 1, 1)
            settingslayout.addWidget(QtWidgets.QLabel("Circular frequency"), 4*n+7, 1, 1, 1)
            self._settings['freq{}'.format(n)] = spin_freq

            # rotation
            spin_rotation = QtWidgets.QDoubleSpinBox()
            spin_rotation.setRange(-2.0, 2.0)
            spin_rotation.setSingleStep(0.1)
            spin_rotation.valueChanged.connect(self.update_ddots_display)
            settingslayout.addWidget(spin_rotation, 4*n+7, 2, 1, 1)
            settingslayout.addWidget(QtWidgets.QLabel("Rotation"), 4*n+7, 3, 1, 1)
            self._settings['rotation{}'.format(n)] = spin_rotation

            # inside colour
            cbutton_incolour = ColourButton()
            cbutton_incolour.valueChanged.connect(self.update_ddots_display)
            settingslayout.addWidget(cbutton_incolour, 4*n+8, 0, 1, 1)
            settingslayout.addWidget(QtWidgets.QLabel("Inside colour"), 4*n+8, 1, 1, 1)
            self._settings['incolour{}'.format(n)] = cbutton_incolour

            # outside colour
            cbutton_outcolour = ColourButton()
            cbutton_outcolour.valueChanged.connect(self.update_ddots_display)
            settingslayout.addWidget(cbutton_outcolour, 4*n+8, 2, 1, 1)
            settingslayout.addWidget(QtWidgets.QLabel("Outside colour"), 4*n+8, 3, 1, 1)
            self._settings['outcolour{}'.format(n)] = cbutton_outcolour

        # add default settings
        settings = {
            'timesmoothing': 0.95, 'power': 2.0,
            'freq0':  2, 'rotation0':  0.75, 'incolour0': (255,   0,   0), 'outcolour0': ( 0, 63, 63),
            'freq1':  3, 'rotation1': -1.0, 'incolour1': (255, 127,   0), 'outcolour1': ( 0, 31, 63),
            'freq2':  4, 'rotation2':  0.5, 'incolour2': (255, 255,   0), 'outcolour2': ( 0,  0, 63),
            'freq3':  8, 'rotation3': -0.5, 'incolour3': (  0, 255,   0), 'outcolour3': (63,  0, 63),
            'freq4': 13, 'rotation4':  0.25, 'incolour4': (  0, 255, 127), 'outcolour4': (63,  0, 31),
        }
        self.set_settings(settings)

        # add widgets to main layout
        self.addWidget(self._no_settings_wdg, 0, 0, 1, 1)
        self._ddots_wdg = OpenGLDancingDots(get_data, self.toggle_fullscreen_dancing_dots, settings)
        self.addWidget(self._ddots_wdg, 0, 1, 1, 1)
        self.setColumnStretch(0, 0)
        self.setColumnStretch(1, 1)

        # start display thread
        time.sleep(1)
        self._showing_ddots = True
        self._ddots_wdg.start()

    def show_settings(self):
        """ Shows the settings pane in the layout
        :return: void
        """
        print("showing settings")
        self.setColumnStretch(0, 1)
        self.setColumnStretch(1, 5)
        self._no_settings_wdg.hide()
        self.removeWidget(self._no_settings_wdg)
        self._settings_wdg.show()
        self.addWidget(self._settings_wdg, 0, 0, 1, 1)

    def hide_settings(self):
        """ Hides the settings pane in the layout
        :return: void
        """
        print("hiding settings")
        self.setColumnStretch(0, 0)
        self.setColumnStretch(1, 1)
        self._settings_wdg.hide()
        self.removeWidget(self._settings_wdg)
        self._no_settings_wdg.show()
        self.addWidget(self._no_settings_wdg, 0, 0, 1, 1)

    def get_settings(self):
        """
        :return: A dictionary with setting names and the values from the GUI
        """
        settings = {}
        for name, widget in self._settings.items():
            settings[name] = widget.value()
        return settings

    def set_settings(self, settings):
        """
        :param settings: A dictionary with values to set
        """
        for name, value in settings.items():
            self._settings[name].setValue(value)

    def update_ddots_display(self):
        """ Set parameters in ddots display
        :return: void
        """
        if self._showing_ddots:
            self._ddots_wdg.set_parameters(self.get_settings())

    def toggle_fullscreen_dancing_dots(self, fullscreen):
        if not fullscreen:
            self.addWidget(self._ddots_wdg, 0, 1, 1, 1)
        if fullscreen:
            self.removeWidget(self._ddots_wdg)
            self._ddots_wdg.setParent(None)
            self._ddots_wdg.showFullScreen()



class RadiantRipplesLayout(QtWidgets.QGridLayout):

    _showing_rripples = False
    _rripples_wdg = None       # opengl widget with the dots
    _settings_wdg = None     # widget with all settings
    _no_settings_wdg = None    # empty settins with a "show" button
    _settings = {}     # dictionary with settings widgets: 'string' → widget

    def __init__(self, parent, get_data, get_2d_layout):
        super().__init__(parent)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        parent.setSizePolicy(sizePolicy)

        # add setting layout
        self._settings_wdg = QtWidgets.QWidget()
        self._no_settings_wdg = QtWidgets.QWidget()
        settingslayout = QtWidgets.QGridLayout(self._settings_wdg)
        no_settingslayout =  QtWidgets.QGridLayout(self._no_settings_wdg)

        # show/hide buttons
        show_settings_btn = QtWidgets.QPushButton("☰")
        show_settings_btn.setStyleSheet("padding: 0.3em; font-size: 12pt;")
        show_settings_btn.clicked.connect(self.show_settings)
        hide_settings_btn = QtWidgets.QPushButton("Hide Settings")
        hide_settings_btn.clicked.connect(self.hide_settings)
        settingslayout.addWidget(hide_settings_btn, 0, 0, 1, 4)
        no_settingslayout.addWidget(show_settings_btn, 0, 0, 1, 1)
        settingslayout.addItem(QtWidgets.QSpacerItem(10, 20,
                                                       QtWidgets.QSizePolicy.Policy.Minimum,
                                                       QtWidgets.QSizePolicy.Policy.Expanding),
                                 1, 0, 1, 4)
        no_settingslayout.addItem(QtWidgets.QSpacerItem(10, 20,
                                                           QtWidgets.QSizePolicy.Policy.Minimum,
                                                           QtWidgets.QSizePolicy.Policy.Expanding),
                                    1, 0, 1, 1)


        # wave speed
        spin_wavespeed = QtWidgets.QDoubleSpinBox()
        spin_wavespeed.setRange(0.0, 20.0)
        spin_wavespeed.setSingleStep(0.5)
        spin_wavespeed.valueChanged.connect(self.update_rripples_display)
        settingslayout.addWidget(spin_wavespeed, 2, 0, 1, 1)
        settingslayout.addWidget(QtWidgets.QLabel("Wave Speed"), 2, 1, 1, 1)
        self._settings['wavespeed'] = spin_wavespeed

        # wave frequency
        spin_wavefrequency = QtWidgets.QDoubleSpinBox()
        spin_wavefrequency.setRange(0.0, 20.0)
        spin_wavefrequency.setSingleStep(0.5)
        spin_wavefrequency.valueChanged.connect(self.update_rripples_display)
        settingslayout.addWidget(spin_wavefrequency, 2, 2, 1, 1)
        settingslayout.addWidget(QtWidgets.QLabel("Wave Frequency"), 2, 3, 1, 1)
        self._settings['wavefrequency'] = spin_wavefrequency

        # add default settings
        settings = {
            'wavespeed': 10.0, 'wavefrequency': 10.0,
        }
        self.set_settings(settings)

        # add widgets to main layout
        self.addWidget(self._no_settings_wdg, 0, 0, 1, 1)
        self._rripples_wdg = OpenGLRadiantRipples(get_data, get_2d_layout, self.toggle_fullscreen_radiant_ripples, settings)
        self.addWidget(self._rripples_wdg, 0, 1, 1, 1)
        self.setColumnStretch(0, 0)
        self.setColumnStretch(1, 1)

        # start display thread
        time.sleep(1)
        self._showing_rripples = True
        self._rripples_wdg.start()

    def show_settings(self):
        """ Shows the settings pane in the layout
        :return: void
        """
        print("showing settings")
        self.setColumnStretch(0, 1)
        self.setColumnStretch(1, 5)
        self._no_settings_wdg.hide()
        self.removeWidget(self._no_settings_wdg)
        self._settings_wdg.show()
        self.addWidget(self._settings_wdg, 0, 0, 1, 1)

    def hide_settings(self):
        """ Hides the settings pane in the layout
        :return: void
        """
        print("hiding settings")
        self.setColumnStretch(0, 0)
        self.setColumnStretch(1, 1)
        self._settings_wdg.hide()
        self.removeWidget(self._settings_wdg)
        self._no_settings_wdg.show()
        self.addWidget(self._no_settings_wdg, 0, 0, 1, 1)

    def get_settings(self):
        """
        :return: A dictionary with setting names and the values from the GUI
        """
        settings = {}
        for name, widget in self._settings.items():
            settings[name] = widget.value()
        return settings

    def set_settings(self, settings):
        """
        :param settings: A dictionary with values to set
        """
        for name, value in settings.items():
            self._settings[name].setValue(value)

    def update_rripples_display(self):
        """ Set parameters in rripples display
        :return: void
        """
        if self._showing_rripples:
            self._rripples_wdg.set_parameters(self.get_settings())

    def toggle_fullscreen_radiant_ripples(self, fullscreen):
        if not fullscreen:
            self.addWidget(self._rripples_wdg, 0, 1, 1, 1)
        if fullscreen:
            self.removeWidget(self._rripples_wdg)
            self._rripples_wdg.setParent(None)
            self._rripples_wdg.showFullScreen()

