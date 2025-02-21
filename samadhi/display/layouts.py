# -*- coding:utf-8 -*-
#!/usr/bin/python3
from PyQt6 import QtWidgets
from .dancingdots import OpenGLDancingDots
import time

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
        settingslayout.addWidget(hide_settings_btn, 0, 0, 1, 2)
        no_settingslayout.addWidget(show_settings_btn, 0, 0, 1, 1)
        settingslayout.addItem(QtWidgets.QSpacerItem(10, 20,
                                                       QtWidgets.QSizePolicy.Policy.Minimum,
                                                       QtWidgets.QSizePolicy.Policy.Expanding),
                                 1, 0, 1, 1)
        no_settingslayout.addItem(QtWidgets.QSpacerItem(10, 20,
                                                           QtWidgets.QSizePolicy.Policy.Minimum,
                                                           QtWidgets.QSizePolicy.Policy.Expanding),
                                    1, 0, 1, 1)

        # controls for repr. frequencies
        spin_freq0 = QtWidgets.QSpinBox()
        spin_freq0.setRange(1, 50)
        settingslayout.addWidget(QtWidgets.QLabel("Freq. 1"), 2, 0, 1, 1)
        settingslayout.addWidget(spin_freq0, 2, 1, 1, 1)
        self._settings['freq0'] = spin_freq0
        spin_freq1 = QtWidgets.QSpinBox()
        spin_freq1.setRange(1, 50)
        settingslayout.addWidget(QtWidgets.QLabel("Freq. 2"), 3, 0, 1, 1)
        settingslayout.addWidget(spin_freq1, 3, 1, 1, 1)
        self._settings['freq1'] = spin_freq1
        spin_freq2 = QtWidgets.QSpinBox()
        spin_freq2.setRange(1, 50)
        settingslayout.addWidget(QtWidgets.QLabel("Freq. 3"), 4, 0, 1, 1)
        settingslayout.addWidget(spin_freq2, 4, 1, 1, 1)
        self._settings['freq2'] = spin_freq2
        spin_freq3 = QtWidgets.QSpinBox()
        spin_freq3.setRange(1, 50)
        settingslayout.addWidget(QtWidgets.QLabel("Freq. 4"), 5, 0, 1, 1)
        settingslayout.addWidget(spin_freq3, 5, 1, 1, 1)
        self._settings['freq3'] = spin_freq3
        spin_freq4 = QtWidgets.QSpinBox()
        spin_freq4.setRange(1, 50)
        settingslayout.addWidget(QtWidgets.QLabel("Freq. 5"), 6, 0, 1, 1)
        settingslayout.addWidget(spin_freq4, 6, 1, 1, 1)
        self._settings['freq4'] = spin_freq4
        self.addWidget(self._no_settings_wdg, 0, 0, 1, 1)

        # add widget
        self._ddots_wdg = OpenGLDancingDots(get_data, self.toggle_fullscreen_dancing_dots)
        self.addWidget(self._ddots_wdg, 0, 1, 1, 1)
        self.setColumnStretch(0, 0)
        self.setColumnStretch(1, 1)

        # add default settings
        self.set_settings({
            'freq0': 1,
            'freq1': 2,
            'freq2': 3,
            'freq3': 5,
            'freq4': 8,
        })

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
            settings[name] = widget.getValue()
        return settings

    def set_settings(self, settings):
        """
        :param settings: A dictionary with values to set
        """
        for name, value in settings.items():
            self._settings[name].setValue(value)

    def toggle_fullscreen_dancing_dots(self, fullscreen):
        if not fullscreen:
            self.addWidget(self._ddots_wdg, 0, 1, 1, 1)
        if fullscreen:
            self.removeWidget(self._ddots_wdg)
            self._ddots_wdg.setParent(None)
            self._ddots_wdg.showFullScreen()

