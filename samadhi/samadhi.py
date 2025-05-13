# -*- coding:utf-8 -*-
#!/usr/bin/python3

import sys

from .mainwindow import *
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QSurfaceFormat
import threading
import numpy as np
import time
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.lsl import resolve_streams
from matplotlib import use as mpl_use
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from .display.layouts import DancingDotsLayout, RadiantRipplesLayout
from mne import channels as chn
import math
import re

mpl_use("QtAgg")


class Mind:
    """
    Implements a complete data and calculation set of a single human.
    """

    # general
    _name = ""           # the person's name

    # data streaming related
    _streaming = False   # whether we're streaming currently
    _resolving = True    # whether we're looking for LSL streams
    _showing_eegpsd = False     # whether we're showing the eeg/psd tab
    _data_seconds = 2.0  # how much data do we have in the _eeg_data array
    _sampling_rate = 0   # sampling rate of eeg data
    _samples = 0         # seconds times sampling rate
    _fft_resolution = 0  # resolution (distance of one FFT bin to the next)
    _fft_max = 0         # max index for fft (upper border of highest frequency band)
    _fft_running_mean = None
    _channels = 0        # number of channels in the data
    _2d_layout = []      # x,y coordinates of channels, will be written after connecting to the stream
    _ch_names = []       # channel names
    _history_length = 600.0   # length of history buffer in seconds
    _eeg_data = []       # pointer to the buffer that has just been filled, either data_a or data_b
    _eeg_lock = threading.Lock()    # lock for eeg data
    _eeg_times = []      # buffer containing eeg time stamps
    _fft_data = []       # buffer containing the fft
    _fft_lock = threading.Lock()    # lock for fft data
    _fft_freqs = []      # buffer containing fft frequencies
    _bnd_data = []       # frequency band data: band frequencies
    _bnd_lock = threading.Lock()    # lock for bnd data
    _hst_data = []       # frequency band data: ring buffer that is constantly rooled
    _hst_lock = threading.Lock()    # lock for hst data
    _sqr_data = []       # mean square activity
    _sqr_lock = threading.Lock()
    _eeg_stream = None   # the lsl eeg input stream inlet, if in eeg mode

    # normalisation
    _bnd_smoothing = 0.95   #
    _bnd_max = []       # maximum values of bands, calculated from data d(t) by x(t+1) = max(0.99*x(t), d(t))
    _bnd_min = []       # minimum values of bands, calculated from data d(t) by x(t+1) = min(1.01*x(t), d(t))
    _bnd_mid = []       # the middle between max and min

    # main mind controls
    _combobox_streamname = False
    _lineedit_name = False
    _checkbox_connect_lsl = False
    _checkbox_display_eegpsd = False
    _checkbox_visualisation_ddots = False
    _checkbox_visualisation_rripples = False
    _parent_tabwidget = False
    _available_streams = []

    # info labels
    _gui_lock = threading.Lock()
    _bnd_label = False
    _bnd_info = ""
    _lsl_label = False
    _lsl_info = ""
    _eeg_label = False
    _eeg_info = ""

    # eeg display research controls → put into new class soon
    _eegpsd_layout = False
    _eegpsd_tab = False
    _eeg_axes = False
    _eeg_canvas = False
    _eeg_channel_height = 70e-6
    _sqr_axes = False
    _sqr_canvas = False
    _sqr_channel_height = 70e-6
    _fft_axes = False
    _fft_canvas = False
    _fft_channel_height = 50e-4
    _bnd_axes = False
    _bnd_canvas = False
    _bnd_height = 110e-4
    _hst_axes = False
    _hst_canvas = False
    _hst_height = 110e-4

    # fancy displays
    _ddots_tab = False     # dancing dots
    _rripples_tab = False  # radiant ripples

    def __init__(self, combobox_streamname, lineedit_name, checkbox_connect,
                       checkbox_display_eegpsd, checkbox_visualisation_ddots, checkbox_visualisation_rripples,
                       lsl_info, eeg_info, bnd_info,
                       parent_tabwidget):
        self._combobox_streamname = combobox_streamname
        self._lineedit_name = lineedit_name
        self._checkbox_connect_lsl = checkbox_connect
        self._checkbox_display_eegpsd = checkbox_display_eegpsd
        self._checkbox_visualisation_ddots = checkbox_visualisation_ddots
        self._checkbox_visualisation_rripples = checkbox_visualisation_rripples
        self._lsl_label = lsl_info
        self._eeg_label = eeg_info
        self._bnd_label = bnd_info
        self._parent_tabwidget = parent_tabwidget

        # Connect slot eeg stream
        self._checkbox_connect_lsl.clicked.connect(self._connect_eeg_stream)
        self._checkbox_display_eegpsd.clicked.connect(self._create_eegpsd_tab)
        self._checkbox_visualisation_ddots.clicked.connect(self._create_dancing_dots_tab)
        self._checkbox_visualisation_rripples.clicked.connect(self._create_radiant_ripples_tab)

        # start stream searching thread (extra, otherwise it blocks to GUI)
        thlsl = threading.Thread(target=self._find_sources)
        thlsl.start()

        # start gui update thread, filling stream selection box
        self._gui_timer = QtCore.QTimer()
        self._gui_timer.timeout.connect(self._update_gui)
        self._gui_timer.start(300)

    def __del__(self):
        self._connect_eeg_stream(False, restart_resolve=False)

    def _reset(self):
        """
        Resets the values after a stream disconnect, also removes the tab
        :return: void
        """

        # set streaming to false, making all threads stop
        self._streaming = False
        self._showing_eegpsd = False
        self._sampling_rate = 0  # sampling rate of eeg data
        self._samples = 0
        self._fft_resolution = 0
        self._channels = 1
        self._history_length = 600.0
        self._eeg_data = []
        self._sqr_data = []
        self._2d_layout = []
        try:
            self._eeg_lock.release()
        except:
            pass
        self._eeg_times = []
        self._fft_data = []
        try:
            self._fft_lock.release()
        except:
            pass
        self._fft_freqs = []
        self._bnd_data = []
        try:
            self._bnd_lock.release()
        except:
            pass
        self._hst_data = []
        try:
            self._hst_lock.release()
        except:
            pass
        self._eeg_stream = None
        self._clc_stream = None

        # set GUI elements
        try:
            self._checkbox_display_eegpsd.setEnabled(False)
            self._checkbox_display_eegpsd.setChecked(False)
            self._checkbox_visualisation_ddots.setEnabled(False)
            self._checkbox_visualisation_ddots.setChecked(False)
            self._checkbox_visualisation_rripples.setEnabled(False)
            self._checkbox_visualisation_rripples.setChecked(False)
            self._lsl_label.setEnabled(False)
            self._checkbox_connect_lsl.setText("Click to connect")
            self._eeg_label.setEnabled(False)
            self._bnd_label.setEnabled(False)
            self._create_eegpsd_tab(False)
            self._create_dancing_dots_tab(False)
            self._create_radiant_ripples_tab(False)
        except:
            pass

    def _connect_eeg_stream(self, connect, restart_resolve=True):
        """
        Connects a single EEG stream
        Add the stream 'name' to the array of streams and starts pulling data
        :return: void
        """

        # if we're connecting
        if connect and not self._streaming:

            # find what's selected
            stream_name = self._combobox_streamname.currentText()
            stream_type = stream_name[0:3]
            stream_name = stream_name[4:]
            self._name = self._lineedit_name.text() or "No Name"
            if stream_name:

                # open LSL stream
                if stream_type == 'LSL':

                    # first resolve an EEG stream on the lab network
                    print("Connecting to LSL stream... ")
                    streams = resolve_streams()

                    # create a new inlet to read from the stream
                    for s in streams:
                        s_name, s_id, s_channels, s_rate = stream_name.split(' | ')
                        if s.source_id == s_id and s_name == s.name and s_channels == "{} Channels".format(s.n_channels):
                            try:
                                # set gui info
                                self._channels = s.n_channels
                                self._sampling_rate = s.sfreq
                                self._samples = int(self._data_seconds * self._sampling_rate)
                                self._eeg_stream = Stream(bufsize=self._data_seconds, name=s_name, stype=s.stype,
                                                          source_id=s.source_id)

                                # connect to stream
                                self._eeg_stream.connect(acquisition_delay=0.1, processing_flags="all")
                                self._streaming = True
                                self._checkbox_connect_lsl.setText("Connected")

                                # start data reading thread
                                thstr = threading.Thread(target=self._read_lsl)
                                thstr.start()

                            except RuntimeError as e:
                                self._eeg_stream = None
                                self._streaming = False
                                self._checkbox_connect_lsl.setText("Connection error")
                                QtWidgets.QMessageBox.warning(None, 'Connection Error',
                                                              f'Problem connecting to the LSL stream: {e}\n\n'
                                                              'Please check your firewall setting or try again.\n\n'
                                                              'Additional information:\n'
                                                              f'name (requested/found): {s_name}/{s.name}\n'
                                                              f'type (found): {s.stype}\n'
                                                              f'source id (requested/found): {s_id} / {s.source_id}')
                                self._checkbox_connect_lsl.setText("Click to connect")

                            # get montage
                            mtgs = chn.get_builtin_montages()
                            success = False
                            for montage_name in mtgs:
                                bad_channels = 0.0     # count the number of bad channels, reject if > 5
                                try:
                                    # get default positions
                                    success = True
                                    mtg = chn.make_standard_montage(montage_name)

                                    # get names and  2-d positions from montage
                                    layout = {}
                                    for ch, crd in mtg.get_positions()['ch_pos'].items():
                                        theta = math.atan2(crd[1], math.sqrt(crd[2] ** 2 + crd[0] ** 2))
                                        phi = math.atan2(crd[0], crd[2])
                                        # x, y = math.log((1 + math.sin(phi)) / (1 - math.sin(phi))), theta     # Mercator projection
                                        # x, y = phi, theta  # equirectangular projection
                                        x, y = phi/(1.0 + theta ** 4.0), theta   # equirectangular projection, but moving frontal and occipital closer together for better looks
                                        layout[ch.lower()] = (x, y)

                                    # get channel names from our data (or make up own)
                                    try:
                                        self._ch_names = self._eeg_stream.ch_names
                                    except:
                                        self._ch_names = [f'Ch{c+1}' for c in range(0, self._channels)]
                                        success = False
                                        break

                                    # match every channel in our data with the montage
                                    for ch in self._eeg_stream.ch_names:

                                        # find channel position, or, if it is not in the montage, put it to the side
                                        label = re.match('^(EEG)? ?([0-9a-zA-Z]*)', ch)[2].lower()
                                        if label in layout.keys():
                                            x, y = layout[label]
                                        else:
                                            x,y = -2.0 + bad_channels, -2.0
                                            bad_channels += 1.0

                                        # if there are more than five non-montage channels, the montage doesn't fit
                                        if bad_channels > 5:
                                            success = False
                                            self._2d_layout = []
                                            break

                                        # add the position to our list, record min and max
                                        self._2d_layout += [[x, y]]

                                    # find min and max 2d positions, for scaling
                                    if len(self._2d_layout):
                                        xmin = min([a[0] for a in self._2d_layout])
                                        xmax = max([a[0] for a in self._2d_layout])
                                        ymin = min([a[1] for a in self._2d_layout])
                                        ymax = max([a[1] for a in self._2d_layout])

                                        # add channel colours according to postions:
                                        # starboard - green, portside - red, bow - yellow, stern - blue  :)
                                        for ch in self._2d_layout:

                                            # scale existing channels to -1,1 box
                                            ch[0] = 2.0 * (ch[0] - xmin) / (xmax - xmin) - 1.0
                                            ch[1] = 2.0 * (ch[1] - ymin) / (ymax - ymin) - 1.0

                                            # calculate channel colour based on topology
                                            red = (-ch[0] / 2.0) + 0.5  # red from left (1.0) to right (0.0)
                                            green = (ch[0] / 2.0) + 0.5  # green from right (1.0) to left (0.0)
                                            yellow = (ch[1] / 2.0) + 0.5  # yellow from front (1.0) to back (0.0)
                                            blue = (-ch[1] / 2.0) + 0.5  # blue from back (1.0) to front (0.0)
                                            red = max(red, 0.5 * yellow)
                                            green = max(green, 0.5 * yellow)
                                            ch += [min(1.0, 2.0 * red / (red+green+blue))]
                                            ch += [min(1.0, 2.0 * green / (red+green+blue))]
                                            ch += [2.0 * blue / (red+green+blue)]
                                    else:
                                        success = False

                                except KeyError:
                                    success = False
                                    self._2d_layout = []

                                if success:
                                    print(f"Using montage {montage_name}, with {bad_channels} unknown channels.")
                                    break

                            if not success:
                                print("Montage not found, setting up arbitrary channel positions.")
                                cols = math.floor(math.sqrt(self._channels))
                                rows = math.ceil(self._channels / cols)
                                for r in range(0, rows):
                                    for c in range(0, cols):
                                        if r * rows + c < self._channels:
                                            x = -1.0 + 2.0*c/(cols-1.0)
                                            y = 1.0 - 2.0*r/(rows-1.0)
                                            red = (-x / 2.0) + 0.5  # red from left (1.0) to right (0.0)
                                            green = (x / 2.0) + 0.5  # green from right (1.0) to left (0.0)
                                            yellow = (y / 2.0) + 0.5  # yellow from front (1.0) to back (0.0)
                                            blue = (-y / 2.0) + 0.5  # blue from back (1.0) to front (0.0)
                                            red = max(red, 0.5 * yellow)
                                            green = max(green, 0.5 * yellow)
                                            self._2d_layout += [[
                                                x / (1.0 + 0.5 * y ** 2.0),
                                                y / (1.0 + 0.5 * x ** 2.0),
                                                min(1.0, 2.0 * red / (red + green + blue)),
                                                min(1.0, 2.0 * green / (red + green + blue)),
                                                2.0 * blue / (red + green + blue)
                                            ]]

                if stream_type == 'SML':

                    # set gui info
                    print("Starting data simulation / self-test...")
                    self._channels = 5
                    self._sampling_rate = 250
                    self._samples = int(self._data_seconds * self._sampling_rate)
                    self._checkbox_connect_lsl.setText("Connected")
                    # start simulation thread
                    self._streaming = True
                    thstr = threading.Thread(target=self._simulate_eeg)
                    thstr.start()
                    self._2d_layout = [[-0.5, 0.5, 1.0, 0.5, 0.0],
                                       [0.5, 0.5, 0.5, 1.0, 0.0],
                                       [0.0, 0.0, 0.5, 0.5, 0.5],
                                       [-0.5, -0.5, 0.5, 0.0, 1.0],
                                       [0.5, -0.5, 0.0, 0.5, 1.0]]

                # start analysis thread
                thanal = threading.Thread(target=self._analyse_psd)
                thanal.start()

                # enable checkbox
                self._checkbox_display_eegpsd.setEnabled(True)
                self._checkbox_visualisation_ddots.setEnabled(True)
                self._checkbox_visualisation_rripples.setEnabled(True)
                print("... Data source connected.")

        # if we're disconnecting
        elif not connect and self._streaming:

            print("Disconnecting from LSL stream... ")
            self._streaming = False   # stop the display threads before we disconnect the stream
            self._eeg_stream and self._eeg_stream.disconnect()
            self._2d_layout = []
            self._reset()
            print("... LSL stream disconnected.")

            # start looking for streams again
            if restart_resolve:
                self._resolving = True
                thlsl = threading.Thread(target=self._find_sources)
                thlsl.start()
            else:
                self._resolving = False

    def _create_eegpsd_tab(self, create):

        if create:

            # note
            print("Creating EEG/PSD display tab.")

            # create widgets
            self._eegpsd_tab = QtWidgets.QWidget()
            self._eegpsd_layout = QtWidgets.QGridLayout(self._eegpsd_tab)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
            self._eegpsd_tab.setSizePolicy(sizePolicy)
            self._parent_tabwidget.addTab(self._eegpsd_tab, "")
            self._parent_tabwidget.setTabText(self._parent_tabwidget.indexOf(self._eegpsd_tab),
                                              self._name + " -- EEG / Spectrum")

            # colours
            frame_c = (0.25, 0.25, 0.25)
            background = self._parent_tabwidget.parent().parent().palette().base().color()
            brightness = sum(background.getRgb()[:3]) / (3 * 255.0)
            outer_c = background.name()
            passepartout_c = self._parent_tabwidget.parent().parent().palette().button().color().name() #(0.25, 0.25, 0.25)
            if brightness < 0.5:
                label_c = (0.8, 0.8, 0.8)
                title_c = (0.9, 0.9, 0.9)
            else:
                label_c = (0.2, 0.2, 0.2)
                title_c = (0.1, 0.1, 0.1)

            # first activity plot (normalised squared signal)
            figure = plt.figure()
            self._sqr_canvas = FigureCanvasQTAgg(figure)
            self._sqr_axes = figure.add_subplot(111)
            self._eegpsd_layout.addWidget(self._sqr_canvas, 0, 0, 1, 1)
            self._sqr_axes.set_ylim(bottom=-0.2, top=self._channels + 1.2)
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=1.0)
            self._sqr_axes.set_xscale('symlog')
            self._sqr_axes.set_xticks([])
            self._sqr_axes.set_yticks(ticks=np.arange(1, self._channels + 1), labels=self._ch_names[::-1], color=label_c, fontsize=8)
            self._sqr_axes.set_title('{} — rel. {:0.1f}-Seconds-Variance over {} minutes'.format(self._name,
                                                                                             self._data_seconds,
                                                                             self._history_length/60.0),
                                     color=title_c, fontsize=10, pad=5)
            self._sqr_axes.set_facecolor(outer_c)
            figure.set_facecolor(passepartout_c)
            plt.setp(self._sqr_axes.spines.values(), color=frame_c)

            # first eeg plot
            figure = plt.figure()
            self._eeg_canvas = FigureCanvasQTAgg(figure)
            self._eeg_axes = figure.add_subplot(111)
            self._eegpsd_layout.addWidget(self._eeg_canvas, 0, 1, 1, 1)
            self._eeg_axes.set_ylim(bottom=-0.2, top=self._channels + 1.2)
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.0, right=0.99)
            self._eeg_axes.set_xticks([])
            self._eeg_axes.set_yticks(ticks=[])
            self._eeg_axes.set_title('{} — EEG over {:0.1f} Seconds'.format(self._name, self._data_seconds),
                                     color=title_c, fontsize=10, pad=5)
            self._eeg_axes.set_facecolor(outer_c)
            figure.set_facecolor(passepartout_c)
            plt.setp(self._eeg_axes.spines.values(), color=frame_c)

            # first psd plot
            figure = plt.figure()
            self._fft_canvas = FigureCanvasQTAgg(figure)
            self._fft_axes = figure.add_subplot(111)
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.15, right=0.99)
            self._eegpsd_layout.addWidget(self._fft_canvas, 0, 2, 1, 1)
            self._fft_axes.set_ylim(bottom=-0.2, top=self._channels + 1.2)
            self._fft_axes.set_yticks(ticks=np.arange(1, self._channels + 1), labels=self._ch_names[::-1], color=label_c, fontsize=8)
            self._fft_axes.set_title('rel. PSD', color=title_c, fontsize=10, pad=5)
            self._fft_axes.set_facecolor(outer_c)
            figure.set_facecolor(passepartout_c)
            plt.setp(self._fft_axes.spines.values(), color=frame_c)

            # bandpass history plot
            figure = plt.figure()
            self._hst_canvas = FigureCanvasQTAgg(figure)
            self._hst_axes = figure.add_subplot(111)
            plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.99)
            self._eegpsd_layout.addWidget(self._hst_canvas, 1, 0, 1, 2)
            self._hst_axes.set_ylim([-0.1, 5.1])
            self._hst_axes.set_xticks([])
            self._hst_axes.set_xscale('symlog')
            self._hst_axes.set_yticks([0, 1, 2, 3, 4], ['δ', 'θ', 'α', 'β', 'γ'], color=label_c)
            self._hst_axes.set_title('{} — rel. PSD History over {} minutes'.format(self._name, self._history_length/60.0),
                                     color=title_c, fontsize=10, pad=5)
            self._hst_axes.set_facecolor(outer_c)
            figure.set_facecolor(passepartout_c)
            plt.setp(self._hst_axes.spines.values(), color=frame_c)

            # bandpass bar graph
            figure = plt.figure()
            self._bnd_canvas = FigureCanvasQTAgg(figure)
            self._bnd_axes = figure.add_subplot(111)
            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.99)
            self._eegpsd_layout.addWidget(self._bnd_canvas, 1, 2, 1, 1)
            self._bnd_axes.set_ylim([0.0, 1.1])
            self._bnd_axes.set_xticks([1, 2, 3, 4, 5], ['δ', 'θ', 'α', 'β', 'γ'], color=label_c)
            self._bnd_axes.set_yticks([])
            self._bnd_axes.set_title('rel. Frequency Band Power'.format(self._data_seconds),
                                     color=title_c, fontsize=10, pad=5)
            self._bnd_axes.set_facecolor(outer_c)
            figure.set_facecolor(passepartout_c)
            plt.setp(self._bnd_axes.spines.values(), color=frame_c)

            self._eegpsd_layout.setColumnStretch(0, 3)
            self._eegpsd_layout.setColumnStretch(1, 2)
            self._eegpsd_layout.setColumnStretch(2, 2)
            self._eegpsd_layout.setRowStretch(0, 2)
            self._eegpsd_layout.setRowStretch(1, 1)

            # start display thread
            time.sleep(1)
            self._showing_eegpsd = True
            thdsp = threading.Thread(target=self._display_eeg_psd)
            thdsp.start()

        else:
            if self._eegpsd_tab:
                print("Removing EEG/PSD display tab.")
                self._parent_tabwidget.removeTab(self._parent_tabwidget.indexOf(self._eegpsd_tab))
            self._showing_eegpsd = False

    def _create_dancing_dots_tab(self, create):

        if create:

            # note
            print("Creating Dancing Dots display tab.")

            # create tab
            self._ddots_tab = QtWidgets.QWidget()

            # add dancing dot display layout to tab
            DancingDotsLayout(self._ddots_tab, self.get_bnd_data)
            self._parent_tabwidget.addTab(self._ddots_tab, "")
            self._parent_tabwidget.setTabText(self._parent_tabwidget.indexOf(self._ddots_tab),
                                              self._name + " -- Dancing Dots")

        else:
            if self._ddots_tab:
                print("Removing Dancing Dots display tab.")
                self._parent_tabwidget.removeTab(self._parent_tabwidget.indexOf(self._ddots_tab))

    def _create_radiant_ripples_tab(self, create):

        if create:

            # note
            print("Creating Radiant Ripples display tab.")

            # create tab
            self._rripples_tab = QtWidgets.QWidget()

            # add radiant ripples display layout to tab
            RadiantRipplesLayout(self._rripples_tab, self.get_sqr_data, self.get_2d_layout)
            self._parent_tabwidget.addTab(self._rripples_tab, "")
            self._parent_tabwidget.setTabText(self._parent_tabwidget.indexOf(self._rripples_tab),
                                              self._name + " -- Radiant Ripples")

        else:
            if self._rripples_tab:
                print("Removing Radiant Ripples display tab.")
                self._parent_tabwidget.removeTab(self._parent_tabwidget.indexOf(self._rripples_tab))

    def _display_eeg_psd(self):

        # starting thread
        print("Starting EEG/PSD display.")
        while not len(self._eeg_data)\
                or not len(self._sqr_data)\
                or not len(self._fft_data)\
                or not len(self._bnd_data)\
                or not len(self._hst_data):
            time.sleep(0.2)

        # eeg + fft
        eeg_lines = self._eeg_axes.plot(self._eeg_data.T)  # the last channel in the simulator has the alpha intensity
        self._eeg_axes.set_xlim(0, self._eeg_data.shape[1])
        sqr_lines = self._sqr_axes.plot(np.arange(float(-self._sqr_data.shape[1]), 0.0), self._sqr_data.T)
        self._sqr_axes.set_xlim(-float(self._sqr_data.shape[1]), -2.0)
        fft_lines = self._fft_axes.plot(self._fft_freqs, self._fft_data.T)
        self._fft_axes.set_xlim(self._fft_freqs[0], self._fft_freqs[-1])
        bnd_bars = self._bnd_axes.bar([1, 2, 3, 4, 5], self._bnd_data)
        hst_lines = self._hst_axes.plot(np.arange(float(-self._hst_data.shape[1]), 0.0), self._hst_data.T)
        self._hst_axes.set_xlim(-self._hst_data.shape[1], -2.0)

        background = self._parent_tabwidget.parent().parent().palette().base().color()
        brightness = sum(background.getRgb()[:3]) / (3 * 255.0)

        # set rainbow colours for eeg and fft
        for c in range(0, len(eeg_lines)):
            if brightness < 0.5:
                colour = (self._2d_layout[c][2], self._2d_layout[c][3], self._2d_layout[c][4])
            else:
                colour = (0.5*self._2d_layout[c][2], 0.5*self._2d_layout[c][3], 0.5*self._2d_layout[c][4])
            eeg_lines[c].set_color(color=colour)
            eeg_lines[c].set_linewidth(0.4)
            fft_lines[c].set_color(color=colour)
            fft_lines[c].set_linewidth(0.4)
            sqr_lines[c].set_color(color=colour)
            sqr_lines[c].set_linewidth(0.4)

        # set rainbow colours for frequency bands
        for n in range(0, 5):
            a = n/4.0
            if brightness < 0.5:
                colour = (0.3+0.7*(1 - a), 0.5+0.5*(1.0 - 2.0*abs(a - 0.5)), 0.3+0.7*a)
            else:
                colour = (0.7 - 0.7 * (1 - a), 0.5 - 0.5 * (1.0 - 2.0 * abs(a - 0.5)), 0.7 - 0.7 * a)
            bnd_bars[n].set(color=colour)
            hst_lines[n].set_color(color=colour)
            hst_lines[n].set_linewidth(0.5)

        while self._streaming and self._showing_eegpsd:
            try:
                with self._eeg_lock:
                    eeg_max = self._eeg_data.max()
                    eeg_min = self._eeg_data.min()
                    for c in range(0, len(eeg_lines)):
                        eeg_lines[c].set_ydata(self._eeg_data[c]/self._eeg_channel_height + float(self._channels - c))
                self._eeg_channel_height = 0.5*(eeg_max - eeg_min)
                with self._sqr_lock:
                    sqr_height = self._sqr_data.max() or 1.0
                    for c in range(0, len(sqr_lines)):
                        sqr_lines[c].set_ydata(self._sqr_data[c] / sqr_height + float(self._channels - c) - 0.5)
                with self._fft_lock:
                    self._fft_channel_height = 0.5*(self._fft_data.max() - self._fft_data.min())
                    for c in range(0, len(fft_lines)):
                        fft_lines[c].set_ydata(self._fft_data[c] / self._fft_channel_height + float(self._channels - c) - 0.5)
                with self._hst_lock:
                    hst_height = self._hst_data.max()
                    hst_height = hst_height or 1.0
                    for b in range(0, len(hst_lines)):
                        hst_lines[b].set_ydata(self._hst_data[b] / hst_height + float(b))
                with self._bnd_lock:
                    for b in range(0, len(bnd_bars)):
                        bnd_bars[b].set_height(self._bnd_data[b] / hst_height)
                self._eeg_canvas.draw()
                self._sqr_canvas.draw()
                self._fft_canvas.draw()
                self._bnd_canvas.draw()
                self._hst_canvas.draw()
            except Exception as e:
                print(e)
                time.sleep(0.5)

        # done.
        print("Ending EEG/PSD display.")

    def _read_lsl(self):
        """
        Read data into buffer a, then call a new thread
        :return:
        """

        # Begin
        print("Starting LSL reading.")

        # init data buffers
        self._eeg_stream.filter(2, 70)
        self._eeg_stream.notch_filter(50)
        self._eeg_stream.get_data()  # reset the number of new samples after the filter is applied
        with self._eeg_lock:
            self._eeg_data = np.zeros((self._channels, self._samples))
        with self._sqr_lock:
            self._sqr_data = np.zeros((self._channels, int(self._history_length * 5.0)))  # history length * update rate of the analysis thread
        with self._fft_lock:
            self._fft_data = np.zeros((self._channels, int(self._samples/2)))
        with self._bnd_lock:
            self._bnd_data = np.zeros(5)
        with self._hst_lock:
            self._hst_data = np.zeros((5, int(self._history_length * 5.0)))  # history length * update rate of the analysis thread

        # start streaming loop
        self._lsl_label.setEnabled(True)
        while self._streaming:
            with self._eeg_lock:
                self._eeg_data, ts = self._eeg_stream.get_data()
            with self._gui_lock:
                self._lsl_info = "LSL Time {:0.1f}".format(ts[-1])
            #time.sleep(0.1)

        # done.
        print("Ending LSL reading.")

    def _simulate_eeg(self):
        """
        Simulate EEG and write into buffer
        :return:
        """

        # Begin
        print("Starting simulation.")

        # init data buffers
        with self._eeg_lock:
            self._eeg_data = np.zeros((self._channels, self._samples))
        with self._sqr_lock:
            self._sqr_data = np.zeros((self._channels, self._samples))
        with self._fft_lock:
            self._fft_data = np.zeros((self._channels, int(self._samples / 2)))
        with self._bnd_lock:
            self._bnd_data = np.zeros(5)
        with self._hst_lock:
            self._hst_data = np.zeros(
                (5, int(self._history_length * 5.0)))  # history length * update rate of the analysis thread

        # init noise sources (sine waves of known frequencies)
        samples = 2 * int(self._sampling_rate)
        freqs = np.round((np.cumsum(np.cumsum(np.ones(40))))*39/820+1)     # going from 1 Hz to 40 Hz, staying longer at the low ones
        waves = np.sin(np.cumsum(2.0*np.pi*np.ones((40, samples))/(self._sampling_rate), axis=1).T * freqs).T

        # start streaming loop
        self._lsl_label.setEnabled(True)
        f_window = np.array([0,0,0,0,0])
        t_window = np.arange(int(samples/10))
        cycle = 0
        metacycle = 0
        while self._streaming:
            with self._eeg_lock:
                self._eeg_data = np.roll(self._eeg_data, -int(samples/10))
                self._eeg_data[:, -int(samples/10):] = (waves[f_window, :])[:, t_window]
            with self._gui_lock:
                self._lsl_info = "SML Time {:0.1f} s".format(1.0)
            t_window = np.mod(t_window + int(samples/10), samples)
            cycle += 1
            if cycle == 10:
                cycle = 0
                metacycle += 1
                f_window = np.mod(f_window+1, 40)
                t_window = np.arange(int(samples/10))
            if metacycle == 40:
                metacycle = 0
                f_window = np.mod(f_window + np.array([0,1,2,3,4]), 40)
            time.sleep(0.1)

        # done.
        print("Ending LSL reading.")

    def _analyse_psd(self):
        """
        Read data into buffer a, then call a new thread
        :return:
        """

        # start
        print("Starting analysis.")
        while not len(self._eeg_data):
            time.sleep(0.5)

        # activate labels
        self._eeg_label.setEnabled(True)
        self._bnd_label.setEnabled(True)

        # initialise data objects
        self._fft_freqs = np.fft.rfftfreq(self._samples, d=1.0 / self._sampling_rate)
        bin_freqs = np.array([3.5, 7.5, 12.5, 30.5, 50.0, 60.0])   # delta, theta, alpha, beta, gamma, total
        bins = [abs(self._fft_freqs - f).argmin() for f in bin_freqs]
        widths = np.insert(bin_freqs[1:] - bin_freqs[:-1], 0, bin_freqs[0])
        self._fft_resolution = self._fft_freqs[1]
        self._fft_max = bins[-1]
        self._fft_running_mean = 1e-6 * np.ones((1, self._fft_max))
        self._fft_freqs = self._fft_freqs[:self._fft_max]
        smooth = self._bnd_smoothing

        # start streaming loop
        while self._streaming:
            try:
                with (self._eeg_lock):
                    with self._sqr_lock:
                        self._sqr_data = np.roll(self._sqr_data, -1)
                        var = self._eeg_data.var(1)
                        self._sqr_data[:, -1] = (var / (var.sum() or 1.0) ) * self._channels    # ensure each channel goes from 0.0 to 1.0
                    with self._fft_lock:
                        eeg_min = self._eeg_data.min()
                        eeg_max = self._eeg_data.max()
                        self._fft_data = np.fft.rfft(self._eeg_data, axis=1)
                    self._fft_data = (np.abs(self._fft_data)**2)[:,:self._fft_max]
                    self._fft_running_mean *= 0.99
                    self._fft_running_mean += 0.01 * self._fft_data.sum(axis=0) / self._channels
                    self._fft_data /= self._fft_running_mean
                    fft_all_channels = self._fft_data.sum(axis=0)[1:] / self._channels     # sum of fft over all channels, excluding DC
                    c = self._fft_resolution     # normalise each band by its width, as if it were 1.0 wide
                    bnd_data = np.array([a[0].sum() * c / a[1] for a in
                                         zip(np.split(fft_all_channels, bins)[:5], widths)])
                    bnd_data = bnd_data / (bnd_data.sum() or 1.0)   # relative power
                    with self._bnd_lock:
                        self._bnd_data = smooth*self._bnd_data + (1.0-smooth)*bnd_data
                        with self._hst_lock:
                            self._hst_data[:, :-1] = self._hst_data[:, 1:]
                            self._hst_data[:, -1] = self._bnd_data
                    self._bnd_info = "Freq δ {:0.1f} | θ {:0.1f} | α {:0.1f} | β {:0.1f} | γ {:0.1f}".format(*self._bnd_data)
                    self._eeg_info = "{:.1f} µV - {:.1f} µV".format(eeg_min * 1e6, eeg_max * 1e6)

            except Exception as e:
                print(e)
                time.sleep(0.5)
            time.sleep(0.1)

        # done.
        print("Ending analysis.")

    def _find_sources(self):
        while self._resolving:
            time.sleep(1.0)
            if not self._streaming:
                streams = resolve_streams(timeout=1)
                with self._gui_lock:
                    self._available_streams = ["LSL {} | {} | {} Channels | {} Hz"
                                               "".format(s.name, s.source_id, s.n_channels, s.sfreq)
                                               for s in streams] + ["SML Simulated data / selftest"]

    def _update_gui(self):
        with self._gui_lock:
            if self._streaming:
                self._lsl_label.setText(self._lsl_info)
                self._bnd_label.setText(self._bnd_info)
                self._eeg_label.setText(self._eeg_info)
            else:
                entries = []
                for n in range(self._combobox_streamname.count()):
                    entries += [self._combobox_streamname.itemText(n)]
                for e in entries:
                    if e not in self._available_streams:
                        self._combobox_streamname.removeItem(self._combobox_streamname.findText(e))
                for s in self._available_streams:
                    if s not in entries:
                        self._combobox_streamname.addItem(s)
                if len(entries):
                    self._checkbox_connect_lsl.setEnabled(True)
                else:
                    self._checkbox_connect_lsl.setEnabled(False)

    def get_bnd_data(self):
        return self._bnd_data

    def get_sqr_data(self):
        return self._sqr_data

    def get_2d_layout(self):
        return self._2d_layout


class SamadhiWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    Implements the main part of the GUI.
    """

    _minds = []

    def __init__(self, filename=None):
        """
        Initialise the application, connect all button signals to application functions, load theme and last file
        """

        print("starting Samadhi...")

        # initialise main window
        super().__init__()
        self.setupUi(self)

        # connect all button signals
        print("setting up GUI...")
        #
        # ...

        # init application settings
        #print("loading system settings...")
        #self.settings = QtCore.QSettings('FreeSoftware', 'Samadhi')

        # add one mind
        self.add_mind(self.comboBoxStreamName01, self.lineEditName01, self.checkBoxConnect01,
                      self.checkBoxDspEegPsd, self.checkBoxDspDancingDots, self.checkBoxDspRadiantRipples,
                      self.labelLslStatus, self.labelEegStatus, self.labelFrequencyBands)

    def __del__(self):
        print("Closing main window... ")
        for m in self._minds:
            m.__del__()
        print("... main windows closed.")

    def add_mind(self, combobox_streamname, lineedit_name, checkbox_connect,
                       checkbox_display_eegpsd, checkbox_display_ddots, checkbox_display_rripples,
                       lsl_info, eeg_info, bnd_info):
        self._minds.append(Mind(combobox_streamname, lineedit_name, checkbox_connect,
                                checkbox_display_eegpsd, checkbox_display_ddots, checkbox_display_rripples,
                                lsl_info, eeg_info, bnd_info, self.tabWidget))


class Samadhi:

    def __init__(self, filename=None):

        format = QSurfaceFormat()
        format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        format.setVersion(3, 3)
        QSurfaceFormat.setDefaultFormat(format)
        app = QtWidgets.QApplication(sys.argv)

        # start main window
        main_window = SamadhiWindow()

        # run
        main_window.show()
        sys.exit(app.exec())
