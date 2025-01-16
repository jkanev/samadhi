# -*- coding:utf-8 -*-
#!/usr/bin/python3

import sys

from .mainwindow import *
from PyQt6 import QtCore, QtWidgets
import threading
import numpy as np
import time
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.lsl import resolve_streams
from matplotlib import use as mpl_use
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from .display.layouts import DancingDotsLayout

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
    _data_seconds = 10.0  # how much data do we have in the _eeg_data array
    _sampling_rate = 0   # sampling rate of eeg data
    _samples = 0         # seconds times sampling rate
    _fft_resolution = 0  # resolution (distance of one FFT bin to the next)
    _channels = 1        # number of channels in the data
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
    _eeg_stream = None   # the lsl eeg input stream inlet, if in eeg mode
    _clc_stream = None   # the lsl calculation output stream outlet, if in calculation mode

    # normalisation
    _bnd_smoothing = 0.95   #
    _bnd_max = []       # maximum values of bands, calculated from data d(t) by x(t+1) = max(0.99*x(t), d(t))
    _bnd_min = []       # minimum values of bands, calculated from data d(t) by x(t+1) = min(1.01*x(t), d(t))
    _bnd_mid = []       # the middle between max and min

    # main mind controls
    _combobox_streamname = False
    _lineedit_name = False
    _checkbox_connect_lsl = False
    _checkbox_analyse_psd = False
    _checkbox_display_eegpsd = False
    _checkbox_visualisation_ddots = False
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
    _fft_axes = False
    _fft_canvas = False
    _fft_channel_height = 50e-4
    _bnd_axes = False
    _bnd_canvas = False
    _bnd_height = 110e-4
    _hst_axes = False
    _hst_canvas = False
    _hst_height = 110e-4

    # dancing dots display
    _ddots_tab = False

    def __init__(self, combobox_streamname, lineedit_name, checkbox_connect, checkbox_analyse_psd,
                       checkbox_display_eegpsd, checkbox_visualisation_ddots,
                       lsl_info, eeg_info, bnd_info,
                       parent_tabwidget):
        self._combobox_streamname = combobox_streamname
        self._lineedit_name = lineedit_name
        self._checkbox_connect_lsl = checkbox_connect
        self._checkbox_analyse_psd = checkbox_analyse_psd
        self._checkbox_display_eegpsd = checkbox_display_eegpsd
        self._checkbox_visualisation_ddots = checkbox_visualisation_ddots
        self._lsl_label = lsl_info
        self._eeg_label = eeg_info
        self._bnd_label = bnd_info
        self._parent_tabwidget = parent_tabwidget

        # Connect slot eeg stream
        self._checkbox_connect_lsl.clicked.connect(self._connect_eeg_stream)
        self._checkbox_display_eegpsd.clicked.connect(self._create_eegpsd_tab)
        self._checkbox_visualisation_ddots.clicked.connect(self._create_dancing_dots_tab)

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
            self._lsl_label.setEnabled(False)
            self._checkbox_connect_lsl.setText("Click to connect")
            self._eeg_label.setEnabled(False)
            self._create_eegpsd_tab(False)
            self._create_dancing_dots_tab(False)
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
                        if s.source_id == s_id:

                            # set gui info
                            self._channels = s.n_channels
                            self._sampling_rate = s.sfreq
                            self._samples = int(self._data_seconds * self._sampling_rate)
                            self._eeg_stream = Stream(self._data_seconds, name=s_name, stype=s.stype, source_id=s.source_id)
                            self._checkbox_connect_lsl.setText("Connected")

                            # connect to stream
                            self._eeg_stream.connect(acquisition_delay=0.1, processing_flags="all")
                            self._streaming = True

                            # start data reading thread
                            thstr = threading.Thread(target=self._read_lsl)
                            thstr.start()

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

                # start analysis thread
                thanal = threading.Thread(target=self._analyse_psd)
                thanal.start()

                # enable checkbox
                self._checkbox_display_eegpsd.setEnabled(True)
                self._checkbox_visualisation_ddots.setEnabled(True)
                print("... Data source connected.")

        # if we're disconnecting
        elif not connect and self._streaming:

            print("Disconnecting from LSL stream... ")
            self._streaming = False   # stop the display threads before we disconnect the stream
            self._eeg_stream and self._eeg_stream.disconnect()
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

            # channel names
            c_names = ((self._eeg_stream and self._eeg_stream.ch_names)
                       or ['C{}'.format(n+1) for n in range(0, self._channels)])

            # colours
            frame_c = (0.25, 0.25, 0.25)
            outer_c = self._parent_tabwidget.parent().parent().palette().base().color().name()
            passepartout_c = self._parent_tabwidget.parent().parent().palette().button().color().name() #(0.25, 0.25, 0.25)
            label_c = (0.8, 0.8, 0.8)
            title_c = (0.9, 0.9, 0.9)

            # first eeg plot
            figure = plt.figure()
            self._eeg_canvas = FigureCanvasQTAgg(figure)
            self._eeg_axes = figure.add_subplot(111)
            self._eegpsd_layout.addWidget(self._eeg_canvas, 0, 0, 1, 1)
            self._eeg_axes.set_ylim(bottom=0.0, top=self._channels + 2)
            self._eeg_axes.set_xticks([])
            self._eeg_axes.set_yticks(ticks=np.arange(1, self._channels + 1), labels=c_names, color=label_c)
            self._eeg_axes.set_title('{} -- EEG over {:0.1f} Seconds'.format(self._name, self._data_seconds),
                                     color=title_c)
            self._eeg_axes.set_facecolor(outer_c)
            figure.set_facecolor(passepartout_c)
            plt.setp(self._eeg_axes.spines.values(), color=frame_c)

            # first psd plot
            figure = plt.figure()
            self._fft_canvas = FigureCanvasQTAgg(figure)
            self._fft_axes = figure.add_subplot(111)
            self._eegpsd_layout.addWidget(self._fft_canvas, 0, 1, 1, 1)
            self._fft_axes.set_ylim(bottom=0.0, top=self._channels + 2)
            self._fft_axes.set_xscale('log')
            self._fft_axes.set_yticks(ticks=np.arange(1, self._channels + 1), labels=c_names, color=label_c)
            self._fft_axes.set_title('Current PSD', color=title_c)
            self._fft_axes.set_facecolor(outer_c)
            figure.set_facecolor(passepartout_c)
            plt.setp(self._fft_axes.spines.values(), color=frame_c)

            # bandpass history plot
            figure = plt.figure()
            self._hst_canvas = FigureCanvasQTAgg(figure)
            self._hst_axes = figure.add_subplot(111)
            self._eegpsd_layout.addWidget(self._hst_canvas, 1, 0, 1, 1)
            self._hst_axes.set_ylim([-0.1, 5.1])
            self._hst_axes.set_xticks([])
            self._hst_axes.set_yticks([0, 1, 2, 3, 4], ['δ', 'θ', 'α', 'β', 'γ'], color=label_c)
            self._hst_axes.set_title('{} -- PSD History over {} minutes'.format(self._name, self._history_length/60.0),
                                     color=title_c)
            self._hst_axes.set_facecolor(outer_c)
            figure.set_facecolor(passepartout_c)
            plt.setp(self._hst_axes.spines.values(), color=frame_c)

            # bandpass bar graph
            figure = plt.figure()
            self._bnd_canvas = FigureCanvasQTAgg(figure)
            self._bnd_axes = figure.add_subplot(111)
            self._eegpsd_layout.addWidget(self._bnd_canvas, 1, 1, 1, 1)
            self._bnd_axes.set_ylim([0.0, 1.1])
            self._bnd_axes.set_xticks([1, 2, 3, 4, 5], ['δ', 'θ', 'α', 'β', 'γ'], color=label_c)
            self._bnd_axes.set_yticks([])
            self._bnd_axes.set_title('Frequency Band Power'.format(self._data_seconds), color=title_c)
            self._bnd_axes.set_facecolor(outer_c)
            figure.set_facecolor(passepartout_c)
            plt.setp(self._bnd_axes.spines.values(), color=frame_c)

            self._eegpsd_layout.setColumnStretch(0, 3)
            self._eegpsd_layout.setColumnStretch(1, 1)
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
            DancingDotsLayout(self._ddots_tab, self.get_data)
            self._parent_tabwidget.addTab(self._ddots_tab, "")
            self._parent_tabwidget.setTabText(self._parent_tabwidget.indexOf(self._ddots_tab),
                                              self._name + " -- Dancing Dots")

        else:
            if self._ddots_tab:
                print("Removing Dancing Dots display tab.")
                self._parent_tabwidget.removeTab(self._parent_tabwidget.indexOf(self._ddots_tab))

    def _display_eeg_psd(self):

        # starting thread
        print("Starting EEG/PSD display.")
        while not len(self._eeg_data)\
                or not len(self._fft_data)\
                or not len(self._bnd_data)\
                or not len(self._hst_data):
            time.sleep(0.2)

        # eeg + fft
        eeg_lines = self._eeg_axes.plot(self._eeg_data.T)  # the last channel in the simulator has the alpha intensity
        fft_lines = self._fft_axes.plot(self._fft_freqs, self._fft_data.T)
        bnd_bars = self._bnd_axes.bar([1, 2, 3, 4, 5], self._bnd_data)
        hst_lines = self._hst_axes.plot(self._hst_data.T)

        # set rainbow colours for eeg and fft
        for c in range(0, len(eeg_lines)):
            a = c/self._channels
            colour = (0.3+0.7*(1 - a), 0.5+0.5*(1.0 - 2.0*abs(a - 0.5)), 0.3+0.7*a)
            eeg_lines[c].set_color(color=colour)
            eeg_lines[c].set_linewidth(0.4)
            fft_lines[c].set_color(color=colour)
            fft_lines[c].set_linewidth(0.4)

        # set rainbow colours for frequency bands
        for n in range(0, 5):
            a = n/4.0
            colour = (0.3+0.7*(1 - a), 0.5+0.5*(1.0 - 2.0*abs(a - 0.5)), 0.3+0.7*a)
            bnd_bars[n].set(color=colour)
            hst_lines[n].set_color(color=colour)
            hst_lines[n].set_linewidth(0.5)

        self._eeg_label.setEnabled(True)
        while self._streaming and self._showing_eegpsd:
            try:
                with self._eeg_lock:
                    eeg_max = self._eeg_data.max()
                    eeg_min = self._eeg_data.min()
                    for c in range(0, len(eeg_lines)):
                        eeg_lines[c].set_ydata(self._eeg_data[c]/self._eeg_channel_height + float(self._channels - c))
                self._eeg_channel_height = 0.5*(eeg_max - eeg_min)
                with self._fft_lock:
                    self._fft_channel_height = 0.5*(self._fft_data.max() - self._fft_data.min())
                    for c in range(0, len(fft_lines)):
                        fft_lines[c].set_ydata(self._fft_data[c]/self._fft_channel_height + float(self._channels - c))
                with self._hst_lock:
                    hst_height = self._hst_data.max()
                    hst_height = hst_height or 1.0
                    for b in range(0, len(hst_lines)):
                        hst_lines[b].set_ydata(self._hst_data[b] / hst_height + float(b))
                with self._bnd_lock:
                    for b in range(0, len(bnd_bars)):
                        bnd_bars[b].set_height(self._bnd_data[b] / hst_height)
                self._eeg_canvas.draw()
                self._fft_canvas.draw()
                self._bnd_canvas.draw()
                self._hst_canvas.draw()
                with self._gui_lock:
                    self._eeg_info = "{:.1f} µV - {:.1f} µV".format(eeg_min*1e6, eeg_max*1e6)
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
        with self._fft_lock:
            self._fft_data = np.zeros((self._channels, int(self._samples / 2)))
        with self._bnd_lock:
            self._bnd_data = np.zeros(5)
        with self._hst_lock:
            self._hst_data = np.zeros(
                (5, int(self._history_length * 5.0)))  # history length * update rate of the analysis thread

        # init noise sources (sine waves of known frequencies)
        samples = 2 * int(self._sampling_rate)
        freqs = np.cumsum(np.ones(40))     # going from 1 Hz to 40 Hz
        waves = np.sin(np.cumsum(2.0*np.pi*np.ones((40, samples))/(self._sampling_rate), axis=1).T * freqs).T

        # start streaming loop
        self._lsl_label.setEnabled(True)
        f_window = np.array([0,0,0,0,0])
        t_window = np.arange(int(samples/10))
        cycle = 0
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
                f_window = np.mod(f_window+1, 40)
                t_window = np.arange(int(samples/10))
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

        self._fft_freqs = np.fft.rfftfreq(self._samples, d=1.0 / self._sampling_rate, device=None)
        bin_freqs = np.array([3.5, 7.5, 12.5, 30.5, 50.0, 70.0])   # delta, theta, alpha, beta, gamma, total
        bins = [abs(self._fft_freqs - f).argmin() for f in bin_freqs]
        widths = np.insert(bin_freqs[1:] - bin_freqs[:-1], 0, bin_freqs[0])
        self._fft_resolution = self._fft_freqs[1]
        smooth = self._bnd_smoothing
        is_relative = True
        is_relative_total = True
        normalisation = [1.0, 2.0, 4.0, 8.0, 16.0]
        is_normalised = True
        # start streaming loop
        while self._streaming:
            try:
                with self._fft_lock:
                    with self._eeg_lock:
                        self._fft_data = np.fft.rfft(self._eeg_data, axis=1)
                    self._fft_data = np.abs(self._fft_data)**2
                    fft_all_channels = self._fft_data.sum(axis=0)[1:] / self._channels     # sum of fft over all channels, excluding DC
                    c = self._fft_resolution     # normalise each band by its width, as if it were 1.0 wide
                    if is_relative:
                        bnd_data = np.array([a[0].sum() * c / a[1] for a in
                                             zip(np.split(fft_all_channels, bins)[:5], widths)])
                    else:
                        bnd_data = np.array([a.sum() * c for a in np.split(fft_all_channels, bins)[:5]])
                    if is_normalised:
                        bnd_data = bnd_data * normalisation
                    if is_relative_total:
                        bnd_data = bnd_data / (bnd_data.sum() or 1.0)   # relative power
                    with self._bnd_lock:
                        self._bnd_data = smooth*self._bnd_data + (1.0-smooth)*bnd_data
                        with self._hst_lock:
                            self._hst_data[:, :-1] = self._hst_data[:, 1:]
                            self._hst_data[:, -1] = self._bnd_data
                    if is_relative_total:
                        self._bnd_info = "{:0.1f} | {:0.1f} | {:0.1f} | {:0.1f} | {:0.1f}".format(*self._bnd_data)
                    else:
                        self._bnd_info = "{:0.1f} | {:0.1f} | {:0.1f} | {:0.1f} | {:0.1f}".format(
                            self._bnd_data[0] * 1e6, self._bnd_data[1] * 1e6, self._bnd_data[2] * 1e6,
                            self._bnd_data[3] * 1e6,
                            self._bnd_data[4] * 1e6)
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

    def get_data(self):
        return self._bnd_data

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
                      self.checkBoxAnalPSD, self.checkBoxDspEegPsd, self.checkBoxDspDancingDots,
                      self.labelLslStatus, self.labelEegStatus, self.labelFrequencyBands)

    def __del__(self):
        print("Closing main window... ")
        for m in self._minds:
            m.__del__()
        print("... main windows closed.")

    def add_mind(self, combobox_streamname, lineedit_name, checkbox_connect,
                       checkbox_analyse,
                       checkbox_display_eegpsd, checkbox_display_ddots,
                       lsl_info, eeg_info, bnd_info):
        self._minds.append(Mind(combobox_streamname, lineedit_name, checkbox_connect,
                                checkbox_analyse,
                                checkbox_display_eegpsd, checkbox_display_ddots,
                                lsl_info, eeg_info, bnd_info, self.tabWidget))


class Samadhi:

    def __init__(self, filename=None):
        app = QtWidgets.QApplication(sys.argv)

        # start main window
        main_window = SamadhiWindow()

        # run
        main_window.show()
        sys.exit(app.exec())
