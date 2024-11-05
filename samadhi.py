
# -*- coding:utf-8 -*-

#!/usr/bin/python3

import sys
from mainwindow import *
from PyQt6 import QtCore, QtGui, QtWidgets
from pylsl import StreamInfo, StreamInlet
import threading
import numpy as np
import time
import mne
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.lsl import resolve_streams
from matplotlib import use as mpl_use
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, FigureManagerQT
from matplotlib import pyplot as plt
mpl_use("QtAgg")

class Mind:
    """
    Implements a complete data and calculation set of a single human.
    """

    # general
    _name = ""           # the person's name

    # data streaming related
    _streaming = False   # whether we're streaming currently
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

    # info labels
    _bnd_info = False
    _lsl_info = False
    _eeg_info = False

    # eeg display research controls → put into new class soon
    _displaylayout = False
    _displaytab = False
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
        self._lsl_info = lsl_info
        self._eeg_info = eeg_info
        self._bnd_info = bnd_info
        self._parent_tabwidget = parent_tabwidget

        # Fill values
        streams = resolve_streams(timeout=1)
        for s in streams:
            identifier = "{} | {} | {} Channels | {} Hz" \
                         "".format(s.name, s.source_id, s.n_channels, s.sfreq)
            self._combobox_streamname.addItem(identifier)

        # Connect slot eeg stream
        self._checkbox_connect_lsl.clicked.connect(self._connect_eeg_stream)
        self._checkbox_display_eegpsd.clicked.connect(self._create_eegpsd_display_tab)

    def _reset(self):
        """
        Resets the values after a stream disconnect, also removes the tab
        :return: void
        """
        self._streaming = False
        self._sampling_rate = 0  # sampling rate of eeg data
        self._samples = 0
        self._fft_resolution = 0
        self._channels = 1
        self._history_length = 600.0
        self._eeg_data = []
        self._eeg_lock.release()
        self._eeg_times = []
        self._fft_data = []
        self._fft_lock.release()
        self._fft_freqs = []
        self._bnd_data = []
        self._bnd_lock.release()
        self._hst_data = []
        self._hst_lock.release()
        self._eeg_stream = None
        self._clc_stream = None
        self._checkbox_connect_lsl.setText("Click to connect")

    def _connect_eeg_stream(self):
        """
        Connects a single EEG stream
        Add the stream 'name' to the array of streams and starts pulling data
        :return: void
        """

        # if we're connecting
        if self._checkbox_connect_lsl.isChecked():

            self._name = self._lineedit_name.text()

            # first resolve an EEG stream on the lab network
            print("Finding an LSL streams.")
            streams = resolve_streams()

            # create a new inlet to read from the stream
            for s in streams:
                s_name, s_id, s_channels, s_rate = self._combobox_streamname.currentText().split(' | ')
                if s.source_id == s_id:

                    # set gui info
                    self._channels = s.n_channels
                    self._sampling_rate = s.sfreq
                    self._samples = int(self._data_seconds * self._sampling_rate)
                    self._eeg_stream = Stream(self._data_seconds, name=s_name, stype=s.stype, source_id=s.source_id)
                    self._checkbox_connect_lsl.setText("Connected")
                    print("Connected to LSL stream")

                    # create display tab
                    self._streaming = True

                    # start data reading thread
                    thstr = threading.Thread(target=self._read_lsl)
                    thstr.start()

                    # start analysis thread
                    thanal = threading.Thread(target=self._analyse_psd)
                    thanal.start()

                    # enable checkbox
                    self._checkbox_display_eegpsd.setEnabled(True)

        # if we're disconnecting
        else:
            self._reset()
            self._checkbox_display_eegpsd.setEnabled(False)
            self._checkbox_display_eegpsd.setChecked(False)
            self._create_eegpsd_display_tab()

    def _create_eegpsd_display_tab(self):

        if self._checkbox_display_eegpsd.isChecked():

            # create widgets
            self._displaytab = QtWidgets.QWidget()
            self._displaylayout = QtWidgets.QGridLayout(self._displaytab)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
            self._displaytab.setSizePolicy(sizePolicy)
            self._parent_tabwidget.addTab(self._displaytab, "")
            self._parent_tabwidget.setTabText(self._parent_tabwidget.indexOf(self._displaytab), self._name)

            # channel names
            #c_names = ['Fp1', 'Fpz', 'Fp2', 'AFz', 'F7',  'F3',  'Fz', 'F4',
            #           'F8',  'FC5', 'FC1', 'FC2', 'FC6', 'T7',  'C3', 'Cz',
            #           'C4',  'T8',  'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
            #           'Pz',  'P4',  'P8', 'POz', 'O1', 'Oz', 'O2']
            c_names = ['C{}'.format(n+1) for n in range(0, self._channels)]

            # first eeg plot
            figure = plt.figure()
            self._eeg_canvas = FigureCanvasQTAgg(figure)
            self._eeg_axes = figure.add_subplot(111)
            self._displaylayout.addWidget(self._eeg_canvas, 0, 0, 1, 1)
            self._eeg_axes.set_ylim(bottom=0.0, top=self._channels + 2)
            self._eeg_axes.set_xticks([])
            self._eeg_axes.set_yticks(ticks=np.arange(1, self._channels + 1), labels=c_names)
            self._eeg_axes.set_title('{} -- EEG over {:0.1f} Seconds'.format(self._name, self._data_seconds))

            # first psd plot
            figure = plt.figure()
            self._fft_canvas = FigureCanvasQTAgg(figure)
            self._fft_axes = figure.add_subplot(111)
            self._displaylayout.addWidget(self._fft_canvas, 0, 1, 1, 1)
            self._fft_axes.set_ylim(bottom=0.0, top=self._channels + 2)
            self._fft_axes.set_xscale('log')
            self._fft_axes.set_yticks(ticks=np.arange(1, self._channels + 1), labels=c_names)
            self._fft_axes.set_title('Current PSD')

            # bandpass history plot
            figure = plt.figure()
            self._hst_canvas = FigureCanvasQTAgg(figure)
            self._hst_axes = figure.add_subplot(111)
            self._displaylayout.addWidget(self._hst_canvas, 1, 0, 1, 1)
            self._hst_axes.set_ylim([-0.1, 5.1])
            self._hst_axes.set_xticks([])
            self._hst_axes.set_yticks([0, 1, 2, 3, 4], ['δ', 'θ', 'α', 'β', 'γ'])
            self._hst_axes.set_title('{} -- PSD History over {} minutes'.format(self._name, self._history_length/60.0))

            # bandpass bar graph
            figure = plt.figure()
            self._bnd_canvas = FigureCanvasQTAgg(figure)
            self._bnd_axes = figure.add_subplot(111)
            self._displaylayout.addWidget(self._bnd_canvas, 1, 1, 1, 1)
            self._bnd_axes.set_ylim([0.0, 1.1])
            self._bnd_axes.set_xticks([1, 2, 3, 4, 5], ['δ', 'θ', 'α', 'β', 'γ'])
            self._bnd_axes.set_yticks([])
            self._bnd_axes.set_title('Frequency Band Power'.format(self._data_seconds))

            self._displaylayout.setColumnStretch(0, 3)
            self._displaylayout.setColumnStretch(1, 1)
            self._displaylayout.setRowStretch(0, 2)
            self._displaylayout.setRowStretch(1, 1)

            # start display thread
            time.sleep(1)
            thdsp = threading.Thread(target=self._display_eeg_psd)
            thdsp.start()

        else:
            self._parent_tabwidget.removeTab(self._parent_tabwidget.indexOf(self._displaytab))

    def _display_eeg_psd(self):

        while not len(self._eeg_data)\
                or not len(self._fft_data)\
                or not len(self._bnd_data)\
                or not len(self._hst_data):
            time.sleep(1.0)

        # eeg + fft
        eeg_lines = self._eeg_axes.plot(self._eeg_data.T)  # the last channel in the simulator has the alpha intensity
        fft_lines = self._fft_axes.plot(self._fft_freqs, self._fft_data.T)
        bnd_bars = self._bnd_axes.bar([1, 2, 3, 4, 5], self._bnd_data)
        hst_lines = self._hst_axes.plot(self._hst_data.T)

        # set rainbow colours for eeg and fft
        for c in range(0, len(eeg_lines)):
            a = c/(self._channels-1.0)
            colour = (0.7*(1 - a), 0.5*(1.0 - 2.0*abs(a - 0.5)), 0.7*a)
            eeg_lines[c].set_color(color=colour)
            eeg_lines[c].set_linewidth(0.4)
            fft_lines[c].set_color(color=colour)
            fft_lines[c].set_linewidth(0.4)

        # set rainbow colours for frequency bands
        for n in range(0, 5):
            a = n/4.0
            colour = (0.7*(1 - a), 0.5*(1.0 - 2.0*abs(a - 0.5)), 0.7*a)
            bnd_bars[n].set(color=colour)
            hst_lines[n].set_color(color=colour)
            hst_lines[n].set_linewidth(0.5)

        self._eeg_info.setEnabled(True)
        while self._streaming:
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
                #self._eeg_info.setText("{:.1f} µV - {:.1f} µV".format(eeg_min*1e6, eeg_max*1e6))
                print('d', end="")
            except Exception as e:
                print(e)
                time.sleep(0.5)

        # finished
        self._eeg_info.setEnabled(False)

    def _read_lsl(self):
        """
        Read data into buffer a, then call a new thread
        :return:
        """

        # init data buffers
        self._eeg_stream.connect(acquisition_delay=0.1, processing_flags="all")
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
        self._lsl_info.setEnabled(True)
        while self._streaming:
            with self._eeg_lock:
                self._eeg_data, ts = self._eeg_stream.get_data()
            #self._lsl_info.setText("LSL Time {:0.1f}".format(ts[-1]))
            print('r', end="")
            time.sleep(0.1)

        # finished
        self._lsl_info.setEnabled(False)

    def _analyse_psd(self):
        """
        Read data into buffer a, then call a new thread
        :return:
        """

        while not len(self._eeg_data):
            time.sleep(0.5)

        self._fft_freqs = np.fft.rfftfreq(self._samples, d=1.0 / self._sampling_rate, device=None)
        self._fft_resolution = self._fft_freqs[1]
        delta = abs(self._fft_freqs - 4.0).argmin()      # delta: 0-4 Hz
        theta = abs(self._fft_freqs - 7.0).argmin()      # theta: 4-7 Hz
        alpha = abs(self._fft_freqs - 12.0).argmin()     # alpha: 8-12 Hz
        beta = abs(self._fft_freqs - 30.0).argmin()      # beta: 13-30 Hz
        gamma = abs(self._fft_freqs - 50.0).argmin()    # gamma: 30-100 Hz
        bins = [delta, theta, alpha, beta, gamma]
        widths = [delta, theta-delta, alpha-theta, beta-alpha, gamma-beta]

        is_relative = True

        # start streaming loop
        self._bnd_info.setEnabled(True)
        while self._streaming:
            try:
                with self._fft_lock:
                    with self._eeg_lock:
                        self._fft_data = np.fft.rfft(self._eeg_data, axis=1)
                    self._fft_data = np.abs(self._fft_data)**2
                    all_channels = self._fft_data.sum(axis=0)[1:]/self._channels     # sum of fft over all channels, excluding DC
                with self._bnd_lock:
                    self._bnd_data = np.array([a[0].sum()*self._fft_resolution for a in np.split(all_channels, bins)[:5]])
                    if is_relative:
                        self._bnd_data = self._bnd_data / self._bnd_data.sum() # relative power
                    with self._hst_lock:
                        self._hst_data[:, :-1] = self._hst_data[:, 1:]
                        self._hst_data[:, -1] = self._bnd_data
                    #if is_relative:
                    #    self._bnd_info.setText("{:0.1f} | {:0.1f} | {:0.1f} | {:0.1f} | {:0.1f}".format(*self._bnd_data))
                    #else:
                    #    self._bnd_info.setText("{:0.1f} | {:0.1f} | {:0.1f} | {:0.1f} | {:0.1f}".format(
                    #        self._bnd_data[0] * 1e6, self._bnd_data[1] * 1e6, self._bnd_data[2] * 1e6,
                    #        self._bnd_data[3] * 1e6,
                    #    self._bnd_data[4] * 1e6))
                    print('a', end="")
            except Exception as e:
                print(e)
                time.sleep(0.5)
            time.sleep(0.2)

        # finished
        self._bnd_info.setEnabled(False)


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
