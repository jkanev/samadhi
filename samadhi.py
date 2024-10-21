
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
    _streaming = True    # whether we're streaming currently
    _data_seconds = 5.0  # how much data do we have in the _eeg_data array
    _sampling_rate = 0   # sampling rate of eeg data
    _samples = 0         # seconds times sampling rate
    _fft_resolution = 0  # resolution (distance of one FFT bin to the next)
    _channels = 1        # number of channels in the data
    _history_length = 600.0   # length of history buffer in seconds
    _eeg_data = []       # pointer to the buffer that has just been filled, either data_a or data_b
    _eeg_times = []      # buffer containing eeg time stamps
    _fft_data = []       # buffer containing the fft
    _fft_freqs = []      # buffer containing fft frequencies
    _bnd_data = []       # frequency band data: band frequencies
    _hst_data = []       # frequency band data: ring buffer that is constantly rooled
    _eeg_stream = None   # the lsl eeg input stream inlet, if in eeg mode
    _clc_stream = None   # the lsl calculation output stream outlet, if in calculation mode

    # normalisation
    _bnd_max = []       # maximum values of bands, calculated from data d(t) by x(t+1) = max(0.99*x(t), d(t))
    _bnd_min = []       # minimum values of bands, calculated from data d(t) by x(t+1) = min(1.01*x(t), d(t))
    _bnd_mid = []       # the middle between max and min

    # state related
    _calculate = False   # whether to do data calculations on the eeg_data buffer
    _display = False     # whether to display data
    _send = False        # whether to send data on

    # main mind controls
    _combobox_streamname = False
    _lineedit_name = False
    _checkbox_connect = False
    _checkbox_analyse = False
    _checkbox_display = False
    _checkbox_visualisation = False
    _checkbox_streamanalysis = False
    _parent_tabwidget = False

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

    def __init__(self, combobox_streamname, lineedit_name, checkbox_connect, checkbox_analyse,
                       checkbox_display, checkbox_visualisation, checkbox_streamanalysis, parent_tabwidget):
        self._combobox_streamname = combobox_streamname
        self._lineedit_name = lineedit_name
        self._checkbox_connect = checkbox_connect
        self._checkbox_analyse = checkbox_analyse
        self._checkbox_display = checkbox_display
        self._checkbox_visualisation = checkbox_visualisation
        self._checkbox_streamanalysis = checkbox_streamanalysis
        self._parent_tabwidget = parent_tabwidget

        # Fill values
        streams = resolve_streams(timeout=1)
        for s in streams:
            identifier = "{} | {} | {} Channels | {} Hz" \
                         "".format(s.name, s.source_id, s.n_channels, s.sfreq)
            self._combobox_streamname.addItem(identifier)

        # Connect slot eeg stream
        self._checkbox_connect.clicked.connect(self.connect_eeg_stream)

    def connect_eeg_stream(self):
        """
        Connects a single EEG stream
        Add the stream 'name' to the array of streams and starts pulling data
        :return: void
        """
        print("connecting EEG stream...")
        self._name = self._lineedit_name.text()

        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
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
                self._checkbox_connect.setText("Connected")

                # create display tab
                self._streaming = True
                self._create_display_tab()

                # start data reading thread
                thstr = threading.Thread(target=self._read_data)
                thstr.start()

                # start analysis thread
                thanal = threading.Thread(target=self._analyse_data)
                thanal.start()

                # start display thread
                time.sleep(1)
                thdsp = threading.Thread(target=self._display_continuous)
                thdsp.start()

    def _create_display_tab(self):

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

        # first fft plot
        figure = plt.figure()
        self._fft_canvas = FigureCanvasQTAgg(figure)
        self._fft_axes = figure.add_subplot(111)
        self._displaylayout.addWidget(self._fft_canvas, 0, 1, 1, 1)
        self._fft_axes.set_ylim(bottom=0.0, top=self._channels + 2)
        self._fft_axes.set_xscale('log')
        self._fft_axes.set_yticks(ticks=np.arange(1, self._channels + 1), labels=c_names)
        self._fft_axes.set_title('Current FFT Amplitude')

        # bandpass history plot
        figure = plt.figure()
        self._hst_canvas = FigureCanvasQTAgg(figure)
        self._hst_axes = figure.add_subplot(111)
        self._displaylayout.addWidget(self._hst_canvas, 1, 0, 1, 1)
        self._hst_axes.set_ylim([0.0, 1.1])
        self._hst_axes.set_xticks([])
        self._hst_axes.set_yticks([])
        self._hst_axes.set_title('{} -- Band History over {} minutes'.format(self._name, self._history_length/60.0))

        # bandpass bar graph
        figure = plt.figure()
        self._bnd_canvas = FigureCanvasQTAgg(figure)
        self._bnd_axes = figure.add_subplot(111)
        self._displaylayout.addWidget(self._bnd_canvas, 1, 1, 1, 1)
        self._bnd_axes.set_ylim([0.0, 1.1])
        self._bnd_axes.set_xticks([1, 2, 3, 4, 5], ['δ', 'θ', 'α', 'β', 'γ'])
        self._bnd_axes.set_yticks([])
        self._bnd_axes.set_title('Frequency Band Average'.format(self._data_seconds))

        self._displaylayout.setColumnStretch(0, 3)
        self._displaylayout.setColumnStretch(1, 1)
        self._displaylayout.setRowStretch(0, 2)
        self._displaylayout.setRowStretch(1, 1)

    def _display_continuous(self):

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
            eeg_lines[c].set_linewidth(1)
            fft_lines[c].set_color(color=colour)
            fft_lines[c].set_linewidth(1)

        # set rainbow colours for frequency bands
        for n in range(0, 5):
            a = n/4.0
            colour = (0.7*(1 - a), 0.5*(1.0 - 2.0*abs(a - 0.5)), 0.7*a)
            bnd_bars[n].set(color=colour)
            hst_lines[n].set_color(color=colour)

        while self._streaming:
            try:
                time.sleep(0.1)
                self._eeg_channel_height = 0.5*(self._eeg_data.max() - self._eeg_data.min())
                self._fft_channel_height = 0.5*(self._fft_data.max() - self._fft_data.min())
                hst_height = self._hst_data.max() - self._hst_data.min()
                hst_height = hst_height or 1.0
                print(hst_height)
                for c in range(0, len(eeg_lines)):
                    eeg_lines[c].set_ydata(self._eeg_data[c]/self._eeg_channel_height + float(self._channels - c))
                    fft_lines[c].set_ydata(self._fft_data[c]/self._fft_channel_height + float(self._channels - c))
                for b in range(0, len(hst_lines)):
                    bnd_bars[b].set_height(self._bnd_data[b] / hst_height)
                    hst_lines[b].set_ydata(self._hst_data[b] / hst_height)
                self._eeg_canvas.draw()
                self._fft_canvas.draw()
                self._bnd_canvas.draw()
                self._hst_canvas.draw()
            except Exception as e:
                print(e)
                time.sleep(0.5)

    def _read_data(self):
        """
        Read data into buffer a, then call a new thread
        :return:
        """

        # init data buffers
        self._eeg_stream.connect(acquisition_delay=0.1, processing_flags="all")
        self._eeg_stream.get_data()  # reset the number of new samples after the filter is applied
        self._eeg_data = np.zeros((self._channels, self._samples))
        self._fft_data = np.zeros((self._channels, int(self._samples/2)))
        self._bnd_data = np.zeros(5)
        self._hst_data = np.zeros((5, int(self._history_length * 5.0)))  # history length * update rate of the analysis thread

        # start streaming loop
        while self._streaming:
            time.sleep(0.1)
            self._eeg_data, ts = self._eeg_stream.get_data()

    def _analyse_data(self):
        """
        Read data into buffer a, then call a new thread
        :return:
        """

        while not len(self._eeg_data):
            time.sleep(0.5)

        self._fft_freqs = np.fft.rfftfreq(self._samples, d=1.0 / self._sampling_rate, device=None)
        delta = abs(self._fft_freqs - 4.0).argmin()      # delta: 0-4 Hz
        theta = abs(self._fft_freqs - 7.0).argmin()      # theta: 4-7 Hz
        alpha = abs(self._fft_freqs - 12.0).argmin()     # alpha: 8-12 Hz
        beta = abs(self._fft_freqs - 30.0).argmin()      # beta: 13-30 Hz
        gamma = abs(self._fft_freqs - 50.0).argmin()    # gamma: 30-100 Hz
        bins = [delta, theta, alpha, beta, gamma]
        widths = [delta, theta-delta, alpha-theta, beta-alpha, gamma-beta]

        # start streaming loop
        while self._streaming:
            try:
                time.sleep(0.2)
                self._fft_data = np.fft.rfft(self._eeg_data, axis=1)
                self._fft_data = np.abs(self._fft_data)
                all_channels = self._fft_data.sum(axis=0)[1:]/self._channels     # sum of fft over all channels, excluding DC
                self._bnd_data = [a[0].sum()/a[1] for a in zip(np.split(all_channels, bins)[:5], widths)]
                self._hst_data[:, :-1] = self._hst_data[:, 1:]
                self._hst_data[:, -1] = self._bnd_data
            except Exception as e:
                print(e)
                time.sleep(0.5)


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
        self.add_mind(self.comboBoxStreamName01, self.lineEditName01, self.checkBoxConnect01, self.checkBoxAnalyse01,
                      self.checkBoxDisplay01, self.checkBoxVisualisation01, self.checkBoxStreamAnalysis01)


    def add_mind(self, combobox_streamname, lineedit_name, checkbox_connect, checkbox_analyse,
                       checkbox_display, checkbox_visualisation, checkbox_streamanalysis):
        self._minds.append(Mind(combobox_streamname, lineedit_name, checkbox_connect, checkbox_analyse,
                       checkbox_display, checkbox_visualisation, checkbox_streamanalysis, self.tabWidget))

class Samadhi:

    def __init__(self, filename=None):
        app = QtWidgets.QApplication(sys.argv)

        # start main window
        main_window = SamadhiWindow()

        # run
        main_window.show()
        sys.exit(app.exec())
