
# -*- coding:utf-8 -*-

#!/usr/bin/python3

import sys
from mainwindow import *
from PyQt6 import QtCore, QtGui, QtWidgets
from pylsl import StreamInfo, StreamInlet, resolve_stream
import threading
import numpy as np
import time
import mne
from mne_lsl.stream import StreamLSL as Stream
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
    _channels = 31       # number of channels in the data
    _eeg_data = False    # pointer to the buffer that has just been filled, either data_a or data_b
    _fft_data = False    # buffer containing the fft
    _bnd_data = False    # frequency band data: band frequencies
    _hst_data = False    # frequency band data: ring buffer that is constantly rooled
    _eeg_stream = None   # the lsl eeg input stream inlet, if in eeg mode
    _clc_stream = None   # the lsl calculation output stream outlet, if in calculation mode

    # state related
    _calculate = False   # whether to do data calculations on the eeg_data buffer
    _display = False     # whether to display data
    _send = False        # whether to send data on

    # main mind controls
    _lineedit_name = False
    _parent_tabwidget = False
    _combobox_streamname = False
    _combobox_streamid = False
    _checkbox_connect = False
    _label_hosttext = False
    _label_channelstext = False
    _label_samplingratetext = False

    # eeg display research controls → put into new class soon
    _displaylayout = False
    _displaytab = False
    _eeg_axes = False
    _eeg_canvas = False
    _fft_axes = False
    _fft_canvas = False
    _bnd_axes = False
    _bnd_canvas = False
    _hst_axes = False
    _hst_canvas = False

    def __init__(self, lineedit_name, combobox_streamname, combobox_streamid, checkbox_connect,
                       label_hosttext, label_channelstext, label_samplingratetext, parent_tabwidget):
        self._lineedit_name = lineedit_name
        self._combobox_streamname = combobox_streamname
        self._combobox_streamid = combobox_streamid
        self._checkbox_connect = checkbox_connect
        self._label_hosttext = label_hosttext
        self._label_channelstext = label_channelstext
        self._label_samplingratetext = label_samplingratetext
        self._combobox_streamname.addItem("EEG")
        self._parent_tabwidget = parent_tabwidget

        # Fill values
        streams = resolve_stream("type", "EEG")
        for s in streams:
            self._combobox_streamid.addItem(s.source_id())

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
        streams = resolve_stream("type", "EEG")

        # create a new inlet to read from the stream
        for s in streams:
            if s.source_id() == self._combobox_streamid.currentText():

                # set gui info
                channels = s.channel_count()
                srate = s.nominal_srate()
                samples = int(self._data_seconds * srate)
                self._sampling_rate = srate
                self._eeg_stream = Stream(self._data_seconds, name=self._combobox_streamname.currentText(), stype="EEG",
                                          source_id=self._combobox_streamid.currentText())
                self._label_hosttext.setText(s.hostname())
                self._label_channelstext.setText("{}".format(channels))
                self._label_samplingratetext.setText("{:0.1f}".format(srate))
                self._checkbox_connect.setText("connected")

                # create display tab
                self._streaming = True
                self._create_display_tab()

                # start data reading thread
                thstr = threading.Thread(target=self._read_data, args=(samples, channels))
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
        c_names = ['Fp1', 'Fpz', 'Fp2', 'AFz', 'F7',  'F3',  'Fz', 'F4',
                   'F8',  'FC5', 'FC1', 'FC2', 'FC6', 'T7',  'C3', 'Cz',
                   'C4',  'T8',  'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
                   'Pz',  'P4',  'P8', 'POz', 'O1', 'Oz', 'O2']

        eeg_ticks = [float(32.0 - c) * 20e-6 for c in range(0, 31)]
        fft_ticks = [float(32.0 - c) * 10e-4 for c in range(0, 31)]

        # first eeg plot
        figure = plt.figure()
        self._eeg_canvas = FigureCanvasQTAgg(figure)
        self._eeg_axes = figure.add_subplot(111)
        self._displaylayout.addWidget(self._eeg_canvas, 0, 0, 1, 1)
        self._eeg_axes.set_ylim(bottom=-20e-6, top=7e-4)
        self._eeg_axes.set_xticks([])
        self._eeg_axes.set_yticks(ticks=eeg_ticks, labels=c_names)
        self._eeg_axes.set_title('EEG over {:0.1f} Seconds'.format(self._data_seconds))

        # first fft plot
        figure = plt.figure()
        self._fft_canvas = FigureCanvasQTAgg(figure)
        self._fft_axes = figure.add_subplot(111)
        self._displaylayout.addWidget(self._fft_canvas, 0, 1, 1, 1)
        self._fft_axes.set_ylim(bottom=-10e-6, top=35e-3)
        self._fft_axes.set_xticks([])
        self._fft_axes.set_yticks(ticks=fft_ticks, labels=c_names)
        self._fft_axes.set_title('FFT Amplitude over {:0.1f} Seconds'.format(self._data_seconds))

        # bandpass history plot
        figure = plt.figure()
        self._hst_canvas = FigureCanvasQTAgg(figure)
        self._hst_axes = figure.add_subplot(111)
        self._displaylayout.addWidget(self._hst_canvas, 1, 0, 1, 1)
        self._hst_axes.set_ylim([0.0, 0.004])
        self._hst_axes.set_xticks([])
        self._hst_axes.set_yticks([])
        self._hst_axes.set_title('Band History'.format(self._data_seconds))

        # bandpass bar graph
        figure = plt.figure()
        self._bnd_canvas = FigureCanvasQTAgg(figure)
        self._bnd_axes = figure.add_subplot(111)
        self._displaylayout.addWidget(self._bnd_canvas, 1, 1, 1, 1)
        self._bnd_axes.set_ylim([0.0, 0.004])
        self._bnd_axes.set_xticks([])
        self._bnd_axes.set_yticks([])
        self._bnd_axes.set_title('Frequency Band Average'.format(self._data_seconds))

        self._displaylayout.setColumnStretch(0, 3)
        self._displaylayout.setColumnStretch(1, 1)
        self._displaylayout.setRowStretch(0, 2)
        self._displaylayout.setRowStretch(1, 1)

    def _display_continuous(self):

        time.sleep(1.0)

        # eeg + fft
        eeg_lines = self._eeg_axes.plot(self._eeg_data[:-1, :].T)  # the last channel in the simulator has the alpha intensity
        fft_lines = self._fft_axes.plot(self._fft_data[:, :120].T)
        bnd_bars = self._bnd_axes.bar([1, 2, 3, 4, 5], self._bnd_data)
        hst_lines = self._hst_axes.plot(self._hst_data.T)

        # set rainbow colours
        for c in range(0, len(eeg_lines)):
            eeg_lines[c].set_color(color=(1-c/32.0, 0.5*c/32.0, c/32.0))
            eeg_lines[c].set_linewidth(1)
            fft_lines[c].set_color(color=(1-c/32.0, 0.5*c/32.0, c/32.0))
            fft_lines[c].set_linewidth(1)

        while self._streaming:
            try:
                time.sleep(0.1)
                for c in range(0, len(eeg_lines)):
                    eeg_lines[c].set_ydata(self._eeg_data[c] + float(32.0 - c) * 20e-6)
                    fft_lines[c].set_ydata(self._fft_data[c][:120] + float(32.0 - c) * 10e-4)
                for b in range(0, len(hst_lines)):
                    bnd_bars[b].set_height(self._bnd_data[b])
                    hst_lines[b].set_ydata(self._hst_data[b])
                self._eeg_canvas.draw()
                self._fft_canvas.draw()
                self._bnd_canvas.draw()
                self._hst_canvas.draw()
            except Exception as e:
                print(e)
                time.sleep(0.5)

    def _read_data(self, samples, channels):
        """
        Read data into buffer a, then call a new thread
        :return:
        """

        self._eeg_stream.connect(acquisition_delay=0.1, processing_flags="all")
        self._eeg_stream.get_data()  # reset the number of new samples after the filter is applied
        self._eeg_data = np.zeros((32, 2500))
        self._fft_data = np.zeros((32, 1250))
        self._bnd_data = np.zeros(5)
        self._hst_data = np.zeros((5, 2500))

        # start streaming loop
        while self._streaming:
            time.sleep(0.1)
            self._eeg_data, ts = self._eeg_stream.get_data()

    def _analyse_data(self):
        """
        Read data into buffer a, then call a new thread
        :return:
        """
        delta = 4.0      # delta: 0-4 Hz
        theta = 7.0      # theta: 4-7 Hz
        alpha = 12.0     # alpha: 8-12 Hz
        beta = 30.0      # beta: 13-30 Hz
        gamma = 100.0    # gamma: 30-100 Hz
        bins = [int(delta * 2000.0/self._sampling_rate),
                int(theta * 2000.0/self._sampling_rate),
                int(alpha * 2000.0/self._sampling_rate),
                int(beta * 2000.0/self._sampling_rate),
                int(gamma * 2000.0/self._sampling_rate)]
        widths = [delta, theta-delta, alpha-theta, beta-alpha, gamma-beta]

        # start streaming loop
        while self._streaming:
            try:
                time.sleep(0.2)
                self._fft_data = np.fft.rfft(self._eeg_data[:-1, :], axis=1)
                self._fft_data = np.abs(self._fft_data)
                all_channels = self._fft_data.sum(axis=0)[1:]/31.0     # sum of fft over all channels, excluding DC
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
        self.add_mind(self.lineEditName, self.comboBoxStreamName, self.comboBoxStreamId, self.checkBoxConnect,
                      self.labelHostText, self.labelChannelsText, self.labelSamplingRateText)


    def add_mind(self, lineedit_name, combobox_streamname, combobox_streamid, checkbox_connect,
                       label_hosttext, label_channelstext, label_samplingratetext):
        self._minds.append(Mind(lineedit_name, combobox_streamname, combobox_streamid, checkbox_connect,
                       label_hosttext, label_channelstext, label_samplingratetext, self.tabWidget))

class Samadhi:

    def __init__(self, filename=None):
        app = QtWidgets.QApplication(sys.argv)

        # start main window
        main_window = SamadhiWindow()

        # run
        main_window.show()
        sys.exit(app.exec())
