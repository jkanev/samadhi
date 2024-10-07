
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
    _streaming = True  # whether we're streaming currently
    _data_a = []       # two 500 ms buffers, will be filled alternatingly
    _data_b = []
    _eeg_data = False    # pointer to the buffer that has just been filled, either data_a or data_b
    _fft_data = False    # buffer containing the fft
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
                samples = int(2.0 * srate)
                self._eeg_stream = StreamInlet(s)
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

        # first fft plot
        figure = plt.figure()
        self._fft_canvas = FigureCanvasQTAgg(figure)
        self._fft_axes = figure.add_subplot(111)
        self._displaylayout.addWidget(self._fft_canvas, 0, 1, 1, 1)
        self._fft_axes.set_ylim(bottom=-10e-6, top=35e-3)
        self._fft_axes.set_xticks([])
        self._fft_axes.set_yticks(ticks=fft_ticks, labels=c_names)

        self._displaylayout.setColumnStretch(0, 3)
        self._displaylayout.setColumnStretch(1, 1)

    def _display_continuous(self):

        # eeg + fft
        eeg_lines = self._eeg_axes.plot(self._eeg_data[:, :-1])  # the last channel in the simulator has the alpha intensity
        fft_lines = self._fft_axes.plot(self._fft_data[:50, :-1])

        # set rainbow colours
        for c in range(0, len(eeg_lines)):
            eeg_lines[c].set_color(color=(1-c/32.0, 0.5*c/32.0, c/32.0))
            eeg_lines[c].set_linewidth(1)
            fft_lines[c].set_color(color=(1-c/32.0, 0.5*c/32.0, c/32.0))
            fft_lines[c].set_linewidth(1)

        while self._streaming:
            time.sleep(0.5)
            for c in range(0, len(eeg_lines)):
                eeg_lines[c].set_ydata((self._eeg_data.T)[c] + float(32.0 - c) * 20e-6)
                fft_lines[c].set_ydata((self._fft_data.T)[c][:50] + float(32.0 - c) * 10e-4)
            self._eeg_canvas.draw()
            self._fft_canvas.draw()

    def _read_data(self, samples, channels):
        """
        Read data into buffer a, then call a new thread
        :return:
        """

        # init buffers
        self._data_a = np.zeros((samples, channels))
        self._data_b = np.zeros((samples, channels))
        self._eeg_data = self._data_b

        # start streaming loop
        while self._streaming:

            # read into buffer a
            for s in range(0, samples):
                self._data_a[s], timestamp = self._eeg_stream.pull_sample()
            self._eeg_data = self._data_a

            # read into buffer b
            for s in range(0, samples):
                self._data_b[s], timestamp = self._eeg_stream.pull_sample()
            self._eeg_data = self._data_b

    def _analyse_data(self):
        """
        Read data into buffer a, then call a new thread
        :return:
        """

        # start streaming loop
        while self._streaming:
            self._fft_data = np.fft.rfft(self._eeg_data, axis=0)
            self._fft_data = np.abs(self._fft_data)




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
