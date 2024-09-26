
# -*- coding:utf-8 -*-

#!/usr/bin/python3

import sys
from mainwindow import *
from PyQt6 import QtCore, QtGui, QtWidgets
from pylsl import StreamInfo, StreamInlet, resolve_stream
import xmltodict


class Mind:
    """
    Implements a complete data and calculation set of a single human.
    """

    _name = ""           # the person's name
    _eeg_stream = None   # the lsl eeg input stream, if in eeg mode
    _clc_stream = None   # the lsl calculation output stream, if in calculation mode
    _eeg_data = None     # the eeg input stream, if applicable
    _analysis = None     # the calculated measurements, if applicable, either from input LSL, or calculated
    _calculate = False   # whether to do data calculations on the eeg_data buffer
    _display = False     # whether to display data
    _send = False        # whether to send data on
    _combobox_streamname = False
    _combobox_streamid = False
    _checkbox_connect = False
    _label_hosttext = False
    _label_channelstext = False
    _label_samplingratetext = False

    def __init__(self, combobox_streamname, combobox_streamid, checkbox_connect,
                       label_hosttext, label_channelstext, label_samplingratetext):
        self._combobox_streamname = combobox_streamname
        self._combobox_streamid = combobox_streamid
        self._checkbox_connect = checkbox_connect
        self._label_hosttext = label_hosttext
        self._label_channelstext = label_channelstext
        self._label_samplingratetext = label_samplingratetext
        self._combobox_streamname.addItem("EEG")

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

        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = resolve_stream("type", "EEG")

        # create a new inlet to read from the stream
        for s in streams:
            if s.source_id() == self._combobox_streamid.currentText():
                self._eeg_stream = StreamInlet(s)
                self._label_hosttext.setText(s.hostname())
                self._label_channelstext.setText("{}".format(s.channel_count()))
                self._label_samplingratetext.setText("{:0.1f}".format(s.nominal_srate()))
                self._checkbox_connect.setText("connected")

                # This must go into an extra thread
                #count = 0
                #while True:
                #    # get a new sample (you can also omit the timestamp part if you're not
                #    # interested in it)
                #    sample, timestamp = self._eeg_stream.pull_sample()
                #    print('{} samples'.format(count), end='\r')
                #    count += 1

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
        self.add_mind(self.comboBoxStreamName, self.comboBoxStreamId, self.checkBoxConnect,
                      self.labelHostText, self.labelChannelsText, self.labelSamplingRateText)
    def add_mind(self, combobox_streamname, combobox_streamid, checkbox_connect,
                       label_hosttext, label_channelstext, label_samplingratetext):
        self._minds.append(Mind(combobox_streamname, combobox_streamid, checkbox_connect,
                       label_hosttext, label_channelstext, label_samplingratetext))

class Samadhi:

    def __init__(self, filename=None):
        app = QtWidgets.QApplication(sys.argv)

        # start main window
        main_window = SamadhiWindow()

        # run
        main_window.show()
        sys.exit(app.exec())
