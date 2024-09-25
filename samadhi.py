
# -*- coding:utf-8 -*-

#!/usr/bin/python3

import sys
#from mainwindow import *
#from PyQt6 import QtCore, QtGui, QtWidgets
from pylsl import StreamInfo, StreamInlet, resolve_stream
import xmltodict



class SamadhiWindow():     #QtWidgets.QMainWindow, Ui_MainWindow):
    """
    Implements the main part of the GUI.
    """

    _eeg_streams = []
    _lsl_streams = []

    def __init__(self, filename=None):
        """
        Initialise the application, connect all button signals to application functions, load theme and last file
        """

        print("starting Samadhi...")

        # initialise main window
        # super().__init__()
        # self.setupUi(self)

        # connect all button signals
        print("setting up GUI...")
        # self.pushButtonNew.clicked.connect(lambda: self.createNode("child", False))
        # ...

        # init application settings
        #print("loading system settings...")
        #self.settings = QtCore.QSettings('FreeSoftware', 'TreeTime')

        self.connect_eeg_stream()

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
        inlet = StreamInlet(streams[0])
        info_xml = inlet.info().as_xml()
        info_dict = xmltodict.parse(info_xml)['info']
        print("... connected to stream '{} / {}', host name {}, {} channels, {} Hz sampling rate:"
              "".format(info_dict['name'], info_dict['source_id'], info_dict['hostname'], info_dict['channel_count'],
                        info_dict['nominal_srate']))

        count = 0
        while True:
            # get a new sample (you can also omit the timestamp part if you're not
            # interested in it)
            sample, timestamp = inlet.pull_sample()
            print('{} samples'.format(count), end='\r')
            count += 1

class Samadhi:

    def __init__(self, filename=None):
        #app = QtWidgets.QApplication(sys.argv)

        # start main window
        main_window = SamadhiWindow()

        # run
        #main_window.show()
        #sys.exit(app.exec())
