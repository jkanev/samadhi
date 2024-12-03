
# -*- coding:utf-8 -*-

#!/usr/bin/python3

import sys

from .mainwindow import *
from PyQt6 import QtCore, QtGui, QtWidgets, QtOpenGLWidgets
import threading
import numpy as np
import time
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.lsl import resolve_streams
from matplotlib import use as mpl_use
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, FigureManagerQT
import matplotlib.pyplot as plt
mpl_use("QtAgg")

class OpenGLDancingDots(QtOpenGLWidgets.QOpenGLWidget):

    _painter = None
    _points = False
    _x_numbers = False
    _y_numbers = False
    _r_numbers = False
    _phi_numbers = False
    _timer = False
    _pen_draw = QtGui.QPen(QtGui.QColor(80, 200, 220))
    _pen_erase = QtGui.QPen(QtGui.QColor(0, 0, 0))
    _brush_draw = QtGui.QBrush(QtGui.QColor(80, 200, 220))
    _brush_erase = QtGui.QBrush(QtGui.QColor(0, 0, 0))
    _function = False

    def __init__(self):
        super().__init__()
        self._painter = QtGui.QPainter(self)
        self._r_numbers = np.arange(0, 2.0*np.pi, 0.00079)
        self._phi_numbers = np.sin(self._r_numbers)
        self._x_numbers = np.zeros(self._r_numbers.shape)
        self._y_numbers = np.zeros(self._r_numbers.shape)
        self._points = [QtCore.QPoint(int(100*x), int(np.sin(x)*100)) for x in self._x_numbers]
        self._functions = self.context()
        self._counter = 0.0

    def start(self):
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.start(10)

    def paintEvent(self, e):

        size = min(e.rect().height(), e.rect().width())
        self._counter += 0.01
        center_x = e.rect().width()/2
        center_y = e.rect().height()/2
        self._phi_numbers = 0.1*np.sin((self._r_numbers+self._counter)*5.0)+0.2
        self._x_numbers = (self._phi_numbers * np.cos(self._r_numbers)) * size + center_x
        self._y_numbers = self._phi_numbers * np.sin(self._r_numbers) * size + center_y
        self._painter.begin(self)
        self._painter.setPen(self._pen_erase)
        self._painter.setBrush(self._brush_erase)
        self._painter.drawRect(e.rect())
        self._painter.setPen(self._pen_draw)
        self._painter.setBrush(self._brush_draw)
        for p in range(len(self._x_numbers)):
            #color = QtGui.QColor(p % 255, 2*p % 255, 3*p % 255)
            #self._painter.setPen(color)
            #self._painter.setBrush(color)
            self._painter.drawEllipse(int(self._x_numbers[p]),
                                      int(self._y_numbers[p]), 6, 6)
        self._painter.end()


class Mind:
    """
    Implements a complete data and calculation set of a single human.
    """

    # general
    _name = ""           # the person's name

    # data streaming related
    _streaming = False   # whether we're streaming currently
    _showing_eegpsd = False     # whether we're showing the eeg/psd tab
    _showing_ddots = False   # whether we're showing the dancing dots display
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
    _bnd_smoothing = 0.9   #
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
    _ddots_layout = False
    _ddots_tab = False
    _ddots_canvas = False
    _ddots_axes = False
    _ddots_ogl_wdg = False

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

        # Fill values
        streams = resolve_streams(timeout=1)
        for s in streams:
            identifier = "{} | {} | {} Channels | {} Hz" \
                         "".format(s.name, s.source_id, s.n_channels, s.sfreq)
            self._combobox_streamname.addItem(identifier)

        # Connect slot eeg stream
        self._checkbox_connect_lsl.clicked.connect(self._connect_eeg_stream)
        self._checkbox_display_eegpsd.clicked.connect(self._create_eegpsd_display_tab)
        self._checkbox_visualisation_ddots.clicked.connect(self._create_dancing_dots_display_opengl_tab)

    def __del__(self):
        self._connect_eeg_stream(False)

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
        self._checkbox_display_eegpsd.setEnabled(False)
        self._checkbox_display_eegpsd.setChecked(False)
        self._checkbox_visualisation_ddots.setEnabled(False)
        self._checkbox_visualisation_ddots.setChecked(False)
        self._lsl_label.setEnabled(False)
        self._checkbox_connect_lsl.setText("Click to connect")
        self._eeg_label.setEnabled(False)
        self._create_eegpsd_display_tab(False)
        self._create_dancing_dots_display_opengl_tab(False)

    def _connect_eeg_stream(self, connect):
        """
        Connects a single EEG stream
        Add the stream 'name' to the array of streams and starts pulling data
        :return: void
        """

        # if we're connecting
        if connect:

            # first resolve an EEG stream on the lab network
            print("Connecting to LSL stream... ")
            self._name = self._lineedit_name.text()
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

                    # start GUI update function (in same thread)
                    self._streaming = True
                    self._gui_timer = QtCore.QTimer()
                    self._gui_timer.timeout.connect(self._update_gui)
                    self._gui_timer.start(300)

                    # start data reading thread
                    thstr = threading.Thread(target=self._read_lsl)
                    thstr.start()

                    # start analysis thread
                    thanal = threading.Thread(target=self._analyse_psd)
                    thanal.start()

                    # enable checkbox
                    self._checkbox_display_eegpsd.setEnabled(True)
                    self._checkbox_visualisation_ddots.setEnabled(True)
                    print("... LSL stream connected.")

        # if we're disconnecting
        else:
            print("Disconnecting from LSL stream... ")
            self._reset()
            print("... LSL stream disconnected.")

    def _create_eegpsd_display_tab(self, create):

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
            c_names = ['C{}'.format(n+1) for n in range(0, self._channels)]

            # first eeg plot
            figure = plt.figure()
            self._eeg_canvas = FigureCanvasQTAgg(figure)
            self._eeg_axes = figure.add_subplot(111)
            self._eegpsd_layout.addWidget(self._eeg_canvas, 0, 0, 1, 1)
            self._eeg_axes.set_ylim(bottom=0.0, top=self._channels + 2)
            self._eeg_axes.set_xticks([])
            self._eeg_axes.set_yticks(ticks=np.arange(1, self._channels + 1), labels=c_names)
            self._eeg_axes.set_title('{} -- EEG over {:0.1f} Seconds'.format(self._name, self._data_seconds))

            # first psd plot
            figure = plt.figure()
            self._fft_canvas = FigureCanvasQTAgg(figure)
            self._fft_axes = figure.add_subplot(111)
            self._eegpsd_layout.addWidget(self._fft_canvas, 0, 1, 1, 1)
            self._fft_axes.set_ylim(bottom=0.0, top=self._channels + 2)
            self._fft_axes.set_xscale('log')
            self._fft_axes.set_yticks(ticks=np.arange(1, self._channels + 1), labels=c_names)
            self._fft_axes.set_title('Current PSD')

            # bandpass history plot
            figure = plt.figure()
            self._hst_canvas = FigureCanvasQTAgg(figure)
            self._hst_axes = figure.add_subplot(111)
            self._eegpsd_layout.addWidget(self._hst_canvas, 1, 0, 1, 1)
            self._hst_axes.set_ylim([-0.1, 5.1])
            self._hst_axes.set_xticks([])
            self._hst_axes.set_yticks([0, 1, 2, 3, 4], ['δ', 'θ', 'α', 'β', 'γ'])
            self._hst_axes.set_title('{} -- PSD History over {} minutes'.format(self._name, self._history_length/60.0))

            # bandpass bar graph
            figure = plt.figure()
            self._bnd_canvas = FigureCanvasQTAgg(figure)
            self._bnd_axes = figure.add_subplot(111)
            self._eegpsd_layout.addWidget(self._bnd_canvas, 1, 1, 1, 1)
            self._bnd_axes.set_ylim([0.0, 1.1])
            self._bnd_axes.set_xticks([1, 2, 3, 4, 5], ['δ', 'θ', 'α', 'β', 'γ'])
            self._bnd_axes.set_yticks([])
            self._bnd_axes.set_title('Frequency Band Power'.format(self._data_seconds))

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

    def _create_dancing_dots_display_tab(self, create):

        if create:

            # note
            print("Creating Dancing Dots display tab.")

            # create widgets
            self._ddots_tab = QtWidgets.QWidget()
            self._ddots_layout = QtWidgets.QGridLayout(self._ddots_tab)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
            self._ddots_tab.setSizePolicy(sizePolicy)
            self._parent_tabwidget.addTab(self._ddots_tab, "")
            self._parent_tabwidget.setTabText(self._parent_tabwidget.indexOf(self._ddots_tab),
                                              self._name + " -- Dancing Dots")

            # plt.plot(t[0], s[0][0])
            ylim = 32
            plt.ioff()
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            fig.subplots_adjust(top=1, bottom=0.0)
            self._ddots_canvas = FigureCanvasQTAgg(fig)
            self._ddots_axes = ax
            self._ddots_layout.addWidget(self._ddots_canvas, 0, 0, 1, 1)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_theta_zero_location("N")
            ax.set_ylim(0.0, ylim)
            ax.set_facecolor('k')
            fig.set_facecolor('k')

            # start display thread
            time.sleep(1)
            self._showing_ddots = True
            thdsp = threading.Thread(target=self._display_dancing_dots_opengl)
            thdsp.start()

        else:
            if self._ddots_tab:
                print("Removing Dancing Dots display tab.")
                self._parent_tabwidget.removeTab(self._parent_tabwidget.indexOf(self._ddots_tab))
            self._showing_ddots = False

    def _display_dancing_dots(self):

        # starting thread
        print("Starting EEG/PSD display.")
        while not len(self._eeg_data)\
                or not len(self._fft_data)\
                or not len(self._bnd_data)\
                or not len(self._hst_data):
            time.sleep(0.2)

        # Create sum of sine waves of different frequencies
        dt = 0.005
        t0 = np.arange(0, 5 * np.pi, dt)
        f = [(abs(np.sin(1 * t0))),
             (abs(np.sin(2 * t0))),
             (abs(np.sin(3 * t0))),
             (abs(np.sin(4 * t0))),
             (abs(np.sin(5 * t0))),
             ((np.sin(2 * t0) + 1) * 10.0)]

        # Create 10 circles of different lengths
        M = 40  # number of circles
        N = 200  # points per circle
        lines = [None] * M
        freq_start = 1.0
        freq_step = 0.005
        r = [[]] * M  # the interpolated circle data, zeros for now, consisting of M*N circles s[M][1..5], all the same length
        s = [[]] * M  # the raw circle data, zeros for now, consisting of M*N circles s[M][1..5]
        t = [[]] * M  # the base to plot against, will go from 0 to 2pi, t[M]
        for m in range(0, M):
            s[m] = [[]] * 5
            freq = freq_start + m * freq_step  # our circle frequency
            print(freq)
            for n in range(0, 5):
                s[m][n] = np.zeros(int((2 / freq) * np.pi / dt))
            t[m] = [m for m in np.arange(0, 2 * np.pi, 2 * np.pi / N)]

        # Average over all circles - pre-calculate parts of the sum of sines
        for m in range(0, M):
            for n in range(0, 5):
                # only plot if you can fill the entire circle
                for i in range(0, (len(f[n]) // len(s[m][n])) * len(s[m][n])):
                    value = f[n][i] - 0.5
                    index = i % len(s[m][n])
                    s[m][n][index] += value  # add to front
                    s[m][n][-(index + 1)] += value  # add to back, so beginning and end match and the circle stays closed

        # downsample so all circles have the same length (N samples)
        for m in range(0, M):
            r[m] = [[]] * 5
            for n in range(0, 5):
                r[m][n] = np.zeros(N)
                for i in range(0, N - 1):
                    index = int(float(i) * (len(s[m][n]) - 1) / (N - 1.0))
                    r[m][n][i] = s[m][n][index]
                r[m][n][N - 1] = s[m][n][0]  # close the circle

        first = True
        p = 0.0
        q = 1.0
        k = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        while self._streaming and self._showing_ddots:

            # k direction
            # p in/out movement

            # cX - amount of frequency ring fX for each frequency X (out of five)
            c1 = self._bnd_data[0]
            c2 = self._bnd_data[1]
            c3 = self._bnd_data[2]
            c4 = self._bnd_data[3]
            c5 = self._bnd_data[3]
            c6 = 0.2*(c1 - c2 + c3 - c4 + c5)     # c6 - amount of in/out movement
            print("Frequency bands: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(c1, c2, c3, c4, c5), end='\r')

            # k[X] - turning speed and direction of each frequency ring
            k += np.array([c1, -c2, c3, -c4, c5])
            kn = np.floor(k)   # left index into ring
            km = kn + 1        # right index into ring
            kp = km - k        # left amount
            kq = k - kn        # right amount

            # plot curves
            last_data = []
            for m in range(0, M):
                # offset = np.sqrt((M-m))
                offset = M - m
                offset = np.sqrt(np.sqrt(offset * offset * offset))  # x^(3/4)

                # soft rolling of circles by interpolating between floor(k) and ceil(k)
                data = (  c1 * (np.roll(r[m][0], kn[0]) * kp[0] + np.roll(r[m][0], km[0]) * kq[0])
                        + c2 * (np.roll(r[m][1], kn[1]) * kp[1] + np.roll(r[m][1], km[1]) * kq[1])
                        + c3 * (np.roll(r[m][2], kn[2]) * kp[2] + np.roll(r[m][2], km[2]) * kq[2])
                        + c4 * (np.roll(r[m][3], kn[3]) * kp[3] + np.roll(r[m][3], km[3]) * kq[3])
                        + c5 * (np.roll(r[m][4], kn[4]) * kp[4] + np.roll(r[m][4], km[4]) * kq[4])
                        + 1)
                data /= data.max()
                data *= offset
                data += offset
                if first:
                    try:
                        if len(last_data):
                            [lines[m]] = self._ddots_axes.plot(t[m],
                                                               p * last_data + q * data,
                                                               '.',
                                                               color=(m / M, 0.5 * m / M, 1 - m / M))
                    except BaseException as e:
                        print("Plot exception {} for curve n={}".format(e, m))
                else:
                    try:
                        if len(last_data):
                            lines[m].set_ydata(p * last_data + q * data)
                            if m == 1:
                                lines[m].set_color(
                                    (c2 * q * m / M, c3 * q * 0.5 * m / M, (1.0 - 0.5 * c4) * q * (1 - m / M)))
                            elif m == M - 1:
                                lines[m].set_color(
                                    (c2 * p * m / M, c3 * p * 0.5 * m / M, (1.0 - 0.5 * c4) * p * (1 - m / M)))
                            else:
                                lines[m].set_color((c2 * m / M, c3 * 0.5 * m / M, (1.0 - 0.5 * c4) * (1 - m / M)))
                    except BaseException as e:
                        print("Plot exception {} for curve n={}".format(e, m))
                last_data = data
            self._ddots_canvas.draw()
            first = False
            time.sleep(0.01)
            p = (p + c6) % 1
            q = 1.0 - p

    def _create_dancing_dots_display_opengl_tab(self, create):

        if create:

            # note
            print("Creating Dancing Dots display tab.")

            # create widgets
            self._ddots_tab = QtWidgets.QWidget()
            self._ddots_layout = QtWidgets.QGridLayout(self._ddots_tab)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
            self._ddots_tab.setSizePolicy(sizePolicy)
            self._parent_tabwidget.addTab(self._ddots_tab, "")
            self._parent_tabwidget.setTabText(self._parent_tabwidget.indexOf(self._ddots_tab),
                                              self._name + " -- Dancing Dots")

            # plt.plot(t[0], s[0][0])
            self._ddots_ogl_wdg = OpenGLDancingDots()
            self._ddots_layout.addWidget(self._ddots_ogl_wdg, 0, 0, 1, 1)

            # start display thread
            time.sleep(1)
            self._showing_ddots = True
            self._ddots_ogl_wdg.start()

        else:
            if self._ddots_tab:
                print("Removing Dancing Dots display tab.")
                self._parent_tabwidget.removeTab(self._parent_tabwidget.indexOf(self._ddots_tab))
            self._showing_ddots = False

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
        self._eeg_stream.connect(acquisition_delay=0.1, processing_flags="all")
        self._eeg_stream.filter(1, 70)
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
        self._fft_resolution = self._fft_freqs[1]
        delta = abs(self._fft_freqs - 4.0).argmin()      # delta: 0-4 Hz
        theta = abs(self._fft_freqs - 7.0).argmin()      # theta: 4-7 Hz
        alpha = abs(self._fft_freqs - 12.0).argmin()     # alpha: 8-12 Hz
        beta = abs(self._fft_freqs - 30.0).argmin()      # beta: 13-30 Hz
        gamma = abs(self._fft_freqs - 50.0).argmin()    # gamma: 30-100 Hz
        bins = [delta, theta, alpha, beta, gamma]
        smooth = self._bnd_smoothing

        is_relative = True

        # start streaming loop
        while self._streaming:
            try:
                with self._fft_lock:
                    with self._eeg_lock:
                        self._fft_data = np.fft.rfft(self._eeg_data, axis=1)
                    self._fft_data = np.abs(self._fft_data)**2
                    all_channels = self._fft_data.sum(axis=0)[1:]/self._channels     # sum of fft over all channels, excluding DC
                    bnd_data = np.array([a[0].sum()*self._fft_resolution for a in np.split(all_channels, bins)[:5]])
                    if is_relative:
                        bnd_data = bnd_data / bnd_data.sum() # relative power
                    with self._bnd_lock:
                        self._bnd_data = smooth*self._bnd_data + (1.0-smooth)*bnd_data
                        with self._hst_lock:
                            self._hst_data[:, :-1] = self._hst_data[:, 1:]
                            self._hst_data[:, -1] = self._bnd_data
                    if is_relative:
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

    def _update_gui(self):
        with self._gui_lock:
            self._lsl_label.setText(self._lsl_info)
            self._bnd_label.setText(self._bnd_info)
            self._eeg_label.setText(self._eeg_info)


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
