# -*- coding:utf-8 -*-
#!/usr/bin/python3
from PyQt6 import QtWidgets
from .dancingdots import OpenGLDancingDots
import time

class DancingDotsLayout(QtWidgets.QGridLayout):

    _showing_ddots = False
    _ddots_wdg = None       # opengl widget with the dots

    def __init__(self, parent, get_data):
        super().__init__(parent)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        parent.setSizePolicy(sizePolicy)

        # plt.plot(t[0], s[0][0])
        self._ddots_wdg = OpenGLDancingDots(get_data, self.toggle_fullscreen_dancing_dots)
        self.addWidget(self._ddots_wdg, 0, 0, 1, 1)

        # start display thread
        time.sleep(1)
        self._showing_ddots = True
        self._ddots_wdg.start()

    def toggle_fullscreen_dancing_dots(self, fullscreen):
        if not fullscreen:
            self.addWidget(self._ddots_wdg, 0, 0, 1, 1)
        if fullscreen:
            self.removeWidget(self._ddots_wdg)
            self._ddots_wdg.setParent(None)
            self._ddots_wdg.showFullScreen()

