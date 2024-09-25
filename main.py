# -*- coding:utf-8 -*-

import sys

# main starting file for use with or with pyinstaller for creating executable binaries.
# For normal running please execute the software as a module (python3 -m samadhi);
import faulthandler
from samadhi import Samadhi

faulthandler.enable()
if len(sys.argv) == 2:
    Samadhi(filename=sys.argv[1])
else:
    Samadhi()
