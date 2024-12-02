# custom hook to include data files from mne, needed by mne_lsl
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

hiddenimports = collect_submodules('mne')
datas = collect_data_files('mne') + copy_metadata('mne_lsl')
