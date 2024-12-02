# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['allhere/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['mne_lsl', 'mne'],
    hookspath=['extra-hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='allhere',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

#splash = Splash('all-here-splash.png',
#                binaries=a.binaries,
#                datas=a.datas,
#                text_pos=(10, 50),
#                text_size=12,
#                text_color='black')
