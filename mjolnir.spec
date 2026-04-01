# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files

# This must match your filename in GitHub exactly
target_script = 'MjölnIR.py'

block_cipher = None

# Collect data files for matplotlib
added_files = collect_data_files('matplotlib')

# Check for icon
icon_path = 'icon.ico' if os.path.exists('icon.ico') else None

a = Analysis(
    [target_script],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'scipy.special.cython_special',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'pandas._libs.tslibs.base',
        'pandas._libs.tslibs.defs',
        'pandas._libs.tslibs.nanops',
        'pandas._libs.tslibs.hashtable',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MjölnIR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MjölnIR',
)
