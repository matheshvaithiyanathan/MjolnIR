# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Ensure the script finds your icon if you have one
icon_path = 'icon.ico' if os.path.exists('icon.ico') else None

added_files = [
    # Add any non-python data files here (e.g., ('images/*.png', 'images'))
]

# Matplotlib and Scipy sometimes require explicit data collection
added_files += collect_data_files('matplotlib')

a = Analysis(
    ['sample_code.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'scipy.special.cython_special',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'pandas._libs.tslibs.base',
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

pyz = PYZ(a.brain, a.zipped_data, cipher=block_cipher)

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
    console=False, # Set to False to hide the terminal window
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
