# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata

datas = []
hiddenimports = ['ai_helper.organizer', 'ai_helper.clipboard_monitor', 'ai_helper.gemma_finetuner', 'pynput']
datas += copy_metadata('pynput')
if collect_submodules('pynput'):
    hiddenimports += collect_submodules('pynput')
# Add metadata and submodules for ML/transformers libraries if available
try:
    datas += copy_metadata('transformers')
    hiddenimports += collect_submodules('transformers')
except Exception:
    pass
try:
    datas += copy_metadata('torch')
except Exception:
    pass
try:
    hiddenimports += collect_submodules('bitsandbytes')
except Exception:
    pass


a = Analysis(
    ['ai_helper\\__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
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
    name='AIHelper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
