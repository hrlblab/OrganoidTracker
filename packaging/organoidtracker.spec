# CPU-only portable directory build. Run from the repository's locked CPU/Qt build environment.
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch
from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).resolve().parent
if torch.version.cuda is not None:
    raise RuntimeError('Use a CPU-only build environment (uv sync --extra cpu --extra qt --group packaging).')

metadata_dir = Path(CONF['workpath']) / 'metadata'
metadata_dir.mkdir(parents=True, exist_ok=True)
try:
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip())
except (OSError, subprocess.CalledProcessError):
    revision, dirty = None, None
build_info = {'source_revision': revision, 'dirty': dirty, 'python': platform.python_version(),
              'platform': platform.platform(), 'torch': torch.__version__, 'accelerator': 'cpu'}
info_path = metadata_dir / '_build_info.json'
info_path.write_text(json.dumps(build_info, indent=2), encoding='utf-8')
datas = collect_data_files('sam2') + collect_data_files('organoidtracker')
datas += [(str(root / 'LICENSE'), '.'), (str(root / 'NOTICE'), '.'), (str(info_path), 'organoidtracker')]
packages = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata['Name']
    packages.append({'name': name, 'version': distribution.version,
                     'license_expression': distribution.metadata.get('License-Expression')})
    # Include the actual license/notice files supplied by installed distributions, including
    # the Qt bindings and native dependencies. No lab data or model weights enter the bundle.
    for file in distribution.files or []:
        if any(part.lower() in ('license', 'licenses', 'copying') for part in file.parts) or file.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')):
            source = Path(distribution.locate_file(file))
            if source.is_file():
                datas.append((str(source), str(Path('licenses') / name / file.parent)))
for package in ['organoidtracker', 'torch', 'torchvision', 'hydra-core', 'omegaconf', 'numpy', 'pyside6', 'reportlab']:
    datas += copy_metadata(package)
package_path = metadata_dir / 'bundled-packages.json'
package_path.write_text(json.dumps(sorted(packages, key=lambda item: item['name'].lower()), indent=2), encoding='utf-8')
datas.append((str(package_path), '.'))
a = Analysis([str(Path(SPECPATH) / 'entry.py')], pathex=[str(root / 'src')], datas=datas,
             hiddenimports=collect_submodules('sam2.modeling'),
             excludes=['organoidtracker.gui_tk', 'tkinter', 'transformers', 'pytest', 'IPython'],
             hooksconfig={'matplotlib': {'backends': ['Agg']}}, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='organoidtracker', console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name='organoidtracker', upx=False)
