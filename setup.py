import sys
import os
from cx_Freeze import setup, Executable
import matplotlib

# Находим путь к данным matplotlib
mpl_data_dir = os.path.dirname(matplotlib.__file__)

# Зависимости, которые нужно включить
build_exe_options = {
    "packages": [
        "numpy",
        "matplotlib",
        "scipy",
        "paramiko",
        "tkinter",
        "numpy.core._methods",
        "numpy.lib.format",
        "matplotlib.backends.backend_tkagg",
        "pkg_resources._vendor",
    ],
    "excludes": ["PyQt4", "PyQt5", "PySide", "unittest", "email", "http", "xml", "pydoc"],
    "include_msvcr": True,
    "include_files": [
        ("README.md", "README.md"),
        ("requirements.txt", "requirements.txt"),
        (os.path.join(mpl_data_dir, "mpl-data"), "mpl-data"),
    ],
    "zip_include_packages": ["*"],
    "zip_exclude_packages": [],
}

# Базовое имя для exe файла
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Используем Win32GUI для Windows-приложения без консоли

setup(
    name="Flashforge Bed Level Analyzer",
    version="1.0",
    description="Анализатор уровня стола для Flashforge",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "bed_level_analyzer.py",
            base=base,
            target_name="FlashforgeBedLevelAnalyzer.exe",
        )
    ]
)
