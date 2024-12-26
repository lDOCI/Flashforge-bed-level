import collections
import importlib
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force matplotlib to use TkAgg backend
import matplotlib.pyplot
import matplotlib.ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import os
import sys
import matplotlib.patches as patches
import paramiko
from scp import SCPClient

def get_working_dir():
    """Получение рабочей директории программы"""
    if getattr(sys, 'frozen', False):
        # Если программа запущена как exe
        return sys._MEIPASS
    else:
        # Если программа запущено как python script
        return os.path.dirname(os.path.abspath(__file__))

def init_directories():
    """Инициализация рабочих директорий"""
    base_dir = get_working_dir()
    print(f"Base directory: {base_dir}")  # Для отладки
    
    # Создаем нужные поддиректории
    dirs = {
        'config': os.path.join(base_dir, 'config'),
        'shaper_data': os.path.join(base_dir, 'shaper_data'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    # Создаем директории если их нет
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")  # Для отладки
        
    return dirs

# Добавляем путь к temp_shaper в PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'temp_shaper'))
from extras import shaper_defs

try:
    pass
except ImportError as e:
    print(f"Error importing shaper_defs: {e}")
    sys.exit(1)

# Константы из оригинального кода
MIN_FREQ = 5.
MAX_FREQ = 200.
WINDOW_T_SEC = 0.5
MAX_SHAPER_FREQ = 150.
TEST_DAMPING_RATIOS = [0.075, 0.1, 0.15]
AUTOTUNE_SHAPERS = ['zv', 'mzv', 'ei', '2hump_ei', '3hump_ei']

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import use as plot_in_window
from matplotlib.patches import Rectangle, Circle, Arc, FancyArrowPatch, Wedge, Arrow, PathPatch
from matplotlib.path import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Tuple, Dict, Optional
import json
import re
import sys
import time
import scipy.signal as signal
from types import SimpleNamespace
from PIL import Image, ImageTk
import os
import math
import matplotlib.patches as patches

# Force matplotlib to use TkAgg backend
plot_in_window('TkAgg')

# Настройка поддержки русского языка в matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

class CalibrationData:
    def __init__(self, freq_bins, psd_sum, psd_x, psd_y, psd_z):
        self.freq_bins = freq_bins
        self.psd_sum = psd_sum
        self.psd_x = psd_x
        self.psd_y = psd_y
        self.psd_z = psd_z
        self._psd_list = [self.psd_sum, self.psd_x, self.psd_y, self.psd_z]
        self._psd_map = {'x': self.psd_x, 'y': self.psd_y, 'z': self.psd_z,
                         'all': self.psd_sum}
        self.data_sets = 1
        self.numpy = np

    def add_data(self, other):
        np = self.numpy
        joined_data_sets = self.data_sets + other.data_sets
        for psd, other_psd in zip(self._psd_list, other._psd_list):
            other_normalized = other.data_sets * np.interp(
                    self.freq_bins, other.freq_bins, other_psd)
            psd *= self.data_sets
            psd[:] = (psd + other_normalized) * (1. / joined_data_sets)
        self.data_sets = joined_data_sets

    def set_numpy(self, numpy):
        self.numpy = numpy

    def normalize_to_frequencies(self):
        for psd in self._psd_list:
            psd /= self.freq_bins + .1
            psd[self.freq_bins < 5.] = 0.

    def get_psd(self, axis='all'):
        return self._psd_map[axis]

CalibrationResult = collections.namedtuple(
        'CalibrationResult',
        ('name', 'freq', 'vals', 'vibrs', 'smoothing', 'score', 'max_accel'))

class ShaperResult:
    """Класс для хранения результатов анализа шейпера."""
    def __init__(self, name, freq, shaper, vals, vibrs, smoothing, score, max_accel):
        self.name = name
        self.freq = freq
        self.shaper = shaper
        self.vals = vals
        self.vibrs = vibrs
        self.smoothing = smoothing
        self.score = score
        self.max_accel = max_accel

class ShaperCalibrate:
    def __init__(self, printer=None):
        self.printer = printer
        self.error = printer.command_error if printer else Exception
        try:
            self.numpy = importlib.import_module('numpy')
        except ImportError:
            raise self.error(
                    "Failed to import `numpy` module, make sure it was "
                    "installed via `~/klippy-env/bin/pip install`")

    def _split_into_windows(self, x, window_size, overlap):
        # Memory-efficient algorithm to split an input 'x' into a series
        # of overlapping windows
        step_between_windows = window_size - overlap
        n_windows = (x.shape[-1] - overlap) // step_between_windows
        shape = (window_size, n_windows)
        strides = (x.strides[-1], step_between_windows * x.strides[-1])
        return self.numpy.lib.stride_tricks.as_strided(
                x, shape=shape, strides=strides, writeable=False)

    def _psd(self, x, fs, nfft):
        # Calculate power spectral density (PSD) using Welch's algorithm
        np = self.numpy
        window = np.kaiser(nfft, 6.)
        # Compensation for windowing loss
        scale = 1.0 / (window**2).sum()

        # Split into overlapping windows of size nfft
        overlap = nfft // 2
        x = self._split_into_windows(x, nfft, overlap)

        # First detrend, then apply windowing function
        x = window[:, None] * (x - np.mean(x, axis=0))

        # Calculate frequency response for each window using FFT
        result = np.fft.rfft(x, n=nfft, axis=0)
        result = np.conjugate(result) * result
        result *= scale / fs
        # For one-sided FFT output the response must be doubled, except
        # the last point for unpaired Nyquist frequency (assuming even nfft)
        # and the 'DC' term (0 Hz)
        result[1:-1,:] *= 2.

        # Welch's algorithm: average response over windows
        psd = result.real.mean(axis=-1)

        # Calculate the frequency bins
        freqs = np.fft.rfftfreq(nfft, 1. / fs)
        return freqs, psd

    def calc_freq_response(self, raw_values):
        np = self.numpy
        if raw_values is None:
            return None
        if isinstance(raw_values, np.ndarray):
            data = raw_values
        else:
            samples = raw_values.get_samples()
            if not samples:
                return None
            data = np.array(samples)

        N = data.shape[0]
        T = data[-1,0] - data[0,0]
        SAMPLING_FREQ = N / T
        # Round up to the nearest power of 2 for faster FFT
        M = 1 << int(SAMPLING_FREQ * WINDOW_T_SEC - 1).bit_length()
        if N <= M:
            return None

        # Calculate PSD (power spectral density) of vibrations per
        # frequency bins (the same bins for X, Y, and Z)
        fx, px = self._psd(data[:,1], SAMPLING_FREQ, M)
        fy, py = self._psd(data[:,2], SAMPLING_FREQ, M)
        fz, pz = self._psd(data[:,3], SAMPLING_FREQ, M)
        return CalibrationData(fx, px+py+pz, px, py, pz)

    def process_accelerometer_data(self, data):
        calibration_data = self.calc_freq_response(data)
        if calibration_data is None:
            raise self.error(
                    "Internal error processing accelerometer data %s" % (data,))
        calibration_data.set_numpy(self.numpy)
        return calibration_data

    def _estimate_shaper(self, shaper, test_damping_ratio, test_freqs):
        np = self.numpy

        A, T = np.array(shaper[0]), np.array(shaper[1])
        inv_D = 1. / A.sum()

        omega = 2. * math.pi * test_freqs
        damping = test_damping_ratio * omega
        omega_d = omega * np.sqrt(1. - test_damping_ratio**2)
        W = A * np.exp(np.outer(-damping, (T[-1] - T)))
        S = W * np.sin(np.outer(omega_d, T))
        C = W * np.cos(np.outer(omega_d, T))
        return np.sqrt(S.sum(axis=1)**2 + C.sum(axis=1)**2) * inv_D

    def _estimate_remaining_vibrations(self, shaper, test_damping_ratio,
                                       freq_bins, psd):
        vals = self._estimate_shaper(shaper, test_damping_ratio, freq_bins)
        # The input shaper can only reduce the amplitude of vibrations by
        # SHAPER_VIBRATION_REDUCTION times, so all vibrations below that
        # threshold can be ignored
        vibr_threshold = psd.max() / shaper_defs.SHAPER_VIBRATION_REDUCTION
        remaining_vibrations = self.numpy.maximum(
                vals * psd - vibr_threshold, 0).sum()
        all_vibrations = self.numpy.maximum(psd - vibr_threshold, 0).sum()
        if all_vibrations < 1e-10:
            all_vibrations = 1e-10
        return (remaining_vibrations / all_vibrations, vals)

    def _get_shaper_smoothing(self, shaper, accel=5000, scv=5.):
        half_accel = accel * .5

        A, T = shaper
        inv_D = 1. / sum(A)
        n = len(T)
        # Calculate input shaper shift
        ts = sum([A[i] * T[i] for i in range(n)]) * inv_D

        # Calculate offset for 90 and 180 degrees turn
        offset_90 = offset_180 = 0.
        for i in range(n):
            if T[i] >= ts:
                # Calculate offset for one of the axes
                offset_90 += A[i] * (scv + half_accel * (T[i]-ts)) * (T[i]-ts)
            offset_180 += A[i] * half_accel * (T[i]-ts)**2
        offset_90 *= inv_D * math.sqrt(2.)
        offset_180 *= inv_D
        return max(offset_90, offset_180)

    def fit_shaper(self, shaper_cfg, calibration_data, max_smoothing):
        np = self.numpy

        test_freqs = np.arange(shaper_cfg.min_freq, MAX_SHAPER_FREQ, .2)

        freq_bins = calibration_data.freq_bins
        psd = calibration_data.psd_sum[freq_bins <= MAX_FREQ]
        freq_bins = freq_bins[freq_bins <= MAX_FREQ]

        best_res = None
        results = []
        for test_freq in test_freqs[::-1]:
            shaper_vibrations = 0.
            shaper_vals = np.zeros(shape=freq_bins.shape)
            shaper = shaper_cfg.init_func(
                    test_freq, shaper_defs.DEFAULT_DAMPING_RATIO)
            shaper_smoothing = self._get_shaper_smoothing(shaper)
            if max_smoothing and shaper_smoothing > max_smoothing and best_res:
                return best_res
            # Exact damping ratio of the printer is unknown, pessimizing
            # remaining vibrations over possible damping values
            for dr in TEST_DAMPING_RATIOS:
                vibrations, vals = self._estimate_remaining_vibrations(
                        shaper, dr, freq_bins, psd)
                shaper_vals = np.maximum(shaper_vals, vals)
                if vibrations > shaper_vibrations:
                    shaper_vibrations = vibrations
            max_accel = self.find_shaper_max_accel(shaper)
            # The score trying to minimize vibrations, but also accounting
            # the growth of smoothing. The formula itself does not have any
            # special meaning, it simply shows good results on real user data
            shaper_score = shaper_smoothing * (shaper_vibrations**1.5 +
                                               shaper_vibrations * .2 + .01)
            results.append(
                    CalibrationResult(
                        name=shaper_cfg.name, freq=test_freq, vals=shaper_vals,
                        vibrs=shaper_vibrations, smoothing=shaper_smoothing,
                        score=shaper_score, max_accel=max_accel))
            if best_res is None or best_res.vibrs > results[-1].vibrs:
                # The current frequency is better for the shaper.
                best_res = results[-1]
        # Try to find an 'optimal' shapper configuration: the one that is not
        # much worse than the 'best' one, but gives much less smoothing
        selected = best_res
        for res in results[::-1]:
            if res.vibrs < best_res.vibrs * 1.1 and res.score < selected.score:
                selected = res
        return selected

    def _bisect(self, func):
        left = right = 1.
        while not func(left):
            right = left
            left *= .5
        if right == left:
            while func(right):
                right *= 2.
        while right - left > 1e-8:
            middle = (left + right) * .5
            if func(middle):
                left = middle
            else:
                right = middle
        return left

    def find_shaper_max_accel(self, shaper):
        # Just some empirically chosen value which produces good projections
        # for max_accel without much smoothing
        TARGET_SMOOTHING = 0.12
        max_accel = self._bisect(lambda test_accel: self._get_shaper_smoothing(
            shaper, test_accel) <= TARGET_SMOOTHING)
        return max_accel

    def find_best_shaper(self, calibration_data, max_smoothing, logger=None):
        best_shaper = None
        all_shapers = []
        for shaper_cfg in shaper_defs.INPUT_SHAPERS:
            if shaper_cfg.name not in AUTOTUNE_SHAPERS:
                continue
            shaper = self.fit_shaper(shaper_cfg, calibration_data, max_smoothing)
            if logger is not None:
                logger("Подобран шейпер '%s' частота = %.1f Гц "
                       "(вибрации = %.1f%%, сглаживание ~= %.3f)" % (
                           shaper.name, shaper.freq, shaper.vibrs * 100.,
                           shaper.smoothing))
                logger("Чтобы избежать слишком сильного сглаживания с '%s', рекомендуемое "
                       "max_accel <= %.0f мм/сек^2" % (
                           shaper.name, round(shaper.max_accel / 100.) * 100.))
            all_shapers.append(shaper)
            if (best_shaper is None or shaper.score * 1.2 < best_shaper.score or
                    (shaper.score * 1.05 < best_shaper.score and
                        shaper.smoothing * 1.1 < best_shaper.smoothing)):
                # Either the shaper significantly improves the score (by 20%),
                # or it improves the score and smoothing (by 5% and 10% resp.)
                best_shaper = shaper
        return best_shaper, all_shapers

class BedLevelAnalyzer:
    def __init__(self):
        """Инициализация анализатора уровня стола"""
        # Инициализируем рабочие директории
        self.working_dirs = init_directories()
        print(f"Working directories: {self.working_dirs}")  # Для отладки
        
        # Инициализируем переменные
        self.mesh_data = None
        self.max_delta = 0.0  # Инициализируем значением по умолчанию
        self.recommendations = []
        self.shaper_calibrate = ShaperCalibrate()  # Создаем объект для анализа шейперов
        self.shaper_data = None  # Инициализируем данные шейпера
        self.accelerometer_file = None  # Добавляем сохранение пути к файлу акселерометра
        self.current_file = None  # Текущий загруженный файл
        self.default_settings = {
            'SHOW_MINUTES': True,
            'SHOW_DEGREES': False,
            'SHOW_BELT_STAGE': True,
            'SHOW_SCREW_STAGE': True,
            'SHOW_TAPE_STAGE': True,
            'BELT_DELTA_THRESHOLD': 0.8,   # Увеличен с 0.41
            'SCREW_DELTA_THRESHOLD': 0.4,  # Уменьшен с 0.3
            'TAPE_DELTA_THRESHOLD': 0.2,   # Уменьшен с 0.25
            'BELT_TOOTH_MM': 0.4,
            'TAPE_THICKNESS': 0.1,
            'SSH_HOST': '',
            'SSH_USERNAME': '',
            'SSH_PASSWORD': '',
            'INTERPOLATION_COEFFICIENT': 100  # Добавлен коэффициент интерполяции
        }
        self.settings = self.default_settings.copy()
        self.create_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Обработчик закрытия окна

    def load_data(self, path: str) -> np.ndarray:
        """Загрузка и парсинг данных уровня стола из конфигурационного файла."""
        mesh_data = []
        line_count = 0

        try:
            with open(path, 'r') as file:
                in_points_section = False
                for line in file:
                    if 'points =' in line:
                        in_points_section = True
                        continue
                    elif in_points_section:
                        if not line.startswith("#*#"):
                            break
                        if line_count > 4:
                            # Преобразуем в numpy массив и сохраняем
                            self.mesh_data = np.array(mesh_data)
                            # Рассчитываем максимальное отклонение
                            self.max_delta = float(np.max(self.mesh_data) - np.min(self.mesh_data))
                            return self.mesh_data

                        # Извлекаем значения из строки
                        line = line[3:].strip()  # Убираем "#*#" и пробелы
                        line = line.replace(',', '')  # Убираем запятые если есть
                        values = [float(x) for x in line.split() if x]  # Конвертируем непустые значения
                        if values:  # Добавляем только если есть значения
                            mesh_data.append(values)
                            line_count += 1

            # Если дошли до конца файла, тоже сохраняем данные
            if mesh_data:
                self.mesh_data = np.array(mesh_data)
                self.max_delta = float(np.max(self.mesh_data) - np.min(self.mesh_data))
                return self.mesh_data

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
            return None

    def analyze_bed_level(self) -> None:
        """Анализ уровня стола с учетом настроек отображения."""
        if self.mesh_data is None:
            return

        max_delta = float(np.max(self.mesh_data) - np.min(self.mesh_data))
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, f"❗ АНАЛИЗ СТОЛА\nМаксимальное отклонение: {max_delta:.3f}мм\n\n")

        # Проверка ремней
        if max_delta > self.settings['BELT_DELTA_THRESHOLD'] and self.settings['SHOW_BELT_STAGE']:
            belt_states = self.analyze_belt_tension()
            self.text_widget.insert(tk.END, "ЭТАП 1: РЕГУЛИРОВКА РЕМНЕЙ\n\n")
            
            self.text_widget.insert(tk.END, "Что такое винты T8:\n")
            self.text_widget.insert(tk.END, "T8 - это трапецеидальные винты с шагом 8мм, которые используются для точного перемещения стола по вертикали. ")
            self.text_widget.insert(tk.END, "На принтере установлено три таких винта:\n")
            self.text_widget.insert(tk.END, "• Один задний (опорный)\n")
            self.text_widget.insert(tk.END, "• Два передних (регулируемых)\n\n")

            self.text_widget.insert(tk.END, "Подготовка:\n")
            self.text_widget.insert(tk.END, "1. Отключите принтер от питания\n")
            self.text_widget.insert(tk.END, "2. Поднимите стол так, чтобы он был близко к заднему винту T8\n")
            self.text_widget.insert(tk.END, "3. Положите принтер на спину\n\n")

            try:
                belt_image_path = os.path.join(os.path.dirname(__file__), 'images/belt_adjustment.png')
                belt_image = Image.open(belt_image_path)
                widget_width = self.text_widget.winfo_width()
                desired_width = int(widget_width * 0.8)
                aspect_ratio = belt_image.size[1] / belt_image.size[0]
                desired_height = int(desired_width * aspect_ratio)
                belt_image = belt_image.resize((desired_width, desired_height), Image.Resampling.LANCZOS)
                belt_photo = ImageTk.PhotoImage(belt_image)
                self.text_widget.image_create(tk.END, image=belt_photo)
                self.text_widget.image = belt_photo
            except Exception as e:
                pass
            self.text_widget.insert(tk.END, "\n")
            
            try:
                screws_image_path = os.path.join(os.path.dirname(__file__), 'images/screws_adjustment.png')
                screws_image = Image.open(screws_image_path)
                screws_photo = ImageTk.PhotoImage(screws_image.resize((desired_width, desired_height), Image.Resampling.LANCZOS))
                self.text_widget.image_create(tk.END, image=screws_photo)
                self.text_widget.screws_image = screws_photo
            except Exception as e:
                pass
            self.text_widget.insert(tk.END, "\n")
            
            for position, info in belt_states.items():
                # Правильные склонения для названий валов
                val_names = {
                    "Передний правый": {"name": "переднего правого"},
                    "Передний левый": {"name": "переднего левого"},
                    "Задний": {"name": "заднего"}
                }
                val_info = val_names.get(info['name'], {"name": info['name']})
                
                self.text_widget.insert(tk.END, f"Регулировка {val_info['name']} вала:\n", "header")
                
                # Определяем направление вращения на основе нужного действия
                action_direction = "против часовой стрелки" if info['direction'] == "вверх" else "по часовой стрелке"
                action_type = "поднять" if info['direction'] == "вверх" else "опустить"
                
                # Добавляем действия с цветовым выделением
                actions = [
                    "1. Найдите серый натяжитель с двумя черными болтами\n",
                    "2. Ослабьте два черных болта фиксации натяжителя\n",
                    "3. Открутите фиксирующую пружину натяжителя\n",
                    f"4. Удерживая натяжитель левой рукой, правой рукой поверните вал {action_direction} чтобы {action_type} угол стола на {info['teeth']} зуба ({info['teeth'] * 0.4:.1f}мм)\n",
                    "5. Проверьте высоту угла относительно задней точки\n",
                    "6. Закрутите фиксирующую пружину\n",
                    "7. Затяните черные болты натяжителя\n"
                ]
                
                for action in actions:
                    self.text_widget.insert(tk.END, action, "action")
                self.text_widget.insert(tk.END, "\n")

            self.text_widget.insert(tk.END, "Важные замечания:\n", "header")
            warnings = [
                "• Задний винт T8 используется как опорная точка - не трогайте его\n",
                "• Для точной настройки используйте щуп между соплом и столом\n",
                "• Проверяйте высоту во всех углах после регулировки\n",
                "• При ослаблении болтов натяжителя ремень становится независимым\n",
                "• При затягивании - синхронизируется с другими валами\n"
            ]
            for warning in warnings:
                self.text_widget.insert(tk.END, warning, "action")
            self.text_widget.insert(tk.END, "\n")

            self.text_widget.insert(tk.END, "После регулировки:\n", "header")
            final_steps = [
                "1. Проверьте натяжку всех ремней (без фанатизма)\n",
                "2. Поставьте принтер в нормальное положение\n",
                "3. Выполните повторное измерение\n",
                "4. Если отклонение < 0.4мм, перейдите к регулировке винтов\n"
            ]
            for step in final_steps:
                self.text_widget.insert(tk.END, step, "action")
            return

        # Проверка винтов
        if max_delta > self.settings['SCREW_DELTA_THRESHOLD'] and self.settings['SHOW_SCREW_STAGE']:
            self.text_widget.insert(tk.END, "ЭТАП 2: РЕГУЛИРОВКА ВИНТОВ\n\n")
            self.text_widget.insert(tk.END, "ВАЖНО:\n")
            self.text_widget.insert(tk.END, "• Удерживайте винт сверху шестигранником\n")
            self.text_widget.insert(tk.END, "• Крутите гайку снизу ключом в указанном направлении\n\n")

            screw_data = self.get_screw_adjustments()
            for corner, (minutes, direction, action) in screw_data.items():
                if action != "норма":
                    self.text_widget.insert(tk.END, f"{corner}:\n")
                    self.text_widget.insert(tk.END, f"1. Найдите винт в {corner.lower()}\n")
                    rotation_text = []
                    if self.settings['SHOW_MINUTES']:
                        rotation_text.append(f"{minutes} минут")
                    if self.settings['SHOW_DEGREES']:
                        degrees = minutes * 6  # 1 минута = 6 градусов
                        rotation_text.append(f"{degrees}°")
                    rotation_str = " (" + " / ".join(rotation_text) + ")"
                    self.text_widget.insert(tk.END, f"2. Поверните винт {direction} на{rotation_str}\n")
                    self.text_widget.insert(tk.END, f"3. Действие: выполните точную {action} угла стола\n")
                    self.text_widget.insert(tk.END, "4. Проверьте уровень после поворота\n\n")
            return

        # Проверка на скотч
        if max_delta > self.settings['TAPE_DELTA_THRESHOLD'] and self.settings['SHOW_TAPE_STAGE']:
            self.text_widget.insert(tk.END, "ЭТАП 3: НАКЛЕЙКА СКОТЧА\n\n")
            tape_data = self.get_tape_adjustments()
            if tape_data:
                self.text_widget.insert(tk.END, "Требуется наклейка скотча:\n")
                for pos, layers in tape_data.items():
                    thickness = layers * self.settings['TAPE_THICKNESS']
                    self.text_widget.insert(tk.END, f"• Точка {pos}: {layers} слой(а) ({thickness:.2f}мм)\n")
            return

        self.text_widget.insert(tk.END, "✅ Стол выровнен! Отклонение в норме.\n")

    def draw_2d_graph(self) -> None:
        """Отрисовка 2D тепловой карты уровня стола."""
        if self.mesh_data is None:
            messagebox.showwarning("Предупреждение", "Пожалуйста, сначала загрузите данные")
            return

        # Закрываем все предыдущие окна matplotlib
        plt.close('all')
        
        # Рассчитываем максимальное отклонение
        self.max_delta = float(np.max(self.mesh_data) - np.min(self.mesh_data))
        
        plt.figure(figsize=(10, 8)).canvas.manager.set_window_title('2D карта уровня стола')
        ax = plt.gca()
        cmap = cm.coolwarm_r
                # Устанавливаем одинаковые пределы для обоих графиков
        vmin, vmax = np.min(self.mesh_data), np.max(self.mesh_data)

        im = ax.imshow(np.flipud(self.mesh_data), cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im)

        for (i, j), val in np.ndenumerate(np.flipud(self.mesh_data)):
            ax.text(j, i, f'{val:.6f}', ha='center', va='center',
                   color='black' if abs(val) < 0.3 else 'white')

        plt.text(-0.1, -0.1, "Передний левый", transform=ax.transAxes)
        plt.text(1.1, -0.1, "Передний правый", transform=ax.transAxes)
        plt.text(-0.1, 1.1, "Задний левый", transform=ax.transAxes)
        plt.text(1.1, 1.1, "Задний правый", transform=ax.transAxes)

        plt.title(f"Карта уровня стола (Макс. отклонение: {self.max_delta:.6f}мм)")
        plt.show()

    def draw_3d_graph(self) -> None:
        """Отрисовка 3D карты уровня стола."""
        if self.mesh_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите файл конфигурации")
            return

        # Очищаем все предыдущие графики
        plt.close('all')

        # Создаем фигуру и 3D оси
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Создаем более плотную сетку для интерполяции
        x_original = np.linspace(-0.5, 4.5, 6)
        y_original = np.linspace(-0.5, 4.5, 6)
        
        # Создаем более плотную сетку для гладкой поверхности
        grid_points = int(self.settings['INTERPOLATION_COEFFICIENT'])  # Используем коэффициент из настроек
        x_smooth = np.linspace(-0.5, 4.5, grid_points)
        y_smooth = np.linspace(-0.5, 4.5, grid_points)
        X_smooth, Y_smooth = np.meshgrid(x_smooth, y_smooth)

        # Расширяем данные
        extended_data = np.zeros((6, 6))
        extended_data[:-1, :-1] = self.mesh_data
        extended_data[-1, :] = extended_data[-2, :]
        extended_data[:, -1] = extended_data[:, -2]

        # Интерполируем данные на более плотную сетку
        from scipy.interpolate import RectBivariateSpline
        interp_spline = RectBivariateSpline(x_original, y_original, extended_data)
        Z_smooth = interp_spline(x_smooth, y_smooth)

        # Применяем экспоненциальное масштабирование
        z_min, z_max = np.min(Z_smooth), np.max(Z_smooth)
        normalized_data = (Z_smooth - z_min) / (z_max - z_min)
        
        # Применяем экспоненциальное масштабирование
        alpha = 0.1  # Параметр сглаживания (меньше = более плавный)
        scaled_data = np.exp(alpha * normalized_data) / np.exp(alpha)
        
        # Возвращаем к исходному диапазону
        scaled_data = scaled_data * (z_max - z_min) + z_min

        # Настраиваем отображение поверхности
        surf = ax.plot_surface(X_smooth, Y_smooth, scaled_data,
                             cmap=cm.coolwarm_r,
                             linewidth=0,
                             antialiased=True)

        # Настраиваем вид
        ax.view_init(elev=20, azim=225)  # Меняем угол обзора
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))

        # Добавляем подписи углов
        z_offset = (np.max(scaled_data) - np.min(scaled_data)) * 0.5  # Вычисляем смещение относительно диапазона данных
        z_label = np.min(scaled_data) - z_offset  # Устанавливаем позицию подписей
        
        # Добавляем подписи с увеличенным размером шрифта и черным цветом
        ax.text(0, 0, z_label, "Передний левый", ha='right', va='bottom', fontsize=12, color='black', zorder=100)
        ax.text(4, 0, z_label, "Передний правый", ha='left', va='bottom', fontsize=12, color='black', zorder=100)
        ax.text(0, 4, z_label, "Задний левый", ha='right', va='top', fontsize=12, color='black', zorder=100)
        ax.text(4, 4, z_label, "Задний правый", ha='left', va='top', fontsize=12, color='black', zorder=100)

        # Добавляем цветовую шкалу и заголовок
        plt.colorbar(surf)
        ax.set_title(f"3D карта уровня стола (Макс. отклонение: {self.max_delta:.6f}мм)")

        # Создаем новое окно верхнего уровня
        graph_window = tk.Toplevel(self.root)
        graph_window.title("3D карта уровня стола")
        graph_window.geometry("800x600")

        # Создаем канвас в новом окне
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Добавляем панель инструментов
        toolbar = NavigationToolbar2Tk(canvas, graph_window)
        toolbar.update()

        # Устанавливаем обработчик закрытия окна
        def on_closing():
            plt.close(fig)  # Закрываем фигуру matplotlib
            graph_window.destroy()  # Закрываем окно

        graph_window.protocol("WM_DELETE_WINDOW", on_closing)

    def create_visual_recommendations(self) -> None:
        """Создание визуальных рекомендаций."""
        if self.mesh_data is None:
            messagebox.showwarning("Предупреждение", "Пожалуйста, сначала загрузите данные")
            return

        # Создаем новое окно для визуальных рекомендаций
        rec_window = tk.Toplevel(self.root)
        rec_window.title("Визуальные рекомендации")
        rec_window.geometry("1200x800")

        # Создаем Figure для matplotlib
        fig = Figure(figsize=(15, 8))

        # Создаем канвас
        canvas = FigureCanvasTkAgg(fig, master=rec_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Добавляем подграфики (2x2)
        ax1 = fig.add_subplot(221)  # Схема стола
        ax2 = fig.add_subplot(222)  # Схема регулировки винтов
        ax3 = fig.add_subplot(223)  # Схема ремней
        ax4 = fig.add_subplot(224)  # Схема скотча

        # Рисуем схему стола с проблемными зонами
        self.draw_bed_scheme(ax1)

        # Рисуем новую схему регулировки винтов
        self.draw_screw_adjustment_visualization_in_subplot(ax2)

        # Рисуем график регулировки винтов стола
        belt_adjustments = self.draw_belt_visualization(ax3)

        # Рисуем схему наклеивания скотча
        self.draw_tape_scheme(ax4)

        fig.tight_layout()
        canvas.draw()

        # Добавляем рекомендации по регулировке ремней
        if belt_adjustments:
            self.recommendations.append("Подробные рекомендации по регулировке ремней:")
            self.recommendations.extend(self.generate_belt_recommendations(belt_adjustments))

    def draw_bed_scheme(self, ax):
        """Отрисовка схемы стола с проблемными зонами."""
        ax.clear()
        # Рисуем контур стола
        bed = Rectangle((0, 0), 4, 4, fill=False, color='black')
        ax.add_patch(bed)

        # Добавляем сетку 5x5
        for i in range(5):
            ax.axhline(y=i, color='gray', linestyle=':', alpha=0.3)
            ax.axvline(x=i, color='gray', linestyle=':', alpha=0.3)

        # Отмечаем проблемные зоны на основе данных
        mean_height = np.mean(self.mesh_data)
        for i in range(5):
            for j in range(5):
                if self.mesh_data[i, j] > mean_height + 0.2:
                    circle = Circle((j, i), 0.2, color='red', alpha=0.5)
                    ax.add_patch(circle)
                elif self.mesh_data[i, j] < mean_height - 0.2:
                    circle = Circle((j, i), 0.2, color='blue', alpha=0.5)
                    ax.add_patch(circle)

        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_title("Карта проблемных зон")
        ax.set_aspect('equal')

        # Добавляем легенду
        red_circle = Circle((0, 0), 0.1, color='red', alpha=0.5)
        blue_circle = Circle((0, 0), 0.1, color='blue', alpha=0.5)
        ax.legend([red_circle, blue_circle], ['Высокие точки', 'Низкие точки'])

    def draw_tape_scheme(self, ax) -> None:
        """Отрисовка схемы наклеивания алюминиевого скотча."""
        if self.mesh_data is None:
            return

        ax.set_aspect('equal')

        # Рисуем контур стола
        rect = Rectangle((-0.5, -0.5), 5, 5, fill=False, color='gray')
        ax.add_patch(rect)

        # Добавляем сетку 5x5
        for i in range(6):
            ax.axhline(y=i-0.5, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=i-0.5, color='gray', linestyle=':', alpha=0.5)

        # Добавляем аннотации для рядов и колонок
        for i in range(5):
            ax.text(-1, i, str(i+1), va='center', ha='center', fontsize=10, color='black')
            ax.text(i, -1, chr(65+i), va='center', ha='center', fontsize=10, color='black')

        mean_height = np.mean(self.mesh_data)

        # Функция для определения количества слоев скотча
        def calculate_tape_layers(height_diff):
            # Толщина одного слоя алюминиевого скотча (примерно 0.1мм)
            TAPE_THICKNESS = 0.1
            return max(1, int(np.ceil(abs(height_diff) / TAPE_THICKNESS)))

        # Создаем сетку точек для анализа
        x = np.linspace(0, 4, self.mesh_data.shape[1])
        y = np.linspace(0, 4, self.mesh_data.shape[0])
        X, Y = np.meshgrid(x, y)

        # Находим области, где нужен скотч (ниже среднего уровня)
        for i in range(len(x)):
            for j in range(len(y)):
                height = self.mesh_data[j, i]
                if height < mean_height - 0.05:  # Порог в 0.05мм
                    # Вычисляем разницу и количество слоев
                    diff = mean_height - height
                    layers = calculate_tape_layers(diff)

                    # Рисуем квадрат
                    square = Rectangle((X[j,i] - 0.4, Y[j,i] - 0.4),
                                    0.8, 0.8,
                                    color='yellow',
                                    alpha=0.5)
                    ax.add_patch(square)

                    # Добавляем текст с количеством слоев
                    ax.text(X[j,i], Y[j,i], str(layers),
                           ha='center', va='center',
                           color='black',
                           fontsize=10,
                           fontweight='bold')

        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_title("Схема наклеивания алюминиевого скотча\n(числа = количество слоев)")
        ax.axis('off')

    def draw_screw_adjustment_visualization_in_subplot(self, ax):
        """Визуализация регулировки винтов стола на подграфике."""
        # Импортируем необходимые классы matplotlib
        from matplotlib.patches import Rectangle, Wedge, Circle
        import matplotlib.animation as animation

        # Рисуем контур стола (вид сверху)
        bed = Rectangle((-2, -2), 4, 4, fill=False, color='black', linewidth=2)
        ax.add_patch(bed)

        # Позиции винтов
        corners = [
            ("Передний левый", -1.5, -1.5),
            ("Передний правый", 1.5, -1.5),
            ("Задний левый", -1.5, 1.5),
            ("Задний правый", 1.5, 1.5)
        ]

        mean_height = np.mean(self.mesh_data)
        
        # Список для хранения данных анимации
        screw_data = []

        # Добавляем винты и их обозначения
        for name, x, y in corners:
            # Получаем высоту для текущего угла
            i, j = {
                "Передний левый": (0, 0),
                "Передний правый": (0, -1),
                "Задний левый": (-1, 0),
                "Задний правый": (-1, -1)
            }[name]
            
            height = float(self.mesh_data[i, j])
            diff = height - mean_height
            
            # Основной круг
            base_circle = Circle((x, y), 0.4, color='gray', fill=False, linewidth=2)
            ax.add_patch(base_circle)

            if abs(diff) > 0.1:
                # Рассчитываем градусы поворота
                total_degrees = abs(diff) * 100 * 5.14  # DEGREES_PER_01MM = 5.14
                minutes = int(total_degrees * 60 / 360)
                
                # Создаем текст с учетом настроек
                rotation_text = []
                if self.settings['SHOW_MINUTES']:
                    rotation_text.append(f"{minutes} минут")
                if self.settings['SHOW_DEGREES']:
                    degrees = minutes * 6  # 1 минута = 6 градусов
                    rotation_text.append(f"{degrees}°")
                
                rotation = " / ".join(rotation_text) if rotation_text else f"{minutes} минут"
                
                if diff < -0.1:  # Нужно поднять угол (против часовой)
                    color = 'green'
                    rotation_text = '↺'
                    wedge_color = 'green'
                    wedge_alpha = 0.2
                    start_angle = 90
                    end_angle = 90 + total_degrees
                    rotation_direction = 'Против часовой'

                    # Анимируемый клин для поворота против часовой
                    animated_wedge = Wedge(
                        center=(x, y),
                        r=0.4,
                        theta1=start_angle,
                        theta2=start_angle,  # Начинаем с начального угла
                        color=wedge_color,
                        alpha=wedge_alpha
                    )
                else:  # Нужно опустить угол (по часовой)
                    color = 'red'
                    rotation_text = '↻'
                    wedge_color = 'red'
                    wedge_alpha = 0.2
                    start_angle = 90
                    end_angle = 90 - total_degrees
                    rotation_direction = 'По часовой'

                    # Анимируемый клин для поворота по часовой
                    animated_wedge = Wedge(
                        center=(x, y),
                        r=0.4,
                        theta2=90,  # Фиксированный начальный угол
                        theta1=90,  # Начальный угол анимации
                        color=wedge_color,
                        alpha=wedge_alpha
                    )

                ax.add_patch(animated_wedge)
                screw_data.append({
                    'wedge': animated_wedge,
                    'start_angle': start_angle,
                    'end_angle': end_angle,
                    'clockwise': diff > -0.1  # True для по часовой
                })

                # Добавляем текст стрелки в центр круга
                ax.text(x, y, rotation_text,
                        ha='center', va='center',
                        fontsize=20,
                        color=color,
                        fontweight='bold')

                # Добавляем текст с градусами/минутами и направлением
                label_text = f"{rotation}\n{rotation_direction}"
                ax.text(x, y - 0.7, label_text,
                        ha='center', va='center',
                        fontsize=8, color=color,
                        fontweight='bold')
            else:
                ax.text(x, y - 0.7, '✓ Норма',
                        ha='center', va='center',
                        fontsize=8,
                        color='gray')

        # Добавляем легенду
        legend_x = 2.1  # Было 1.8, делаем правее
        legend_y = 1.8
        
        # Добавляем обозначения
        ax.text(legend_x, legend_y, 'Обозначения', fontweight='bold', fontsize=9)
        ax.text(legend_x, legend_y - 0.3, 'Опустить', color='red', fontsize=8)
        ax.text(legend_x, legend_y - 0.5, 'Поднять', color='green', fontsize=8)
        ax.text(legend_x, legend_y - 0.7, 'Норма', color='gray', fontsize=8)

        # Функция инициализации анимации
        def init():
            for data in screw_data:
                if data['clockwise']:  # По часовой
                    data['wedge'].set_theta1(data['start_angle'])
                else:  # Против часовой
                    data['wedge'].set_theta2(data['start_angle'])
            return [data['wedge'] for data in screw_data]

        # Функция анимации
        def animate(frame):
            for data in screw_data:
                if data['clockwise']:  # По часовой
                    current_angle = data['start_angle'] - (data['start_angle'] - data['end_angle']) * (frame / 20)
                    data['wedge'].set_theta1(current_angle)
                else:  # Против часовой
                    current_angle = data['start_angle'] + (data['end_angle'] - data['start_angle']) * (frame / 20)
                    data['wedge'].set_theta2(current_angle)
            return [data['wedge'] for data in screw_data]

        # Создаем анимацию
        if screw_data:  # Только если есть что анимировать
            anim = animation.FuncAnimation(
                ax.figure,
                animate,
                init_func=init,
                frames=20,
                interval=50,
                blit=True,
                repeat=True,
                repeat_delay=1000
            )
            # Сохраняем ссылку на анимацию
            ax.figure.animation = anim

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title('Регулировка винтов стола (вид сверху принтера)')
        ax.axis('equal')
        ax.axis('off')

    def draw_belt_visualization(self, ax) -> None:
        """
        Визуализация схемы Z-валов с направлением регулировки.
        """
        ax.clear()
        ax.set_title('Схема регулировки Z-валов (вид снизу принтера)', fontsize=12)
        
        # Параметры платформы
        width, depth = 220, 220
        
        # Позиции Z-валов с русскими названиями
        z_positions = {
            'front_left':  (50, depth-50),    # Передний левый
            'front_right': (width-50, depth-50),  # Правый передний
            'back':        (width/2, 50)  # Задний
        }
        
        # Расчет натяжения ремней
        belt_adjustments = self.analyze_belt_tension()
        
        # Отрисовка платформы
        platform = plt.Polygon([
            (0, 0), (width, 0), 
            (width, depth), (0, depth)
        ], fill=False, edgecolor='gray', linewidth=2)
        ax.add_patch(platform)
        
        # Цвета для разных состояний
        colors = {
            'Натянуть': 'red', 
            'Ослабить': 'blue'
        }
        
        # Отрисовка Z-валов и аннотаций
        for position, (x, y) in z_positions.items():
            # Базовые параметры
            color = 'gray'
            marker_size = 200
            
            # Если есть регулировка для этого вала
            if position in belt_adjustments:
                info = belt_adjustments[position]
                color = colors.get(info['action'], 'gray')
                marker_size = 300
                
                # Стрелка с направлением
                arrow_props = dict(
                    facecolor=color, 
                    edgecolor=color, 
                    width=0.3, 
                    head_width=10, 
                    head_length=10
                )
                
                # Вертикальная стрелка вниз
                if info['direction'] == 'вниз':
                    ax.arrow(x, y+20, 0, -30, **arrow_props)
                    ax.text(x-20, y+40, 'Опустить', 
                            color=color, fontsize=8, 
                            verticalalignment='bottom', 
                            horizontalalignment='right')
                # Вертикальная стрелка вверх
                else:
                    ax.arrow(x, y-20, 0, 30, **arrow_props)
                    ax.text(x+20, y-50, 'Поднять', 
                            color=color, fontsize=8, 
                            verticalalignment='top', 
                            horizontalalignment='left')
            
            # Отрисовка винта
            ax.scatter(x, y, c=color, s=marker_size, alpha=0.5, edgecolors='black')
            
            # Подпись винта с русскими названиями
            valve_names = {
                'front_left': 'Передний левый',
                'front_right': 'Передний правый', 
                'back': 'Задний'
            }
            ax.text(x, y-30, valve_names[position], 
                    horizontalalignment='center', 
                    verticalalignment='bottom', 
                    fontsize=10)
        
        # Соединительные линии ремней
        for (start_name, start_pos), (end_name, end_pos) in [
            (('front_left', z_positions['front_left']), 
             ('front_right', z_positions['front_right'])),
            (('front_right', z_positions['front_right']), 
             ('back', z_positions['back'])),
            (('back', z_positions['back']), 
             ('front_left', z_positions['front_left']))
        ]:
            ax.plot([start_pos[0], end_pos[0]], 
                    [start_pos[1], end_pos[1]], 
                    'k--', linewidth=1, alpha=0.5)
        
        # Настройка осей
        ax.set_xlim(0, width)
        ax.set_ylim(0, depth)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Легенда
        legend_elements = [
            plt.scatter([], [], c='red', s=50, label='Натянуть'),
            plt.scatter([], [], c='blue', s=50, label='Ослабить'),
            plt.scatter([], [], c='gray', s=50, label='В норме')
        ]
        ax.legend(handles=legend_elements, loc='best', title='Состояние валов', fontsize=8)

    def generate_belt_recommendations(self, belt_adjustments: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Генерация текстовых рекомендаций по регулировке ремней.

        Returns:
            list: Список подробных рекомендаций по регулировке
        """
        if not belt_adjustments:
            return ["✅ Регулировка ремней не требуется. Стол выровнен."]

        # Общие инструкции добавляются только один раз
        recommendations = [
            "🔧 Порядок регулировки:",
            "1. Отключите принтер",
            "2. Положите принтер на спину",
            "3. Ослабьте фиксаторы натяжителя"
        ]

        # Детальные рекомендации для каждого вала
        for valve, info in belt_adjustments.items():
            if abs(info['diff']) > self.settings['BELT_DELTA_THRESHOLD']:
                recommendations.append(
                    f"\n• {info['name']}"
                )

        # Заключительные инструкции
        recommendations.extend([
            "\n4. После регулировки:",
            "   - Затяните натяжитель",
            "   - Проверьте калибровку"
        ])

        return recommendations

    def format_number(self, number: float, unit: str) -> str:
        """Форматирование числа с единицей измерения."""
        return f"{number:.3f} {unit}"

    def format_belt_teeth(self, teeth: int) -> str:
        """Форматирование количества зубьев ремня."""
        if teeth == 1:
            return "1 зуб"
        elif teeth in [2, 3, 4]:
            return f"{teeth} зуба"
        else:
            return f"{teeth} зубьев"

    def calculate_belt_adjustment(self, height_diff: float) -> int:
        if height_diff <= 0.4:
            return 0
        if 0.4 < height_diff < 0.8:
            return 1
        return max(1, int(height_diff / self.settings['BELT_TOOTH_MM']))

    def create_settings_window(self):
        """Создание окна настроек разработчика."""
        # Проверяем, существует ли уже окно настроек
        if hasattr(self, 'settings_window') and self.settings_window.winfo_exists():
            self.settings_window.lift()  # Поднимаем окно на передний план
            return
            
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Настройки разработчика")
        self.settings_window.geometry("400x800")  # Увеличили высоту окна

        def update_analysis(*args):
            """Обновить анализ при изменении настроек."""
            try:
                # Обновляем настройки отображения
                self.settings['SHOW_MINUTES'] = self.show_minutes_var.get()
                self.settings['SHOW_DEGREES'] = self.show_degrees_var.get()
                
                # Обновляем этапы
                self.settings['SHOW_BELT_STAGE'] = self.show_belt_var.get()
                self.settings['SHOW_SCREW_STAGE'] = self.show_screw_var.get()
                self.settings['SHOW_TAPE_STAGE'] = self.show_tape_var.get()
                
                try:
                    # Обновляем пороги
                    self.settings['BELT_DELTA_THRESHOLD'] = float(self.belt_threshold_entry.get())
                    self.settings['SCREW_DELTA_THRESHOLD'] = float(self.screw_threshold_entry.get())
                    self.settings['TAPE_DELTA_THRESHOLD'] = float(self.tape_threshold_entry.get())
                    self.settings['BELT_TOOTH_MM'] = float(self.belt_tooth_entry.get())
                    self.settings['TAPE_THICKNESS'] = float(self.tape_thickness_entry.get())
                    self.settings['INTERPOLATION_COEFFICIENT'] = float(self.interpolation_entry.get())
                    
                    # Обновляем анализ и график, если данные загружены
                    if hasattr(self, 'mesh_data') and self.mesh_data is not None:
                        self.analyze_bed_level()
                        # Обновляем график только если он существует
                        if hasattr(self, 'canvas') and self.canvas is not None:
                            self.canvas.get_tk_widget().destroy()
                            fig = self.plot_adjustments()
                            if fig:
                                self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
                                self.canvas.draw()
                                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                except ValueError:
                    messagebox.showerror("Ошибка", "Проверьте правильность ввода числовых значений")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка обновления: {str(e)}")

        # Создаем фрейм для чекбоксов
        checkbox_frame = tk.LabelFrame(self.settings_window, text="Отображение")
        checkbox_frame.pack(fill="x", padx=10, pady=5)

        # Чекбоксы для режимов отображения
        self.show_minutes_var = tk.BooleanVar(value=self.settings['SHOW_MINUTES'])
        show_minutes_cb = tk.Checkbutton(checkbox_frame, text="Показывать минуты", 
                                       variable=self.show_minutes_var, command=update_analysis)
        show_minutes_cb.pack(anchor="w")

        self.show_degrees_var = tk.BooleanVar(value=self.settings['SHOW_DEGREES'])
        show_degrees_cb = tk.Checkbutton(checkbox_frame, text="Показывать градусы", 
                                       variable=self.show_degrees_var, command=update_analysis)
        show_degrees_cb.pack(anchor="w")

        # Создаем фрейм для этапов
        stages_frame = tk.LabelFrame(self.settings_window, text="Этапы калибровки")
        stages_frame.pack(fill="x", padx=10, pady=5)

        # Чекбоксы для этапов
        self.show_belt_var = tk.BooleanVar(value=self.settings['SHOW_BELT_STAGE'])
        show_belt_cb = tk.Checkbutton(stages_frame, text="Этап регулировки ремней", 
                                    variable=self.show_belt_var, command=update_analysis)
        show_belt_cb.pack(anchor="w")

        self.show_screw_var = tk.BooleanVar(value=self.settings['SHOW_SCREW_STAGE'])
        show_screw_cb = tk.Checkbutton(stages_frame, text="Этап регулировки винтов", 
                                     variable=self.show_screw_var, command=update_analysis)
        show_screw_cb.pack(anchor="w")

        self.show_tape_var = tk.BooleanVar(value=self.settings['SHOW_TAPE_STAGE'])
        show_tape_cb = tk.Checkbutton(stages_frame, text="Этап наклейки скотча", 
                                    variable=self.show_tape_var, command=update_analysis)
        show_tape_cb.pack(anchor="w")

        # Создаем фрейм для порогов
        thresholds_frame = tk.LabelFrame(self.settings_window, text="Пороговые значения (мм)")
        thresholds_frame.pack(fill="x", padx=10, pady=5)

        def validate_float(P):
            """Проверка ввода числа с плавающей точкой."""
            if P == "" or P == ".":
                return True
            try:
                float(P)
                return True
            except ValueError:
                return False
        vcmd = self.settings_window.register(validate_float)

        # Поля ввода для порогов с валидацией
        tk.Label(thresholds_frame, text="Порог регулировки ремней:").pack(anchor="w")
        self.belt_threshold_entry = tk.Entry(thresholds_frame, validate="key", validatecommand=(vcmd, '%P'))
        self.belt_threshold_entry.insert(0, str(self.settings['BELT_DELTA_THRESHOLD']))
        self.belt_threshold_entry.bind('<Return>', update_analysis)
        self.belt_threshold_entry.bind('<FocusOut>', update_analysis)
        self.belt_threshold_entry.pack(fill="x", padx=5)

        tk.Label(thresholds_frame, text="Порог регулировки винтов:").pack(anchor="w")
        self.screw_threshold_entry = tk.Entry(thresholds_frame, validate="key", validatecommand=(vcmd, '%P'))
        self.screw_threshold_entry.insert(0, str(self.settings['SCREW_DELTA_THRESHOLD']))
        self.screw_threshold_entry.bind('<Return>', update_analysis)
        self.screw_threshold_entry.bind('<FocusOut>', update_analysis)
        self.screw_threshold_entry.pack(fill="x", padx=5)

        tk.Label(thresholds_frame, text="Порог наклейки скотча:").pack(anchor="w")
        self.tape_threshold_entry = tk.Entry(thresholds_frame, validate="key", validatecommand=(vcmd, '%P'))
        self.tape_threshold_entry.insert(0, str(self.settings['TAPE_DELTA_THRESHOLD']))
        self.tape_threshold_entry.bind('<Return>', update_analysis)
        self.tape_threshold_entry.bind('<FocusOut>', update_analysis)
        self.tape_threshold_entry.pack(fill="x", padx=5)

        # Дополнительные параметры
        params_frame = tk.LabelFrame(self.settings_window, text="Дополнительные параметры")
        params_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(params_frame, text="мм на зуб ремня:").pack(anchor="w")
        self.belt_tooth_entry = tk.Entry(params_frame, validate="key", validatecommand=(vcmd, '%P'))
        self.belt_tooth_entry.insert(0, str(self.settings['BELT_TOOTH_MM']))
        self.belt_tooth_entry.bind('<Return>', update_analysis)
        self.belt_tooth_entry.bind('<FocusOut>', update_analysis)
        self.belt_tooth_entry.pack(fill="x", padx=5)

        tk.Label(params_frame, text="Толщина слоя скотча (мм):").pack(anchor="w")
        self.tape_thickness_entry = tk.Entry(params_frame, validate="key", validatecommand=(vcmd, '%P'))
        self.tape_thickness_entry.insert(0, str(self.settings['TAPE_THICKNESS']))
        self.tape_thickness_entry.bind('<Return>', update_analysis)
        self.tape_thickness_entry.bind('<FocusOut>', update_analysis)
        self.tape_thickness_entry.pack(fill="x", padx=5)

        # Коэффициент интерполяции
        tk.Label(params_frame, text="Коэффициент интерполяции 3D карты:").pack(anchor="w")
        self.interpolation_entry = tk.Entry(params_frame, validate="key", validatecommand=(vcmd, '%P'))
        self.interpolation_entry.insert(0, str(self.settings['INTERPOLATION_COEFFICIENT']))
        self.interpolation_entry.bind('<Return>', update_analysis)
        self.interpolation_entry.bind('<FocusOut>', update_analysis)
        self.interpolation_entry.pack(fill="x", padx=5)
        tk.Label(params_frame, text="(от 10 до 500, меньшие значения = меньше ресурсов)", 
                 font=('Arial', 8)).pack(anchor="w", padx=5)

        # SSH параметры
        ssh_frame = tk.LabelFrame(self.settings_window, text="SSH параметры")
        ssh_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(ssh_frame, text="IP:").pack(anchor="w")
        self.ssh_host_entry = tk.Entry(ssh_frame)
        self.ssh_host_entry.insert(0, str(self.settings.get('SSH_HOST', '')))
        self.ssh_host_entry.pack(fill="x", padx=5)

        tk.Label(ssh_frame, text="Логин:").pack(anchor="w")
        self.ssh_username_entry = tk.Entry(ssh_frame)
        self.ssh_username_entry.insert(0, str(self.settings.get('SSH_USERNAME', '')))
        self.ssh_username_entry.pack(fill="x", padx=5)

        tk.Label(ssh_frame, text="Пароль:").pack(anchor="w")
        self.ssh_password_entry = tk.Entry(ssh_frame, show="*")
        self.ssh_password_entry.insert(0, str(self.settings.get('SSH_PASSWORD', '')))
        self.ssh_password_entry.pack(fill="x", padx=5)

        # Кнопки SSH
        ssh_buttons_frame = tk.Frame(ssh_frame)
        ssh_buttons_frame.pack(fill="x", padx=5, pady=5)

        self.connect_button = tk.Button(ssh_buttons_frame, text="Подключиться", command=self.connect_ssh)
        self.connect_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.get_config_button = tk.Button(ssh_buttons_frame, text="Получить printer.cfg", 
                                         command=self.get_printer_cfg, state=tk.DISABLED)
        self.get_config_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.get_shaper_button = tk.Button(ssh_buttons_frame, text="Получить шейперы", 
                                         command=self.get_shaper_files, state=tk.DISABLED)
        self.get_shaper_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Кнопки
        button_frame = tk.Frame(self.settings_window)
        button_frame.pack(fill="x", padx=10, pady=10)

        def reset_settings():
            """Сбросить настройки к значениям по умолчанию."""
            self.settings = self.default_settings.copy()
            self.settings_window.destroy()
            self.create_settings_window()
            if hasattr(self, 'mesh_data') and self.mesh_data is not None:
                self.analyze_bed_level()
                # Обновляем график только если он существует
                if hasattr(self, 'canvas') and self.canvas is not None:
                    self.canvas.get_tk_widget().destroy()
                    fig = self.plot_adjustments()
                    if fig:
                        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
                        self.canvas.draw()
                        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        tk.Button(button_frame, text="Сбросить", command=reset_settings).pack(side="left", padx=5)
        tk.Button(button_frame, text="Закрыть", command=self.settings_window.destroy).pack(side="right", padx=5)

    def create_gui(self) -> None:
        """Создание главного окна интерфейса."""
        # Создаем главное окно
        self.root = tk.Tk()
        self.root.title("Flashforge A5M Analyzer")
        self.root.geometry("800x600")

        # Создание главного меню
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Добавляем меню файла
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить конфигурацию", command=self.load_file)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)

        # Добавляем меню разработчика
        dev_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Разработчик", menu=dev_menu)
        dev_menu.add_command(label="Настройки", command=self.create_settings_window)

        # Создаем вкладки
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both')

        # Вкладка выравнивания стола
        self.bed_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bed_frame, text='Выравнивание стола')

        # Вкладка шейперов
        self.shaper_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.shaper_frame, text='Шейперы')

        # Создание основной рамки с отступами (для вкладки выравнивания)
        main_frame = ttk.Frame(self.bed_frame, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Секция загрузки файла
        file_frame = ttk.LabelFrame(main_frame, text="Операции с файлами", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.file_path_var = tk.StringVar()
        ttk.Label(file_frame, textvariable=self.file_path_var).grid(row=0, column=0, padx=5)
        ttk.Button(file_frame, text="Загрузить конфигурацию", command=self.load_file).grid(row=0, column=1, padx=5)

        # Секция визуализации
        viz_frame = ttk.LabelFrame(main_frame, text="Визуализация", padding="5")
        viz_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(viz_frame, text="Показать 2D карту", command=self.draw_2d_graph).grid(row=0, column=0, padx=5)
        ttk.Button(viz_frame, text="Показать 3D карту", command=self.draw_3d_graph).grid(row=0, column=1, padx=5)
        ttk.Button(viz_frame, text="Показать рекомендации", command=self.create_visual_recommendations).grid(row=0, column=2, padx=5)

        # Секция рекомендаций
        rec_frame = ttk.LabelFrame(main_frame, text="Анализ и рекомендации", padding="5")
        rec_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.text_widget = tk.Text(rec_frame, wrap=tk.WORD)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Создаем теги для форматирования текста
        self.text_widget.tag_configure("action", foreground="red")
        self.text_widget.tag_configure("header", font=("Helvetica", 10, "bold"))

        scrollbar = ttk.Scrollbar(rec_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.configure(yscrollcommand=scrollbar.set)

        # Настройка вкладки шейперов
        self.create_shaper_tab()

        # Привязываем обработчик изменения размера окна
        self.root.bind('<Configure>', self.on_window_resize)

        # Настройка весов сетки
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        rec_frame.columnconfigure(0, weight=1)
        rec_frame.rowconfigure(0, weight=1)

    def load_file(self) -> None:
        """Обработка загрузки файла."""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Все поддерживаемые", "*.cfg;*.csv"),
                ("Конфигурационные файлы", "*.cfg"),
                ("CSV файлы", "*.csv"),
                ("Все файлы", "*.*")
            ]
        )
        if file_path:
            if file_path.lower().endswith('.csv'):
                # Загрузка файла шейпера
                try:
                    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
                    
                    if data.shape[1] >= 5:  # Проверяем, что есть нужные колонки
                        self.shaper_data = CalibrationData(
                            freq_bins=data[:, 0],
                            psd_sum=data[:, 4],
                            psd_x=data[:, 1],
                            psd_y=data[:, 2],
                            psd_z=data[:, 3]
                        )
                        self.shaper_data.set_numpy(np)
                        self.shaper_data.normalize_to_frequencies()
                        self.accelerometer_file = file_path
                        self.current_file = file_path
                        self.analyze_shaper_data()
                    else:
                        messagebox.showerror("Ошибка", "Неверный формат файла")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось загрузить файл шейпера: {str(e)}")
            else:
                # Загрузка конфигурационного файла
                self.file_path_var.set(f"Файл: {file_path}")
                self.mesh_data = self.load_data(file_path)
                if self.mesh_data is not None:
                    self.analyze_bed_level()
                else:
                    messagebox.showerror("Ошибка", "Не удалось загрузить данные конфигурации")

    def create_shaper_tab(self):
        """Создание элементов вкладки шейперов."""
        # Основной фрейм для вкладки шейперов
        main_frame = ttk.Frame(self.shaper_frame)
        main_frame.pack(fill='both', expand=True)

        # Фрейм для кнопок загрузки
        load_frame = ttk.LabelFrame(main_frame, text="Загрузка данных", padding="5")
        load_frame.pack(pady=5, padx=5, fill='x')

        # Одна кнопка для загрузки данных
        ttk.Button(load_frame, text="Загрузить данные", command=self.load_shaper_data).pack(padx=5, pady=5)

        # Текстовый виджет для результатов (сверху)
        self.shaper_text = tk.Text(main_frame, wrap=tk.WORD, height=4)
        self.shaper_text.pack(padx=10, pady=5, fill='x')

        # Область для графиков
        self.shaper_canvas = ttk.Frame(main_frame)
        self.shaper_canvas.pack(expand=True, fill='both', padx=10, pady=5)

    def load_shaper_data(self):
        """Загрузка данных шейпера."""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
        )
        if file_path:
            try:
                # Определяем ось по имени файла
                axis = 'X' if '_x_' in file_path.lower() else 'Y' if '_y_' in file_path.lower() else None
                if axis is None:
                    messagebox.showerror("Ошибка", "Не удалось определить ось из имени файла. Используйте файлы с '_x_' или '_y_' в названии.")
                    return

                self.shaper_data = self.parse_shaper_log(file_path)
                if self.shaper_data is not None:
                    print(f"Загружены данные {axis} из {file_path}")
                    self.accelerometer_file = file_path
                    self.current_file = file_path
                    self.analyze_shaper_data(axis=axis)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def parse_shaper_log(self, logname: str):
        """Загрузка данных калибровки из CSV файла."""
        try:
            # Проверяем заголовок файла
            with open(logname) as f:
                for header in f:
                    if not header.startswith('#'):
                        break
                if not header.startswith('freq,psd_x,psd_y,psd_z,psd_xyz'):
                    # Сырые данные акселерометра
                    data = np.loadtxt(logname, comments='#', delimiter=',')
                    return self.shaper_calibrate.process_accelerometer_data(data)
            
            # Парсим данные спектральной плотности мощности
            data = np.loadtxt(logname, skiprows=1, comments='#', delimiter=',', dtype=np.float64)
            
            # Проверяем размерность данных
            if data.size == 0:
                raise ValueError("Файл не содержит данных")
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            if data.shape[1] < 5:
                raise ValueError("Недостаточно колонок в файле")
                
            # Создаем объект CalibrationData с правильными типами данных
            calibration_data = CalibrationData(
                freq_bins=data[:, 0].astype(np.float64),
                psd_sum=data[:, 4].astype(np.float64),
                psd_x=data[:, 1].astype(np.float64),
                psd_y=data[:, 2].astype(np.float64),
                psd_z=data[:, 3].astype(np.float64)
            )
            
            # Если input shapers присутствуют в CSV файле,
            # частотный отклик уже нормализован к входным частотам
            if 'mzv' not in header:
                calibration_data.normalize_to_frequencies()
            
            return calibration_data

        except Exception as e:
            print(f"Не удалось загрузить файл: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
            return None

    def analyze_shaper_data(self, axis='X'):
        """Анализ данных input shaper."""
        if not hasattr(self, 'shaper_data'):
            return

        def logger(msg):
            # Игнорируем промежуточные сообщения
            pass

        # Очищаем текстовое поле
        self.shaper_text.delete(1.0, tk.END)
        self.shaper_text.insert(tk.END, f"Анализ лога: {self.accelerometer_file}\n")

        # Находим лучший шейпер
        best_shaper, all_shapers = self.shaper_calibrate.find_best_shaper(
            self.shaper_data, None, logger)

        if best_shaper:
            self.shaper_text.insert(tk.END, f"\nРекомендуемый шейпер для оси {axis}: {best_shaper.name} @ {best_shaper.freq:.1f} Гц\n")
            self.shaper_text.insert(tk.END, f"Вибрации: {best_shaper.vibrs*100:.1f}%, Сглаживание: {best_shaper.smoothing:.2f}, Макс. ускорение: {int(round(best_shaper.max_accel/100.)*100)} мм/с²\n")

            # Создаем график
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.set_xlabel('Частота, Гц')
            ax.set_xlim([0, 200])
            ax.set_ylabel('Спектральная плотность мощности')

            # Отрисовка PSD данных
            freqs = self.shaper_data.freq_bins
            freq_mask = freqs <= 200.
            freqs = freqs[freq_mask]
            
            psd = self.shaper_data.psd_sum[freq_mask]
            px = self.shaper_data.psd_x[freq_mask]
            py = self.shaper_data.psd_y[freq_mask]
            pz = self.shaper_data.psd_z[freq_mask]

            ax.plot(freqs, psd, label='X+Y+Z', color='purple')
            ax.plot(freqs, px, label='X', color='red')
            ax.plot(freqs, py, label='Y', color='green')
            ax.plot(freqs, pz, label='Z', color='blue')

            # Настройка сетки и форматирования
            fontP = matplotlib.font_manager.FontProperties()
            fontP.set_size('x-small')
            
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
            ax.grid(which='major', color='grey')
            ax.grid(which='minor', color='lightgrey')

            # Создаем вторую ось Y для шейперов
            ax2 = ax.twinx()
            ax2.set_ylabel('Снижение вибраций шейпером (коэффициент)')

            # Отображаем все шейперы
            best_shaper_vals = None
            for shaper in all_shapers:
                label = "%s (%.1f Гц, vibr=%.1f%%, sm~=%.2f, accel<=%.f)" % (
                    shaper.name.upper(), shaper.freq,
                    shaper.vibrs * 100., shaper.smoothing,
                    round(shaper.max_accel / 100.) * 100.)
                linestyle = 'dotted'
                if shaper.name == best_shaper.name:
                    linestyle = 'dashdot'
                    best_shaper_vals = shaper.vals
                ax2.plot(freqs, shaper.vals[freq_mask], label=label, linestyle=linestyle)

            # Отображаем результат после применения лучшего шейпера
            ax.plot(freqs, psd * best_shaper_vals[freq_mask], label='После\nшейпера', color='cyan')

            # Добавляем рекомендацию в легенду
            ax2.plot([], [], ' ', label="Рекомендуемый шейпер: %s" % (best_shaper.name.upper()))

            # Настройка легенд
            ax.legend(loc='upper left', prop=fontP)
            ax2.legend(loc='upper right', prop=fontP)

            # Компоновка графика
            fig.tight_layout()

            # Отображаем график
            if hasattr(self, 'shaper_canvas'):
                for widget in self.shaper_canvas.winfo_children():
                    widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=self.shaper_canvas)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas, self.shaper_canvas)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def on_closing(self):
        """Обработка закрытия окна."""
        print("Закрытие приложения...")
        self.root.destroy()

    def run(self) -> None:
        """
        Запуск основного цикла приложения с обработкой прерываний.
        """
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nПрограмма принудительно завершена пользователем.")
        except Exception as e:
            print(f"Произошла ошибка: {e}")
        finally:
            # Безопасное закрытие окна
            try:
                if self.root and hasattr(self.root, 'destroy'):
                    self.root.destroy()
            except Exception:
                pass

    def draw_screw_adjustment(self) -> None:
        """Отображение схемы регулировки винтов."""
        if self.mesh_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите файл конфигурации")
            return

        # Создаем новое окно
        window = tk.Toplevel(self.root)
        window.title("Схема регулировки винтов")
        
        # Создаем холст
        canvas = tk.Canvas(window, width=400, height=400)
        canvas.pack(padx=10, pady=10)
        
        # Рисуем круги для винтов
        # Передний левый
        canvas.create_oval(30, 30, 70, 70)
        canvas.create_text(50, 40, text="21 мин")
        canvas.create_text(50, 55, text="По часовой", fill="red")
        
        # Передний правый
        canvas.create_oval(330, 30, 370, 70)
        
        # Задний левый
        canvas.create_oval(30, 330, 70, 370)
        canvas.create_text(50, 340, text="38 мин")
        canvas.create_text(50, 355, text="По часовой", fill="red")
        
        # Задний правый
        canvas.create_oval(330, 330, 370, 370)
        canvas.create_text(350, 340, text="18 мин")
        canvas.create_text(350, 355, text="Против часовой", fill="green")
        
        # Добавляем легенду
        legend_frame = ttk.LabelFrame(window, text="Обозначения", padding="5")
        legend_frame.pack(padx=10, pady=10)

        ttk.Label(legend_frame, text="Опустить", foreground="red").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(legend_frame, text="Поднять", foreground="green").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(legend_frame, text="Норма", foreground="gray").grid(row=2, column=0, sticky=tk.W)
        
        # Добавляем текстовое поле с рекомендациями
        text = tk.Text(window, height=15, width=50)
        text.pack(fill=tk.BOTH, expand=True)

        # Получаем и выводим рекомендации
        recommendations = self.generate_screw_recommendations()
        for rec in recommendations:
            text.insert(tk.END, rec + "\n")

    def generate_screw_recommendations(self):
        """Генерация рекомендаций по регулировке винтов."""
        recommendations = []
        recommendations.append("ЭТАП 2: РЕГУЛИРОВКА ВИНТОВ\n")
        recommendations.append("ВАЖНО:\n")
        recommendations.append("• Удерживайте винт сверху шестигранником")
        recommendations.append("• Крутите гайку снизу ключом в указанном направлении\n")
        # Получаем актуальные данные о регулировке винтов
        adjustments = self.get_screw_adjustments()
        
        for corner_name, (minutes, direction, action) in adjustments.items():
            recommendations.append(f"{corner_name}:")
            recommendations.append(f"1. Найдите винт в {corner_name.lower()}")
            
            if minutes == 0:
                recommendations.append("2. Регулировка не требуется (в норме)")
            else:
                recommendations.append(f"2. Поверните винт {direction} на {minutes} минут")  # Используем "минут" вместо сокращения "мин"
                recommendations.append(f"3. Действие: выполните точную {action} угла стола")
                recommendations.append("4. Проверьте уровень после поворота")
            
            recommendations.append("")  # Пустая строка между углами
        
        return recommendations

    def analyze_belt_tension(self) -> Dict[str, Dict[str, float]]:
        if self.mesh_data is None:
            return {}

        left_front = float(self.mesh_data[0, 0])    
        right_front = float(self.mesh_data[0, -1])  
        back_center = float(self.mesh_data[-1, 2])  

        # Анализ перекоса между передними точками
        lr_diff = abs(right_front - left_front)
        front_avg = (left_front + right_front) / 2
        back_diff = abs(back_center - front_avg)
   
        belt_adjustments = {}
   
        if lr_diff > self.settings['BELT_DELTA_THRESHOLD']:
            teeth = self.calculate_belt_adjustment(lr_diff)
            # Определяем какой край ниже по модулю
            lower_side = 'front_right' if abs(right_front) > abs(left_front) else 'front_left'
   
            belt_adjustments[lower_side] = {
                'name': 'Передний правый' if lower_side == 'front_right' else 'Передний левый',
                'diff': lr_diff,
                'direction': 'вверх',
                'action': 'Натянуть', 
                'teeth': teeth
            }

        # Анализ заднего вала
        if back_diff > self.settings['BELT_DELTA_THRESHOLD']:
            back_teeth = self.calculate_belt_adjustment(back_diff)
            belt_adjustments['back'] = {
                'name': 'Задний',
                'diff': back_diff,
                'direction': 'вверх' if abs(back_center) > abs(front_avg) else 'вниз',
                'action': 'Натянуть' if abs(back_center) > abs(front_avg) else 'Ослабить',
                'teeth': back_teeth
            }

        return belt_adjustments

    def calculate_belt_adjustment(self, diff_mm: float) -> int:
        if diff_mm <= 0.4:
            return 0
        if 0.4 < diff_mm < 0.8:
            return 1
        return max(1, int(diff_mm / self.settings['BELT_TOOTH_MM']))

    def get_screw_adjustments(self) -> Dict[str, Tuple[int, str, str]]:
        """Расчет регулировок винтов относительно средней высоты"""
        corners = {}
        
        # Позиции углов
        positions = [
            ("Передний левый", (0, 0)),
            ("Передний правый", (0, -1)), 
            ("Задний левый", (-1, 0)),
            ("Задний правый", (-1, -1))
        ]
        
        mean_height = np.mean(self.mesh_data)  # Средняя высота
        corner_heights = {name: float(self.mesh_data[i, j]) for name, (i, j) in positions}
        
        print("\n=== ТЕКСТОВЫЕ РЕКОМЕНДАЦИИ ===")
        print(f"Средняя высота: {mean_height}")
        
        SCREW_PITCH_MM = 0.7  # Константа шага винта
        DEGREES_PER_01MM = 5.14  # Константа градусов на 0.1 мм
        
        for corner_name, (i, j) in positions:
            height = corner_heights[corner_name]
            diff = height - mean_height
            
            print(f"\n{corner_name}:")
            print(f"Высота: {height}")
            print(f"Разница со средней: {diff}")
            print(f"Условие diff < -0.1: {diff < -0.1}")
            print(f"Условие diff > 0.1: {diff > 0.1}")
            
            if abs(diff) > 0.1:  # Учитываем только значительные отклонения
                # Расчет градусов и минут
                total_degrees = abs(diff) * 100 * 5.14  # DEGREES_PER_01MM = 5.14
                minutes = int(total_degrees * 60 / 360)
                
                # Создаем текст с учетом настроек
                rotation_text = []
                if self.settings['SHOW_MINUTES']:
                    rotation_text.append(f"{minutes} минут")
                if self.settings['SHOW_DEGREES']:
                    degrees = minutes * 6  # 1 минута = 6 градусов
                    rotation_text.append(f"{degrees}°")
                
                rotation = " / ".join(rotation_text) if rotation_text else f"{minutes} минут"
                
                if diff > 0:  # Высокий угол — опускание
                    print("РЕШЕНИЕ: Опустить угол (по часовой)")
                    corners[corner_name] = (minutes, "по часовой", "опускание")
                else:  # Низкий угол — подъем
                    print("РЕШЕНИЕ: Поднять угол (против часовой)")
                    corners[corner_name] = (minutes, "против часовой", "подъём")
            else:  # Угол в пределах нормы
                print("РЕШЕНИЕ: В норме")
                corners[corner_name] = (0, "", "норма")
        
        return corners

    def get_tape_adjustments(self) -> Dict[str, int]:
        """Анализ точек для наклейки скотча"""
        tape_points = {}
        mean_height = np.mean(self.mesh_data)

        for i in range(self.mesh_data.shape[0]):
            for j in range(self.mesh_data.shape[1]):
                height = float(self.mesh_data[i, j])
                diff = mean_height - height
                if diff > 0.05:  # Уменьшаем порог для более точной регулировки
                    # Вычисляем разницу и количество слоев
                    layers = max(1, int(np.ceil(abs(diff) / self.settings['TAPE_THICKNESS'])))
                    position = f"{i+1}{chr(65+j)}"  # Формат: номер строки + буква столбца
                    tape_points[position] = layers

        return tape_points

    def generate_tape_recommendations(self):
        """Генерация рекомендаций по наклейке скотча."""
        recommendations = []
        mean_height = np.mean(self.mesh_data)
        for i in range(self.mesh_data.shape[0]):
            for j in range(self.mesh_data.shape[1]):
                height = self.mesh_data[j, i]
                if height < mean_height - 0.05:
                    diff = mean_height - height
                    layers = max(1, int(np.ceil(diff / 0.1)))
                    recommendations.append(f"Позиция {i+1}{chr(65+j)}: {layers} слоев скотча")
        return recommendations

    def plot_adjustments(self) -> None:
        """Построение графика регулировок."""
        if self.mesh_data is None:
            return

        # Создаем фигуру и оси
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')

        # Рисуем сетку и точки
        self.plot_mesh_heatmap(ax)

        # Получаем данные регулировок
        adjustments = self.get_screw_adjustments()

        # Позиции винтов
        positions = {
            'Передний левый': (0, 0),
            'Передний правый': (4, 0),
            'Задний левый': (0, 4),
            'Задний правый': (4, 4)
        }

        mean_height = np.mean(self.mesh_data)
        
        # Список для хранения данных анимации
        screw_data = []

        # Добавляем винты и их обозначения
        for corner_name, (x, y) in positions.items():
            if corner_name in adjustments:
                minutes, rotation_direction, action = adjustments[corner_name]
                if action != "норма":
                    # Рисуем круг
                    circle = plt.Circle((x, y), 0.3, fill=False, color='black')
                    ax.add_artist(circle)

                    # Определяем цвет стрелки
                    color = 'red' if rotation_direction == "по часовой" else 'green'

                    # Рисуем стрелку
                    if rotation_direction == "по часовой":
                        arrow = patches.Arc((x, y), 0.4, 0.4, theta1=0, theta2=45,
                                         color=color, linewidth=2)
                        ax.add_patch(arrow)
                        ax.arrow(x + 0.2, y + 0.2, 0.05, -0.05, head_width=0.1,
                                head_length=0.1, fc=color, ec=color)
                    else:
                        arrow = patches.Arc((x, y), 0.4, 0.4, theta1=180, theta2=225,
                                         color=color, linewidth=2)
                        ax.add_patch(arrow)
                        ax.arrow(x - 0.2, y - 0.2, -0.05, 0.05, head_width=0.1,
                                head_length=0.1, fc=color, ec=color)

                    # Добавляем текст стрелки в центр круга
                    ax.text(x, y, rotation_text,
                            ha='center', va='center',
                            fontsize=20,
                            color=color,
                            fontweight='bold')

                    # Добавляем текст с градусами/минутами и направлением
                    label_text = f"{rotation}\n{rotation_direction}"
                    ax.text(x, y - 0.7, label_text,
                            ha='center', va='center',
                            fontsize=8, color=color,
                            fontweight='bold')
            else:
                ax.text(x, y - 0.7, '✓ Норма',
                        ha='center', va='center',
                        fontsize=8,
                        color='gray')

        # Добавляем легенду
        red_patch = patches.Patch(color='red', label='По часовой')
        green_patch = patches.Patch(color='green', label='Против часовой')
        ax.legend(handles=[red_patch, green_patch], loc='upper right')

        # Отображаем график
        plt.tight_layout()
        return fig

    def on_window_resize(self, event):
        """Обработчик изменения размера окна"""
        if hasattr(self, 'last_resize_time'):
            # Проверяем, прошло ли достаточно времени с последнего ресайза
            if time.time() - self.last_resize_time < 0.3:  # Задержка в секундах
                return
        self.last_resize_time = time.time()
        
        # Перезагружаем изображение с новым размером
        self.reload_belt_image()

    def reload_belt_image(self):
        """Перезагрузка изображения с новым размером"""
        try:
            # Находим тег изображения в тексте
            image_index = "1.0"
            while True:
                image_index = self.text_widget.index(f"{image_index}+1c")
                if image_index == "end":
                    break
                    
                image_names = self.text_widget.image_names(image_index)
                if image_names:
                    # Открываем и масштабируем изображение
                    belt_image = Image.open(self.belt_image_path)
                    widget_width = self.text_widget.winfo_width()
                    desired_width = int(widget_width * 0.8)
                    aspect_ratio = belt_image.size[1] / belt_image.size[0]
                    desired_height = int(desired_width * aspect_ratio)
                    
                    belt_image = belt_image.resize((desired_width, desired_height), Image.Resampling.LANCZOS)
                    belt_photo = ImageTk.PhotoImage(belt_image)
                    
                    # Обновляем изображение
                    self.text_widget.image = belt_photo
                    self.text_widget.delete(image_index)
                    self.text_widget.image_create(image_index, image=belt_photo)
                    break
        except Exception as e:
            pass  # Игнорируем ошибки при ресайзе

    def draw_visualization(self) -> None:
        """Отрисовка визуальных рекомендаций."""
        fig = plt.figure(figsize=(12, 12))
        
        # Карта проблемных зон
        ax1 = plt.subplot2grid((3, 2), (0, 0))
        self.draw_bed_scheme(ax1)
        ax1.set_title("Карта проблемных зон")
        
        # Регулировка винтов стола
        ax2 = plt.subplot2grid((3, 2), (0, 1))
        self.draw_screw_adjustment(ax2)
        ax2.set_title("Регулировка винтов стола\n(вид сверху принтера)")
        
        # Схема регулировки Z-валов
        ax3 = plt.subplot2grid((3, 2), (1, 0))
        self.draw_belt_visualization(ax3)
        ax3.set_title("Схема регулировки Z-валов\n(вид снизу принтера)")
        
        # Схема наклеивания скотча
        ax4 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)
        self.draw_tape_scheme(ax4)
        ax4.set_title("Схема наклеивания алюминиевого скотча\n(числа = количество слоев)")

    def show_success_dialog(self, message):
        dialog = tk.Toplevel(self.root)
        dialog.title("Успех")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Центрируем диалог относительно главного окна
        dialog.geometry(f"+{self.root.winfo_x() + 50}+{self.root.winfo_y() + 50}")
        
        label = ttk.Label(dialog, text=message, wraplength=350)
        label.pack(pady=20)
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def open_file_location():
            import subprocess
            file_path = message.split(" ")[-1]  # Получаем путь к файлу из сообщения
            subprocess.run(['explorer', '/select,', file_path])
            
        ttk.Button(button_frame, text="Открыть папку с файлом", command=open_file_location).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="OK", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.focus_set()

    def get_printer_cfg(self):
        """Получение файла printer.cfg."""
        try:
            # Создаем SSH клиент
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Подключаемся к принтеру
            ssh.connect(self.settings['SSH_HOST'], 
                       username=self.settings['SSH_USERNAME'], 
                       password=self.settings['SSH_PASSWORD'])
            
            # Создаем SCP клиент
            scp = SCPClient(ssh.get_transport())
            
            # Сохраняем в директорию config
            local_path = os.path.join(self.working_dirs['config'], 'printer.cfg')
            print(f"Saving printer.cfg to: {local_path}")  # Для отладки
            
            # Получаем файл printer.cfg
            scp.get("/opt/config/printer.cfg", local_path)
            
            # Закрываем соединение
            scp.close()
            ssh.close()
            
            self.show_success_dialog(f"Файл printer.cfg успешно загружен в {local_path}")
            return True
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось получить файл printer.cfg: {str(e)}")
            return False

    def get_shaper_files(self):
        """Получение файлов шейперов из /tmp/."""
        try:
            # Создаем SSH клиент
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Подключаемся к принтеру
            ssh.connect(self.settings['SSH_HOST'], 
                       username=self.settings['SSH_USERNAME'], 
                       password=self.settings['SSH_PASSWORD'])
            
            # Создаем SCP клиент
            scp = SCPClient(ssh.get_transport())
            
            # Сохраняем в директорию shaper_data
            shaper_dir = self.working_dirs['shaper_data']
            print(f"Saving shaper files to: {shaper_dir}")  # Для отладки
            
            # Получаем файлы шейпера
            pattern = "/tmp/calibration_data_*.csv"
            print(f"Searching for files: {pattern}")  # Для отладки
            
            # Получаем список файлов по шаблону
            stdin, stdout, stderr = ssh.exec_command(f"ls {pattern}")
            remote_files = stdout.read().decode().strip().split("\n")
            print(f"Found files: {remote_files}")  # Для отладки
            
            downloaded_files = []
            for remote_file in remote_files:
                if remote_file:  # Проверяем что файл существует
                    local_file = os.path.join(shaper_dir, os.path.basename(remote_file))
                    print(f"Downloading {remote_file} to {local_file}")  # Для отладки
                    scp.get(remote_file, local_file)
                    downloaded_files.append(local_file)
            
            # Закрываем соединение
            scp.close()
            ssh.close()

            if downloaded_files:
                self.show_success_dialog(f"Файлы шейпера успешно загружены в {shaper_dir}")
                return True
            else:
                messagebox.showwarning("Предупреждение", "Не найдено файлов шейпера")
                return False
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось получить файлы шейпера: {str(e)}")
            return False

    def connect_ssh(self):
        """Подключение к принтеру по SSH."""
        try:
            # Сохраняем настройки SSH
            self.settings['SSH_HOST'] = self.ssh_host_entry.get()
            self.settings['SSH_USERNAME'] = self.ssh_username_entry.get()
            self.settings['SSH_PASSWORD'] = self.ssh_password_entry.get()

            # Закрываем предыдущее соединение если есть
            if hasattr(self, 'ssh_client') and self.ssh_client:
                try:
                    self.ssh_client.close()
                except:
                    pass

            try:
                # Пробуем создать SSH подключение с bcrypt
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.ssh_client.connect(
                    self.settings['SSH_HOST'],
                    port=22,
                    username=self.settings['SSH_USERNAME'],
                    password=self.settings['SSH_PASSWORD'],
                    timeout=10,
                    allow_agent=False,
                    look_for_keys=False
                )
            except ImportError as e:
                if "_bcrypt" in str(e):
                    # Если _bcrypt не доступен, используем более простой способ
                    transport = paramiko.Transport((self.settings['SSH_HOST'], 22))
                    transport.connect(username=self.settings['SSH_USERNAME'],
                                   password=self.settings['SSH_PASSWORD'])
                    self.ssh_client = paramiko.SSHClient()
                    self.ssh_client._transport = transport
                else:
                    raise

            # Проверяем подключение
            stdin, stdout, stderr = self.ssh_client.exec_command('echo "test"')
            if stdout.channel.recv_exit_status() == 0:
                messagebox.showinfo("Успех", "Подключение установлено успешно")
                self.get_config_button.config(state=tk.NORMAL)
                self.get_shaper_button.config(state=tk.NORMAL)
            else:
                raise Exception("Не удалось выполнить тестовую команду")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка подключения: {str(e)}")
            if hasattr(self, 'ssh_client'):
                try:
                    self.ssh_client.close()
                except:
                    pass
            self.ssh_client = None
            self.get_config_button.config(state=tk.DISABLED)
            self.get_shaper_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    analyzer = BedLevelAnalyzer()
    analyzer.run()
