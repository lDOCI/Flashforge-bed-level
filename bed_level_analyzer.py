import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import use as plot_in_window
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Tuple, Dict
from matplotlib.patches import Rectangle, Arrow, Circle
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Force matplotlib to use TkAgg backend
plot_in_window('TkAgg')

# Настройка поддержки русского языка в matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

class BedLevelAnalyzer:
    def __init__(self):
        self.mesh_data = None
        self.recommendations = []
        self.max_delta = None
        self.create_gui()

    def load_data(self, path: str) -> np.ndarray:
        """Загрузка и парсинг данных уровня стола из конфигурационного файла."""
        mesh_data = []
        row = []
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
                            return np.array(mesh_data)
                        line = line.strip().replace(',', '').split(' ')[3:]
                        try:
                            row = [float(x) for x in line]
                        finally:
                            line_count += 1
                            if row:
                                mesh_data.append(row)

            if not mesh_data or line_count < 5:
                raise ValueError("Неверный формат файла или пустые данные")

            return np.array(mesh_data)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
            return None

    def analyze_bed_level(self) -> None:
        """Анализ данных уровня стола и генерация рекомендаций."""
        if self.mesh_data is None:
            return

        self.recommendations.clear()
        
        # Расчет ключевых метрик
        self.max_delta = np.max(self.mesh_data) - np.min(self.mesh_data)
        std_dev = np.std(self.mesh_data)
        mean_height = np.mean(self.mesh_data)
        
        # Анализ углов
        corners = [
            self.mesh_data[0, 0],    # Передний левый
            self.mesh_data[0, -1],   # Передний правый
            self.mesh_data[-1, 0],   # Задний левый
            self.mesh_data[-1, -1]   # Задний правый
        ]
        corner_diff = max(corners) - min(corners)

        # Анализ центра стола
        center_idx = self.mesh_data.shape[0] // 2
        center_region = self.mesh_data[center_idx-1:center_idx+2, center_idx-1:center_idx+2]
        center_mean = np.mean(center_region)
        corners_mean = np.mean(corners)
        center_deviation = abs(center_mean - corners_mean)

        # Генерация рекомендаций на основе анализа
        if center_deviation > 0.3 and corner_diff <= 0.2:
            if center_mean > corners_mean:
                self.recommendations.append("⚠️ Обнаружен горб в центре стола (%.3f мм выше краёв)" % center_deviation)
                self.recommendations.append("🔧 Рекомендации:")
                self.recommendations.append("  1. Проверьте нагревательный стол на деформацию")
                self.recommendations.append("  2. Убедитесь, что стол не изогнут из-за перегрева")
                self.recommendations.append("  3. Возможно потребуется замена стола или использование стекла")
            else:
                self.recommendations.append("⚠️ Обнаружена яма в центре стола (%.3f мм ниже краёв)" % center_deviation)
                self.recommendations.append("🔧 Рекомендации:")
                self.recommendations.append("  1. Проверьте поверхность стола на вмятины")
                self.recommendations.append("  2. Рассмотрите использование стекла для компенсации")
                self.recommendations.append("  3. При необходимости замените стол")

        if self.max_delta > 0.5:
            if corner_diff > 0.3:
                self.recommendations.append("⚠️ Критично: Стол сильно не выровнен. Рекомендуется полная перекалибровка.")
            else:
                self.recommendations.append("⚠️ Внимание: Большой перепад высот, но углы выровнены относительно друг друга.")
            
            # Проверка проблем с натяжением ремня
            if std_dev > 0.3:
                self.recommendations.append("🔧 Проверьте и отрегулируйте натяжение ремня. Возможно, требуется перекинуть зубья ремня.")
                
        if corner_diff > 0.3:
            self.recommendations.append("🔧 Требуется выравнивание углов:")
            for i, (corner, value) in enumerate([
                ("Передний левый", corners[0]),
                ("Передний правый", corners[1]),
                ("Задний левый", corners[2]),
                ("Задний правый", corners[3])
            ]):
                if abs(value - mean_height) > 0.2:
                    direction = "по часовой стрелке" if value > mean_height else "против часовой стрелки"
                    self.recommendations.append(f"  • {corner}: {direction} ({abs(value - mean_height):.3f}мм)")
        
        # Проверка локальных проблем
        high_spots = np.where(self.mesh_data > mean_height + 0.2)
        low_spots = np.where(self.mesh_data < mean_height - 0.2)
        
        if len(high_spots[0]) > 0:
            self.recommendations.append("📌 Обнаружены высокие точки:")
            for y, x in zip(*high_spots):
                self.recommendations.append(f"  • Позиция ({x}, {y}): Проверьте наличие мусора или нажмите на стол")
                
        if len(low_spots[0]) > 0:
            self.recommendations.append("📌 Обнаружены низкие точки:")
            for y, x in zip(*low_spots):
                self.recommendations.append(f"  • Позиция ({x}, {y}): Рекомендуется наклеить алюминиевый скотч")

        # Обновление рекомендаций в интерфейсе
        self.update_recommendations()

    def draw_2d_graph(self) -> None:
        """Отрисовка 2D тепловой карты уровня стола."""
        if self.mesh_data is None:
            messagebox.showwarning("Предупреждение", "Пожалуйста, сначала загрузите данные")
            return

        plt.figure(figsize=(10, 8))
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
        """Отрисовка 3D поверхности уровня стола."""
        if self.mesh_data is None:
            messagebox.showwarning("Предупреждение", "Пожалуйста, сначала загрузите данные")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Создаем расширенную сетку для корректного отображения всех ячеек
        x = np.linspace(-0.5, 4.5, 6)
        y = np.linspace(-0.5, 4.5, 6)
        X, Y = np.meshgrid(x, y)

        # Расширяем данные, дублируя крайние значения
        extended_data = np.zeros((6, 6))
        extended_data[:-1, :-1] = self.mesh_data
        extended_data[-1, :] = extended_data[-2, :]
        extended_data[:, -1] = extended_data[:, -2]

        vmin, vmax = np.min(self.mesh_data), np.max(self.mesh_data)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Отображаем поверхность с правильной ориентацией данных
        surf = ax.plot_surface(X, Y, extended_data,  # Убрали np.flipud
                             cmap=cm.coolwarm_r,
                             linewidth=0.5,
                             antialiased=True,
                             norm=norm)

        # Добавляем линии сетки
        for i in range(6):
            ax.plot([i-0.5, i-0.5], [-0.5, 4.5], [vmin, vmin], 'k-', alpha=0.2)
            ax.plot([-0.5, 4.5], [i-0.5, i-0.5], [vmin, vmin], 'k-', alpha=0.2)

        plt.colorbar(surf)
        ax.view_init(elev=30, azim=45)

        # Устанавливаем пределы и метки
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))

        z_min = vmin - 0.1
        # Обновляем позиции подписей
        ax.text(0, 0, z_min, "Передний левый")
        ax.text(4, 0, z_min, "Передний правый")
        ax.text(0, 4, z_min, "Задний левый")
        ax.text(4, 4, z_min, "Задний правый")

        plt.title(f"3D карта уровня стола (Макс. отклонение: {self.max_delta:.6f}мм)")
        plt.show()

    def create_visual_recommendations(self) -> None:
        """Создание визуальных рекомендаций."""
        if self.mesh_data is None:
            return

        # Создаем новое окно для визуальных рекомендаций
        rec_window = tk.Toplevel(self.root)
        rec_window.title("Визуальные рекомендации")
        rec_window.geometry("800x600")

        # Создаем Figure для matplotlib
        fig = Figure(figsize=(8, 6))
        
        # Создаем канвас
        canvas = FigureCanvasTkAgg(fig, master=rec_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Добавляем подграфики
        ax1 = fig.add_subplot(221)  # Схема стола
        ax2 = fig.add_subplot(222)  # Схема винтов
        ax3 = fig.add_subplot(223)  # Схема ремней
        ax4 = fig.add_subplot(224)  # Схема скотча

        # Рисуем схему стола с проблемными зонами
        self.draw_bed_scheme(ax1)
        
        # Рисуем схему регулировки винтов
        self.draw_screw_adjustments(ax2)
        
        # Рисуем схему натяжения ремней
        self.draw_belt_scheme(ax3)
        
        # Рисуем схему наклеивания скотча
        self.draw_tape_scheme(ax4)

        fig.tight_layout()
        canvas.draw()

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

    def draw_screw_adjustments(self, ax):
        """Отрисовка схемы регулировки винтов."""
        ax.clear()
        # Рисуем контур стола
        bed = Rectangle((0, 0), 4, 4, fill=False, color='black')
        ax.add_patch(bed)
        
        # Добавляем стрелки регулировки для каждого угла
        corners = [
            (0, 0, "Передний левый"),
            (4, 0, "Передний правый"),
            (0, 4, "Задний левый"),
            (4, 4, "Задний правый")
        ]
        
        for x, y, label in corners:
            if abs(self.mesh_data[int(y//1), int(x//1)] - np.mean(self.mesh_data)) > 0.2:
                direction = 1 if self.mesh_data[int(y//1), int(x//1)] < np.mean(self.mesh_data) else -1
                ax.add_patch(plt.Circle((x, y), 0.3, color='orange', fill=False))
                ax.text(x, y-0.5, f"{'↻' if direction > 0 else '↺'}\n1/4", ha='center')

        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_title("Регулировка винтов")
        ax.set_aspect('equal')

    def draw_belt_scheme(self, ax):
        """Отрисовка схемы натяжения ремней."""
        ax.clear()
        if np.std(self.mesh_data) > 0.3:
            ax.text(0.5, 0.5, "⚠️ Проверьте натяжение ремней!\n\n"
                   "1. Ослабьте винты крепления\n"
                   "2. Натяните ремни\n"
                   "3. Затяните винты",
                   ha='center', va='center')
        ax.set_title("Натяжение ремней")
        ax.axis('off')

    def draw_tape_scheme(self, ax):
        """Отрисовка схемы наклеивания скотча."""
        ax.clear()
        # Рисуем контур стола
        bed = Rectangle((0, 0), 4, 4, fill=False, color='black')
        ax.add_patch(bed)
        
        # Отмечаем места для наклеивания скотча
        low_spots = np.where(self.mesh_data < np.mean(self.mesh_data) - 0.2)
        for y, x in zip(*low_spots):
            rect = Rectangle((x-0.3, y-0.3), 0.6, 0.6, color='yellow', alpha=0.5)
            ax.add_patch(rect)

        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_title("Места наклеивания скотча")
        ax.set_aspect('equal')

    def create_gui(self) -> None:
        """Создание главного окна интерфейса."""
        self.root = tk.Tk()
        self.root.title("Анализатор уровня стола Flashforge A5M")
        self.root.geometry("800x600")

        # Создание основной рамки с отступами
        main_frame = ttk.Frame(self.root, padding="10")
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
        
        self.rec_text = tk.Text(rec_frame, wrap=tk.WORD, height=15)
        self.rec_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(rec_frame, orient=tk.VERTICAL, command=self.rec_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.rec_text.configure(yscrollcommand=scrollbar.set)

        # Настройка весов сетки
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        rec_frame.columnconfigure(0, weight=1)
        rec_frame.rowconfigure(0, weight=1)

    def load_file(self) -> None:
        """Обработка загрузки файла."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Конфигурационные файлы", "*.cfg"), ("Все файлы", "*.*")]
        )
        if file_path:
            self.file_path_var.set(f"Файл: {file_path}")
            self.mesh_data = self.load_data(file_path)
            if self.mesh_data is not None:
                self.analyze_bed_level()

    def update_recommendations(self) -> None:
        """Обновление виджета рекомендаций."""
        self.rec_text.delete(1.0, tk.END)
        if self.max_delta is not None:
            self.rec_text.insert(tk.END, f"📊 Результаты анализа:\n")
            self.rec_text.insert(tk.END, f"Максимальное отклонение: {self.max_delta:.6f}мм\n\n")
            
        self.rec_text.insert(tk.END, "🔍 Рекомендации:\n\n")
        for rec in self.recommendations:
            self.rec_text.insert(tk.END, f"{rec}\n")

    def run(self) -> None:
        """Запуск приложения."""
        self.root.mainloop()

if __name__ == "__main__":
    analyzer = BedLevelAnalyzer()
    analyzer.run()
