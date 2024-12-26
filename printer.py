import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
from matplotlib import cm
from matplotlib import use as plot_in_window
import tkinter as tk
from tkinter import filedialog, messagebox

# принудительно заставляем рисовать в окне TkAgg - это соответствующий бэкенд
plot_in_window('TkAgg')

def load_data(path):
    mesh_data = []
    row = []
    line_count = 0

    with open(path, 'r') as file:
        in_points_section = False
        for line in file:
            if 'points =' in line:
                in_points_section = True
                continue
            # нашли mesh - считываем
            elif in_points_section:
                # нам нужны только эти 5 строк из файла
                if not line.startswith("#*#"):
                    break
                if line_count > 4:
                    return mesh_data
                # оставляем без всякой фигни в начале
                line = line.strip().replace(',', '').split(' ')[3:]
                try:
                    row = [float(x) for x in line]
                finally:
                    line_count += 1
                    if row:
                        mesh_data.append(row)

    if not mesh_data or line_count < 5:
        print("Файл не тот!")

    return mesh_data

def calculate_max_delta_absolute(data):
    max_height = np.max(data)
    min_height = np.min(data)
    max_delta = abs(max_height - min_height)
    return max_delta

def draw_2d_graph(data):
    fig, ax = plt.subplots()
    cmap = cm.coolwarm_r  # Инвертированная цветовая карта
    cax = ax.matshow(data, cmap=cmap)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val:.6f}', ha='center', va='center', color='black')

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(np.arange(data.shape[1]))
    ax.set_yticklabels(np.arange(data.shape[0]))

    # Отступы для надписей
    ax.text(-0.1, 1.1, 'Левый дальний', va='center', transform=ax.transAxes)
    ax.text(0.8, 1.1, 'Правый дальний', va='center', transform=ax.transAxes)
    ax.text(-0.1, -0.1, 'Левый ближний', va='center', transform=ax.transAxes)
    ax.text(0.8, -0.1, 'Правый ближний', va='center', transform=ax.transAxes)

    plt.show()

def draw_3d_graph(data):
    # Создание сетки для координат
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)

    # Создание цветового градиента
    cmap = cm.coolwarm_r  # Инвертированная цветовая карта
    # Создание 3D-графика
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data, cmap=cmap, edgecolor='none', rstride=1, cstride=1, antialiased=True)

    # Добавление цветовой шкалы
    fig.colorbar(surf, shrink=1, aspect=6)

    # Настройка осей
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(np.arange(data.shape[1]))
    ax.set_yticklabels(np.arange(data.shape[0]))

    # Отражение сетки по оси X
    ax.invert_xaxis()

    # Добавление надписей к углам графика
    ax.text(0, 0, -1.5, 'Левый дальний', color='black')
    ax.text(data.shape[1]-1, 0, -1.5, 'Правый дальний', color='black')
    ax.text(0, data.shape[0]-1, -1.5, 'Левый ближний', color='black')
    ax.text(data.shape[1]-1, data.shape[0]-1, -1.5, 'Правый ближний', color='black')

    delta = calculate_max_delta_absolute(data)
    # Установка заголовка графика с матрицей высот
    ax.text2D(0.05, 0.95, f'MAX ∆ - {delta} мм', color='red', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # Установка заголовка окна
    fig.canvas.manager.set_window_title('Mesh level F5M Adv')

    plt.show()

def select_file():
    file_path = filedialog.askopenfilename(title="Выберите файл printer.cfg", filetypes=[("Config files", "*.cfg")])
    return file_path

def on_load_button_click():
    global data
    file_path = select_file()
    if file_path:
        data = load_data(file_path)
        if data:
            data = np.array(data)
            messagebox.showinfo("Успех", "Файл успешно загружен!")
        else:
            messagebox.showerror("Ошибка", "Файл некорректен!")
    else:
        messagebox.showwarning("Предупреждение", "Файл не выбран!")

def on_2d_button_click():
    if 'data' in globals():
        draw_2d_graph(data)
    else:
        messagebox.showwarning("Предупреждение", "Сначала загрузите файл!")

def on_3d_button_click():
    if 'data' in globals():
        draw_3d_graph(data)
    else:
        messagebox.showwarning("Предупреждение", "Сначала загрузите файл!")

def create_gui():
    root = tk.Tk()
    root.title("Визуализатор уровня сетки")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    load_button = tk.Button(frame, text="Загрузить файл", command=on_load_button_click)
    load_button.grid(row=0, column=0, padx=5, pady=5)

    button_2d = tk.Button(frame, text="2D График", command=on_2d_button_click)
    button_2d.grid(row=0, column=1, padx=5, pady=5)

    button_3d = tk.Button(frame, text="3D График", command=on_3d_button_click)
    button_3d.grid(row=0, column=2, padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
