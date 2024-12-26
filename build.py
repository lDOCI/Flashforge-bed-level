import os
import platform
import subprocess
import shutil

def build_for_platform():
    """Сборка приложения для текущей платформы."""
    
    # Определяем имя исполняемого файла в зависимости от платформы
    if platform.system() == "Windows":
        exe_name = "FlashforgeA5MAnalyzer.exe"
    else:
        exe_name = "FlashforgeA5MAnalyzer"

    # Создаем директорию для сборки если её нет
    os.makedirs("dist", exist_ok=True)

    # Команда для сборки
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name", exe_name,
        "--add-data", "*.json;.",  # Добавляем конфигурационные файлы
        "bed_level_analyzer.py"
    ]

    # Добавляем иконку для Windows
    if platform.system() == "Windows":
        if os.path.exists("icon.ico"):
            cmd.extend(["--icon", "icon.ico"])

    # Запускаем сборку
    subprocess.run(cmd, check=True)

    # Создаем zip архив с приложением
    platform_name = platform.system().lower()
    zip_name = f"FlashforgeA5MAnalyzer_{platform_name}.zip"
    
    if os.path.exists(zip_name):
        os.remove(zip_name)
    
    shutil.make_archive(
        f"FlashforgeA5MAnalyzer_{platform_name}",
        'zip',
        'dist'
    )

    print(f"Сборка завершена. Создан файл: {zip_name}")

if __name__ == "__main__":
    build_for_platform()
