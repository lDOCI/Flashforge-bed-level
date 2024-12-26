# Flashforge-bed-level
Визуализация выравнивания стола для Flashforge 5M Adventure

## Описание
Flashforge 5M Adventure с завода может иметь неидеальное выравнивание стола.
Эта программа поможет вам настроить уровень стола, регулируя винты по углам.

## Требования
- Python 3.9
- Зависимости из файла requirements.txt

## Установка
1. Клонируйте репозиторий
2. Установите зависимости: `pip install -r requirements.txt`
3. Запустите программу: `python bed_level_analyzer.py`

### Для Windows 7
Если вы используете Windows 7, убедитесь что у вас установлены следующие DLL:
- vcruntime140.dll
- msvcp140.dll
- python39.dll (для Python 3.9)
- libcrypto-1_1.dll
- libssl-1_1.dll

Эти файлы можно найти в папке `dll` этого репозитория. Скопируйте их в папку с программой или в системную папку Windows.

Также можно установить Microsoft Visual C++ Redistributable для Visual Studio 2015-2022:
- [VC_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) для 64-bit Windows
- [VC_redist.x86.exe](https://aka.ms/vs/17/release/vc_redist.x86.exe) для 32-bit Windows

## Использование
1. Запустите программу
2. Следуйте инструкциям на экране для настройки уровня стола
3. Используйте визуальное отображение для корректировки винтов

## Решение проблем
### Windows 7: Ошибка "_bcrypt"
Если вы видите ошибку связанную с "_bcrypt", убедитесь что:
1. Установлены все необходимые DLL файлы
2. Установлен Microsoft Visual C++ Redistributable
3. Используются совместимые версии библиотек из requirements.txt