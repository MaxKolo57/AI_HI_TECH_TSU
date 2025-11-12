#!/bin/bash

set -e  # Прервать выполнение при ошибке

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAND_SEG_DIR="$SCRIPT_DIR/scr/land_segmentation"

# Создаём .venv внутри land_segmentation
python3 -m venv "$LAND_SEG_DIR/.venv"

# Активируем виртуальное окружение
source "$LAND_SEG_DIR/.venv/bin/activate"

# Обновляем pip
pip install --upgrade pip

# Устанавливаем зависимости
pip install -r "$LAND_SEG_DIR/requirements.txt"

echo "Установка завершена. Виртуальное окружение создано в scr/land_segmentation/.venv"