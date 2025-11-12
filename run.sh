#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAND_SEG_DIR="$SCRIPT_DIR/scr/land_segmentation"
VENV_DIR="$LAND_SEG_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Ошибка: виртуальное окружение не найдено. Сначала запустите ./install.sh"
    exit 1
fi

# Активируем виртуальное окружение и запускаем main.py
source "$VENV_DIR/bin/activate"
python "$LAND_SEG_DIR/main.py" "$@"