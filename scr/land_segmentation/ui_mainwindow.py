# ui_mainwindow.py
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import rasterio
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox, QProgressBar, QTextEdit, QSizePolicy, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from core.segmentation import SegmentationWorker


class OrthoSegApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ortho Segmentation — demo")
        self.resize(1000, 700)

        self.tif_path = None
        self.tmp_dir = None  # временная директория
        self.last_result = None
        self.worker = None

        self._build_ui()

    def closeEvent(self, event):
        """Удаляем временную директорию при закрытии окна"""
        self._cleanup_tmp()
        event.accept()

    def _cleanup_tmp(self):
        """Удаляет текущую временную директорию, если она существует"""
        if self.tmp_dir and Path(self.tmp_dir).exists():
            try:
                shutil.rmtree(self.tmp_dir)
            except Exception as e:
                print(f"Ошибка при удалении временной директории: {e}")
            self.tmp_dir = None

    def _build_ui(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        btn_layout = QVBoxLayout()
        view_layout = QVBoxLayout()

        # Кнопки
        self.btn_load = QPushButton("Загрузить ортофотоплан")
        self.btn_segment = QPushButton("Сегментация ортофотоплана")
        self.btn_save = QPushButton("Сохранить результаты")
        self.btn_exit = QPushButton("Выход")

        self.btn_segment.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.btn_load.clicked.connect(self.on_load)
        self.btn_segment.clicked.connect(self.on_segment)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_exit.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_segment)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_exit)

        # Превью изображения
        self.image_label = QLabel("Здесь будет превью загруженного ортофотоплана")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        view_layout.addWidget(scroll)

        top_layout.addLayout(btn_layout, 0)
        top_layout.addLayout(view_layout, 1)

        # Прогресс и лог
        self.progress = QProgressBar()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(140)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.log)

        self.setLayout(main_layout)

    def log_msg(self, text):
        self.log.append(text)

    def on_load(self):
        """Загрузка нового ортофотоплана"""
        fn, _ = QFileDialog.getOpenFileName(self, "Выберите GeoTIFF файл", ".", "GeoTIFF Files (*.tif *.tiff)")
        if not fn:
            return
        try:
            with rasterio.open(fn) as src:
                w, h = src.width, src.height
                bands = src.count

            self.tif_path = fn
            self.log_msg(f"Файл загружен: {fn} ({w}×{h}, bands={bands})")
            self.show_preview(fn)
            self.btn_segment.setEnabled(True)

            # Создаем временную директорию
            self._cleanup_tmp()
            self.tmp_dir = tempfile.mkdtemp(prefix="ortho_seg_")
            self.log_msg(f"Временная директория создана: {self.tmp_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл: {e}")
            self.log_msg(str(e))
            self.tif_path = None
            self.btn_segment.setEnabled(False)

    def show_preview(self, tif_path):
        """Создание мини-превью TIFF"""
        try:
            with rasterio.open(tif_path) as src:
                w, h = src.width, src.height
                scale = max(1, int(max(w, h) / 1000))
                out_h = max(1, int(h / scale))
                out_w = max(1, int(w / scale))

                img = src.read(out_shape=(src.count, out_h, out_w),
                               resampling=rasterio.enums.Resampling.average)
                arr = np.transpose(img, (1, 2, 0)).astype(np.uint8)

                if arr.shape[2] == 1:
                    qimg = QImage(arr[:, :, 0].copy(), out_w, out_h, out_w, QImage.Format_Grayscale8)
                else:
                    arr_rgb = arr[:, :, :3].copy()
                    qimg = QImage(arr_rgb.data, out_w, out_h, 3 * out_w, QImage.Format_RGB888)

                pix = QPixmap.fromImage(qimg)
                self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.log_msg(f"Ошибка превью: {e}")

    def on_segment(self):
        """Запуск сегментации"""
        if not self.tif_path:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите файл.")
            return

        self.btn_load.setEnabled(False)
        self.btn_segment.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.progress.setValue(0)
        self.log.clear()
        self.log_msg("Запуск процесса сегментации...")

        # SegmentationWorker использует временную директорию
        self.worker = SegmentationWorker(self.tif_path, self.tmp_dir)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.message.connect(self.log_msg)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, info):
        """Обработка завершения сегментации"""
        self.btn_load.setEnabled(True)
        self.btn_segment.setEnabled(True)
        if info.get("error"):
            QMessageBox.critical(self, "Ошибка", info["error"])
            self.log_msg("Сегментация завершилась с ошибкой.")
            self.last_result = None
            self.btn_save.setEnabled(False)
            return

        self.last_result = info
        self.btn_save.setEnabled(True)
        self.log_msg("Сегментация успешно завершена.")

        try:
            mask = np.load(info["mask_npy"])
            img8 = (mask * 255).astype(np.uint8)
            h0, w0 = img8.shape
            qimg = QImage(img8.data, w0, h0, w0, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio))
        except Exception as e:
            self.log_msg(f"Ошибка отображения маски: {e}")

    def on_save(self):
        """Сохранение финального mask.tif в выбранную пользователем папку"""
        if not self.last_result:
            QMessageBox.warning(self, "Нет результатов", "Сначала выполните сегментацию.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Папка для сохранения", ".")
        if not folder:
            return

        folder = Path(folder)
        try:
            mask_src = Path(self.last_result["mask_tif"])
            mask_dst = folder / mask_src.name
            if mask_dst.exists():
                mask_dst.unlink()
            shutil.copy(mask_src, mask_dst)
            self.log_msg(f"Маска сохранена: {mask_dst}")
            QMessageBox.information(self, "Готово", f"Файл сохранен:\n{mask_dst}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            self.log_msg(str(e))
