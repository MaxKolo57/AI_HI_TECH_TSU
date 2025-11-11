# core/segmentation.py
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

from .processing import tile_raster_to_pngs, reconstruct_mask_from_tiles, save_mask_as_geotiff


def otsu_threshold(image_gray):
    """Вычисление порога Отсу"""
    hist, _ = np.histogram(image_gray.ravel(), bins=256, range=(0, 255))
    total = image_gray.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    max_var = 0.0
    threshold = 0

    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i
    return threshold


def simple_tile_segmentation(tile_png_path, mask_out_path):
    """Сегментация одного тайла с помощью порога Отсу"""
    img = np.array(Image.open(tile_png_path).convert("RGB"))
    gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
    th = otsu_threshold(gray)
    mask = (gray > th).astype(np.uint8)
    np.save(mask_out_path, mask)
    return mask_out_path


class SegmentationWorker(QThread):
    """Фоновый поток для сегментации ортофотоплана"""
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)

    def __init__(self, tif_path, work_dir, tile_size=512):
        super().__init__()
        self.tif_path = Path(tif_path)
        self.work_dir = Path(work_dir)  # временная директория
        self.tile_size = tile_size

    def run(self):
        try:
            self.message.emit("Нарезка ортофотоплана на тайлы...")
            tiles_dir = self.work_dir

            # Очистка временной директории
            if tiles_dir.exists():
                for child in tiles_dir.glob("*"):
                    if child.is_dir():
                        for c in child.glob("*"):
                            c.unlink()
                        child.rmdir()
                    else:
                        child.unlink()
            tiles_dir.mkdir(parents=True, exist_ok=True)

            n_tiles = tile_raster_to_pngs(str(self.tif_path), tiles_dir, tile_size=self.tile_size)
            self.message.emit(f"Тайлов создано: {n_tiles}")
            self.progress.emit(10)

            images_dir = tiles_dir / "images"
            masks_dir = tiles_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

            image_files = sorted(images_dir.glob("tile_*.png"))
            total = len(image_files)
            if total == 0:
                raise RuntimeError("Не найдены PNG тайлы после нарезки.")

            # Сегментация по тайлам
            for i, imgf in enumerate(image_files, start=1):
                mask_out = masks_dir / (imgf.name.replace("tile_", "mask_").replace(".png", ".npy"))
                simple_tile_segmentation(str(imgf), str(mask_out))
                pct = 10 + int(80 * (i / total))
                self.progress.emit(pct)
                self.message.emit(f"Сегментация тайла {i}/{total}")

            # Сборка полной маски
            self.message.emit("Сборка маски...")
            mask_npy = str(tiles_dir / "reconstructed_mask.npy")
            reconstruct_mask_from_tiles(tiles_dir, mask_npy)
            self.progress.emit(95)

            # Сохранение маски как GeoTIFF
            self.message.emit("Сохранение маски как GeoTIFF...")
            mask_tif = str(tiles_dir / "mask.tif")
            save_mask_as_geotiff(mask_npy, tiles_dir / "meta", mask_tif)
            self.progress.emit(100)
            self.message.emit("Готово: маска сохранена.")

            self.finished_ok.emit({
                "tiles_dir": str(tiles_dir),
                "mask_npy": mask_npy,
                "mask_tif": mask_tif
            })

        except Exception as e:
            self.message.emit(f"ОШИБКА: {str(e)}")
            self.finished_ok.emit({"error": str(e)})
