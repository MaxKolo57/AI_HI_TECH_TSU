# core/processing.py
import json
import math
from pathlib import Path
from PIL import Image
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine


def tile_raster_to_pngs(tif_path, out_dir, tile_size=512, overlap=0):
    """
    Нарезка GeoTIFF на PNG-тайлы + сохранение meta json (x_off, y_off, width, height, transform, crs)
    Возвращает общее количество тайлов.
    """
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    meta_dir = out_dir / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        w, h = src.width, src.height
        transform = src.transform
        crs = src.crs

        nx = math.ceil((w - overlap) / (tile_size - overlap))
        ny = math.ceil((h - overlap) / (tile_size - overlap))

        count = 0
        for iy in range(ny):
            for ix in range(nx):
                x_off = int(ix * (tile_size - overlap))
                y_off = int(iy * (tile_size - overlap))
                win_w = int(min(tile_size, w - x_off))
                win_h = int(min(tile_size, h - y_off))
                if win_w <= 0 or win_h <= 0:
                    continue
                window = Window(x_off, y_off, win_w, win_h)

                arr = src.read(window=window)
                arr_hwc = np.transpose(arr, (1, 2, 0))
                if arr_hwc.shape[2] == 1:
                    img = Image.fromarray(arr_hwc[:, :, 0])
                else:
                    img = Image.fromarray(arr_hwc[:, :, :3].astype(np.uint8))

                fname = f"tile_{iy:04d}_{ix:04d}.png"
                meta = {
                    'x_off': int(x_off),
                    'y_off': int(y_off),
                    'width': int(win_w),
                    'height': int(win_h),
                    'transform': list(rasterio.windows.transform(window, transform).to_gdal()),
                    'crs': str(crs) if crs else None
                }
                img.save(images_dir / fname)
                with open(meta_dir / (fname.replace('.png', '.json')), 'w', encoding='utf-8') as mf:
                    json.dump(meta, mf, ensure_ascii=False, indent=2)
                count += 1

    return count


def tile_raster_to_pngs(tif_path, out_dir, tile_size=512, overlap=0):
    """
    Нарезка GeoTIFF на PNG-тайлы размером строго tile_size × tile_size.
    Правый и нижний края дополняются нулями (zero-padding), если нужно.
    Сохраняет meta json (x_off, y_off, original_width, original_height, transform, crs).
    Возвращает общее количество тайлов.
    """
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    meta_dir = out_dir / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        w, h = src.width, src.height
        transform = src.transform
        crs = src.crs

        # Количество тайлов по ширине и высоте
        nx = math.ceil(w / tile_size)
        ny = math.ceil(h / tile_size)

        count = 0
        for iy in range(ny):
            for ix in range(nx):
                x_off = ix * tile_size
                y_off = iy * tile_size

                # Размер окна в оригинальном изображении (может быть < tile_size у краёв)
                read_w = min(tile_size, w - x_off)
                read_h = min(tile_size, h - y_off)

                if read_w <= 0 or read_h <= 0:
                    continue

                window = Window(x_off, y_off, read_w, read_h)
                arr = src.read(window=window)  # shape: (C, H, W)

                # Создаём тайл tile_size × tile_size
                if arr.ndim == 2:
                    arr = arr[np.newaxis, :, :]  # добавить канал
                c = arr.shape[0]
                full_tile = np.zeros((c, tile_size, tile_size), dtype=arr.dtype)
                full_tile[:, :read_h, :read_w] = arr

                # Преобразуем в (H, W, C)
                arr_hwc = np.transpose(full_tile, (1, 2, 0))

                # Конвертируем в RGB или grayscale
                if arr_hwc.shape[2] == 1:
                    img_array = arr_hwc[:, :, 0].astype(np.uint8)
                    img = Image.fromarray(img_array)
                else:
                    img_array = arr_hwc[:, :, :3].astype(np.uint8)
                    img = Image.fromarray(img_array)

                fname = f"tile_{iy:04d}_{ix:04d}.png"
                # Сохраняем мета с указанием оригинальных координат и размеров
                meta = {
                    'x_off': int(x_off),
                    'y_off': int(y_off),
                    'original_width': int(read_w),   # реальный размер в исходном изображении
                    'original_height': int(read_h),
                    'tile_size': tile_size,
                    'transform': list(rasterio.windows.transform(window, transform).to_gdal()),
                    'crs': str(crs) if crs else None
                }
                img.save(images_dir / fname)
                with open(meta_dir / (fname.replace('.png', '.json')), 'w', encoding='utf-8') as mf:
                    json.dump(meta, mf, ensure_ascii=False, indent=2)
                count += 1

    return count



def reconstruct_mask_from_tiles(tiles_dir, out_path="reconstructed_mask.npy"):
    """Собирает все mask_*.npy в одну большую маску"""
    tiles_dir = Path(tiles_dir)
    mask_dir = tiles_dir / "masks"
    meta_dir = tiles_dir / "meta"

    mask_files = sorted(mask_dir.glob("mask_*.npy"))
    if not mask_files:
        raise FileNotFoundError("No NPY masks found in masks/")

    coords = []
    for f in mask_files:
        fname = f.name.replace("mask_", "tile_").replace(".npy", ".json")
        meta_path = meta_dir / fname
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        coords.append((f, int(meta["x_off"]), int(meta["y_off"]), int(meta["width"]), int(meta["height"])))

    max_x = max(c[1] + c[3] for c in coords)
    max_y = max(c[2] + c[4] for c in coords)
    full_mask = np.zeros((max_y, max_x), dtype=np.uint8)

    for f, x, y, w, h in coords:
        tile_mask = np.load(f)
        h0, w0 = tile_mask.shape
        full_mask[y:y + h0, x:x + w0] = tile_mask[:h0, :w0]

    np.save(out_path, full_mask)
    return out_path


def reconstruct_mask_from_tiles(tiles_dir, out_path="reconstructed_mask.npy"):
    """
    Собирает все mask_*.npy (или изображения) в одну большую маску или RGB-изображение.
    Обрабатывает как монохромные маски (uint8), так и RGB-изображения (uint8).
    """
    tiles_dir = Path(tiles_dir)
    mask_dir = tiles_dir / "masks"
    meta_dir = tiles_dir / "meta"

    mask_files = sorted(mask_dir.glob("mask_*.npy"))
    if not mask_files:
        raise FileNotFoundError("No NPY masks found in masks/")

    coords = []
    for f in mask_files:
        fname = f.name.replace("mask_", "tile_").replace(".npy", ".json")
        meta_path = meta_dir / fname
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        coords.append((f, int(meta["x_off"]), int(meta["y_off"]), int(meta["width"]), int(meta["height"])))

    # Получаем максимальные размеры для конечной маски
    max_x = max(c[1] + c[3] for c in coords)
    max_y = max(c[2] + c[4] for c in coords)
    
    # Определяем, RGB это или одноцветная маска
    first_tile = np.load(coords[0][0])
    if len(first_tile.shape) == 2:  # Если одноцветная маска
        full_mask = np.zeros((max_y, max_x), dtype=np.uint8)
    else:  # Если RGB изображение
        full_mask = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    # Собираем маску или RGB изображение
    for f, x, y, w, h in coords:
        tile = np.load(f)
        h0, w0 = tile.shape[:2]
        
        if len(tile.shape) == 2:  # Одноцветная маска
            full_mask[y:y + h0, x:x + w0] = tile[:h0, :w0]
        elif len(tile.shape) == 3:  # RGB изображение
            full_mask[y:y + h0, x:x + w0, :] = tile[:h0, :w0, :]

    # Убедимся, что директория для сохранения существует
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if full_mask.ndim == 2:  # Если это одноцветная маска
        np.save(out_path, full_mask)
    else:  # Если это RGB изображение
        result_img = Image.fromarray(full_mask)
        result_img.save(out_path.replace('.npy', '.png'))
        np.save(out_path, full_mask)  # <-- добавляем эту строку

    return out_path



def reconstruct_mask_from_tiles(tiles_dir, out_path="reconstructed_mask.npy"):
    """
    Собирает mask_*.npy тайлы (всегда 512×512 или 512×512×3) в полную маску.
    Использует original_width/height из мета, чтобы не включать padded области.
    """
    tiles_dir = Path(tiles_dir)
    mask_dir = tiles_dir / "masks"
    meta_dir = tiles_dir / "meta"

    mask_files = sorted(mask_dir.glob("mask_*.npy"))
    if not mask_files:
        raise FileNotFoundError("No NPY masks found in masks/")

    # Считаем общий размер по исходному изображению
    coords = []
    max_x = 0
    max_y = 0
    for f in mask_files:
        fname = f.name.replace("mask_", "tile_").replace(".npy", ".json")
        meta_path = meta_dir / fname
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        x_off = meta['x_off']
        y_off = meta['y_off']
        orig_w = meta['original_width']
        orig_h = meta['original_height']
        coords.append((f, x_off, y_off, orig_w, orig_h))
        max_x = max(max_x, x_off + orig_w)
        max_y = max(max_y, y_off + orig_h)

    # Определяем тип маски (2D или 3D)
    first_tile = np.load(mask_files[0])
    if len(first_tile.shape) == 2:
        full_mask = np.zeros((max_y, max_x), dtype=np.uint8)
    else:
        full_mask = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    for f, x, y, w, h in coords:
        tile = np.load(f)
        # Берём только оригинальную часть (без паддинга)
        if len(tile.shape) == 2:
            full_mask[y:y + h, x:x + w] = tile[:h, :w]
        else:
            full_mask[y:y + h, x:x + w, :] = tile[:h, :w, :]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, full_mask)
    return out_path


    # # Сохраняем результат
    # if full_mask.ndim == 2:  # Если это одноцветная маска
    #     np.save(out_path, full_mask)
    # else:  # Если это RGB изображение
    #     # Сохраняем как изображение
    #     result_img = Image.fromarray(full_mask)
    #     result_img.save(out_path.replace('.npy', '.png'))

    # return out_path



# def save_mask_as_geotiff(mask_npy_path, meta_dir, out_tif="mask.tif"):
#     """Сохраняет маску (uint8) как GeoTIFF"""
#     mask = np.load(mask_npy_path)
#     meta_dir = Path(meta_dir)
#     meta_files = sorted(meta_dir.glob("*.json"))
#     if not meta_files:
#         raise FileNotFoundError("No meta files found!")

#     with open(meta_files[0], "r", encoding="utf-8") as f:
#         meta0 = json.load(f)

#     transform = Affine.from_gdal(*meta0["transform"])
#     crs = meta0.get("crs")

#     profile = {
#         "driver": "GTiff",
#         "height": mask.shape[0],
#         "width": mask.shape[1],
#         "count": 1,
#         "dtype": mask.dtype,
#         "crs": crs,
#         "transform": transform
#     }

#     with rasterio.open(out_tif, "w", **profile) as dst:
#         dst.write(mask, 1)

#     return out_tif



def save_mask_as_geotiff(mask_npy_path, meta_dir, out_tif="mask.tif"):
    """
    Сохраняет маску (uint8, монохромную или RGB) как GeoTIFF.
    """
    mask = np.load(mask_npy_path)
    meta_dir = Path(meta_dir)
    meta_files = sorted(meta_dir.glob("*.json"))
    if not meta_files:
        raise FileNotFoundError("No meta files found!")

    with open(meta_files[0], "r", encoding="utf-8") as f:
        meta0 = json.load(f)

    transform = Affine.from_gdal(*meta0["transform"])
    crs = meta0.get("crs")

    # Определяем количество каналов
    if mask.ndim == 2:
        count = 1
        mask_to_write = mask[np.newaxis, :, :]  # (1, H, W)
    elif mask.ndim == 3:
        count = mask.shape[2]
        # rasterio ожидает порядок (bands, height, width)
        mask_to_write = np.transpose(mask, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": count,
        "dtype": mask.dtype,
        "crs": crs,
        "transform": transform
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(mask_to_write)

    return out_tif