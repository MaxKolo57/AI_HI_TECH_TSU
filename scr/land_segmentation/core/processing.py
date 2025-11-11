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


def save_mask_as_geotiff(mask_npy_path, meta_dir, out_tif="mask.tif"):
    """Сохраняет маску (uint8) как GeoTIFF"""
    mask = np.load(mask_npy_path)
    meta_dir = Path(meta_dir)
    meta_files = sorted(meta_dir.glob("*.json"))
    if not meta_files:
        raise FileNotFoundError("No meta files found!")

    with open(meta_files[0], "r", encoding="utf-8") as f:
        meta0 = json.load(f)

    transform = Affine.from_gdal(*meta0["transform"])
    crs = meta0.get("crs")

    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": mask.dtype,
        "crs": crs,
        "transform": transform
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(mask, 1)

    return out_tif
