# core/segmentation.py
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
import torch

from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw, ImageFont
import segmentation_models_pytorch as smp
import gc


from .processing import tile_raster_to_pngs, reconstruct_mask_from_tiles, save_mask_as_geotiff

# -----------------------------
# Определяем 10 классов
# -----------------------------
CLASS_MAP = {
    0: "Фон",
    1: "Водные объекты",
    2: "Дороги",
    3: "Залежи (Деревья)",
    4: "Залежи (Кустарник)",
    5: "Залежи (Травы)",
    6: "Земли под коммуникациями",
    7: "Леса и древесно-кустарниковая растительность",
    8: "Пахотные земли",
    9: "Неиспользуемые участки"
}

# Цветовая палитра для превью (RGB)
CLASS_COLORS = [
    (0, 0, 0),  # фон
    (128, 128, 128),  # дорога
    (0, 0, 255),      # вода
    (0, 100, 0),      # деревья
    (34, 139, 34),    # кустарник
    (124, 252, 0),    # травы
    (255, 215, 0),    # коммуникации
    (0, 128, 0),      # леса
    (210, 105, 30),   # пахотные земли
    (169, 169, 169)   # неиспользуемые
]

# -----------------------------
# Функция сегментации одного тайла
# -----------------------------
def multi_class_tile_segmentation(tile_png_path, mask_out_path):
    """
    Сегментация одного тайла на 10 классов.
    На основе простых цветовых фильтров (пример, можно заменить CNN).
    """
    img = np.array(Image.open(tile_png_path).convert("RGB"))
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # 1. Дорога: серые тона
    mask[(np.abs(r - g) < 10) & (np.abs(r - b) < 10) & (r > 100)] = 1

    # 2. Вода: синие
    mask[(b > r) & (b > g) & (b > 80)] = 2

    # 3. Деревья: темно-зеленые
    mask[(g > r) & (g > b) & (g > 100)] = 3

    # 4. Кустарник: светло-зеленые
    mask[(g > r) & (g > b) & (g > 150)] = 4

    # 5. Травы: желтовато-зеленые
    mask[(r > 100) & (g > 100) & (b < 80)] = 5

    # 6. Земли под коммуникациями — заглушка (можно добавить по прямым линиям)
    # mask[...] = 6

    # 7. Леса естественные: темные участки зеленого
    mask[(g > 80) & (g < 120) & (r < 80)] = 7

    # 8. Пахотные земли: коричневые
    mask[(r > 120) & (g > 100) & (b < 80)] = 8

    # 9. Неиспользуемые участки: светло-серые или пустые
    # mask[...] = 9

    np.save(mask_out_path, mask)
    return mask_out_path


def multi_class_tile_segmentation_rgb(tile_png_path, mask_out_path):
    """
    Сегментация одного тайла на 10 классов.
    Возвращает изображение с RGB цветами для каждого класса.
    """

    img = np.array(Image.open(tile_png_path).convert("RGB"))
    rgb_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # 1. Дорога: серые тона
    rgb_mask[(np.abs(r - g) < 10) & (np.abs(r - b) < 10) & (r > 100)] = CLASS_COLORS[1]

    # 2. Вода: синие
    rgb_mask[(b > r) & (b > g) & (b > 80)] = CLASS_COLORS[2]

    # 3. Деревья: темно-зеленые
    rgb_mask[(g > r) & (g > b) & (g > 100)] = CLASS_COLORS[3]

    # 4. Кустарник: светло-зеленые
    rgb_mask[(g > r) & (g > b) & (g > 150)] = CLASS_COLORS[4]

    # 5. Травы: желтовато-зеленые
    rgb_mask[(r > 100) & (g > 100) & (b < 80)] = CLASS_COLORS[5]

    # 6. Земли под коммуникациями — заглушка (можно добавить по прямым линиям)
    # rgb_mask[...] = CLASS_COLORS[6]

    # 7. Леса естественные: темные участки зеленого
    rgb_mask[(g > 80) & (g < 120) & (r < 80)] = CLASS_COLORS[7]

    # 8. Пахотные земли: коричневые
    rgb_mask[(r > 120) & (g > 100) & (b < 80)] = CLASS_COLORS[8]

    # 9. Неиспользуемые участки: светло-серые или пустые
    # rgb_mask[...] = CLASS_COLORS[9]

    # Сохраняем результат как PNG
    # result_img = Image.fromarray(rgb_mask)
    # result_img.save(mask_out_path)
    print(rgb_mask.shape)
    np.save(mask_out_path, rgb_mask)

    return mask_out_path






# -----------------------------
# Преобразование маски в RGB для превью
# -----------------------------
def mask_to_rgb_preview(mask):
    """Преобразует маску с классами 0..9 в RGB"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_id] = color
    return rgb

# -----------------------------
# Worker для сегментации ортофотоплана
# -----------------------------
class SegmentationWorker(QThread):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)

    FILE_PATH = Path(__file__) 

    def __init__(self, tif_path, work_dir, tile_size=512):
        super().__init__()


        self.tif_path = Path(tif_path)
        self.work_dir = Path(work_dir)
        self.tile_size = tile_size
        self.pth_file = self.FILE_PATH.parent.parent / 'models'/ '00100.pt'


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        checkpoint = torch.load(self.pth_file, map_location=self.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
        # Ищем параметры классификатора
            classifier_weight_key = None
            for key in checkpoint['model_state_dict'].keys():
                if 'classifier_3' in key and 'weight' in key:
                    classifier_weight_key = key
                    break

            if classifier_weight_key:
                num_classes = checkpoint['model_state_dict'][classifier_weight_key].shape[0]
                print(f"Автоматически определено количество классов из чекпоинта: {num_classes}")
            else:
                print("Не удалось определить количество классов из чекпоинта, используем значение по умолчанию 6")
                num_classes = 10
        else:
            print("Чекпоинт не содержит 'model_state_dict', используем значение по умолчанию 6")
            num_classes = 10


        self.model = smp.Segformer(
            # encoder_name="resnet18",  # State-of-the-art Transformer энкодер
            
            encoder_name="resnext101_32x16d",  # State-of-the-art Transformer энкодер
            encoder_weights=None,
            classes=10,
            in_channels=3,
            decoder_segmentation_channels=128,
            decoder_attn_channels=128,  # Attention для точных границ
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Модель успешно загружена")

        del checkpoint
        gc.collect()

    def model_segmentation(self,tile_png_path, mask_out_path):
        """
        Сегментация одного тайла на 10 классов.
        Возвращает изображение с RGB цветами для каждого класса.
        """

        tile = np.array(Image.open(tile_png_path).convert("RGB"))
        # tile = np.transpose(tile, (1, 2, 0))


        rgb_mask = np.zeros((tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
        
        r, g, b = tile[:, :, 0], tile[:, :, 1], tile[:, :, 2]

        print(tile.dtype)

        tile_tensor = to_tensor(tile).unsqueeze(0).to(self.device)

        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        def normalize_tensor(tensor):
            """Нормализация тензора на GPU"""
            # tensor = tensor.to('cuda')
            return (tensor - IMAGENET_MEAN) / IMAGENET_STD
        
        tile_tensor = normalize_tensor(tile_tensor)
        print(tile_tensor.shape)
        output = self.model(tile_tensor)
        output = output.cpu().detach().numpy()[0]
        mask_2d = np.argmax(output, axis=0)  # shape: (512, 512) 

        # # 2. Добавить ось в конце → (512, 512, 1)
        # mask_3d = np.expand_dims(mask_2d, axis=-1)

            # Преобразуем в RGB по палитре
        h, w = mask_2d.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for cls_id, color in enumerate(CLASS_COLORS):
            rgb_mask[mask_2d == cls_id] = color

        # Сохраняем как (512, 512, 3) массив
        np.save(mask_out_path, rgb_mask)
        return mask_out_path

    def run(self):
        try:
            self.message.emit("Нарезка ортофотоплана на тайлы...")
            tiles_dir = self.work_dir

            # очистка временной директории
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

            # Сегментация тайлов
            for i, imgf in enumerate(image_files, start=1):
                mask_out = masks_dir / (imgf.name.replace("tile_", "mask_").replace(".png", ".npy"))
                # multi_class_tile_segmentation(str(imgf), str(mask_out))
                # multi_class_tile_segmentation_rgb(str(imgf), str(mask_out))
                self.model_segmentation(str(imgf), str(mask_out))
                pct = 10 + int(80 * (i / total))
                self.progress.emit(pct)
                self.message.emit(f"Сегментация тайла {i}/{total}")

            # Сборка маски
            self.message.emit("Сборка полной маски...")
            mask_npy = str(tiles_dir / "reconstructed_mask.npy")
            reconstruct_mask_from_tiles(tiles_dir, mask_npy)
            self.progress.emit(95)

            # Сохранение GeoTIFF
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
