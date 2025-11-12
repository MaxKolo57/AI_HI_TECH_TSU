from PIL import Image
import torch
from sync_transforms import SyncCompose, SyncRandomHorizontalFlip, SyncRotate360_plus, \
    SyncToTensor, SyncRandomVerticalFlip, TrickyResize_UpDwn, SyncResize, \
    RandomNoiseSP, RandomResizeCrop, RandomShift, RandomGridDistortion, RandomElasticTransform, RandomMirror
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm

class Data_task2(Dataset):
    def __init__(self, masks_paths, images_paths, transforms=[]):
        self.masks = list()
        self.imgs = list()

        for i in tqdm(range(len(masks_paths)), desc='Формирование изображений'):
            img = Image.open(images_paths / masks_paths[i].name.replace('.npy', '.png'))
            mask = np.load(masks_paths[i])
            background_channel = ~np.any(mask, axis=2)
            background_channel = background_channel[..., np.newaxis]
            mask = np.concatenate([background_channel, mask], axis=2)
            mask = (mask.argmax(2) * (255 / 9)).astype(np.uint8)
            mask = Image.fromarray(mask)
            self.imgs.append(img)
            self.masks.append(mask)

        self.names = list(range(len(self.imgs)))
        self.transforms = SyncCompose(transforms)
        self.ten_cls = np.array(range(10)) / 9

        self.ten_cls_diff = np.zeros(len(self.ten_cls) - 1)
        for i in range(len(self.ten_cls) - 1):
            self.ten_cls_diff[i] = self.ten_cls[i + 1] - self.ten_cls[i]

        self.aug = True

        print(f'Количество изображений = {len(self.imgs)}', f'Количество масок = {len(self.masks)}')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        try:
            if self.aug:
                pic, msk = self.transforms(img=self.imgs[item], mask=self.masks[item],
                                           names=self.names, img_path=self.imgs, mask_path=self.masks)
                # После трансформа значения классов для маски меняются
                # Востановление значений классов
                for i in range(len(self.ten_cls_diff)):
                    if i == 0:
                        msk[(msk <= self.ten_cls[i] + (self.ten_cls_diff[i] / 2)).bool()] = self.ten_cls[i]
                    elif i == len(self.ten_cls_diff) - 1:
                        msk[(np.logical_and(self.ten_cls[i] + (self.ten_cls_diff[i] / 2) >= msk,
                                            msk > self.ten_cls[i] - (self.ten_cls_diff[i - 1] / 2))).bool()] = self.ten_cls[
                            i]
                        msk[(msk > self.ten_cls[len(self.ten_cls_diff)] - (self.ten_cls_diff[i] / 2)).bool()] = \
                            self.ten_cls[
                                len(self.ten_cls_diff)]
                    else:
                        msk[(np.logical_and(self.ten_cls[i] + (self.ten_cls_diff[i] / 2) >= msk,
                                            msk > self.ten_cls[i] - (self.ten_cls_diff[i - 1] / 2))).bool()] = self.ten_cls[
                            i]


                cl_mask = torch.zeros(10, msk.shape[1], msk.shape[2])
                for ch, t in enumerate(self.ten_cls):
                    cl_mask[ch][msk[0] == t] = 1
                return pic, cl_mask # замена cl_mask
        except Exception as e:
            print(f"Ошибка в файле {self.imgs[item]}: {e}")
        return self.imgs[item], self.masks[item]



def prepare_datasets(resolution=512):
    sync_hor = SyncRandomHorizontalFlip()
    sync_v = SyncRandomVerticalFlip()
    syns_rp = SyncRotate360_plus(resolution=resolution)
    sync_rs = SyncResize(resolution)
    rs_up_dwn = TrickyResize_UpDwn(resolution=resolution, minmax_size_up=[117, 200])
    r_m = RandomMirror(p=0.7, resolution=resolution)
    rshift = RandomShift()
    rrc = RandomResizeCrop(erode=False)
    ret = RandomElasticTransform()
    rgd = RandomGridDistortion()
    sync_totensor = SyncToTensor()
    rn = RandomNoiseSP()

    main_transforms_list = []
    main_transforms_list.append(sync_hor)
    main_transforms_list.append(sync_v)
    main_transforms_list.append(rrc)
    main_transforms_list.append(r_m)
    main_transforms_list.append(rs_up_dwn)
    main_transforms_list.append(syns_rp)
    main_transforms_list.append(rgd)
    main_transforms_list.append(ret)
    main_transforms_list.append(rs_up_dwn)
    main_transforms_list.append(rshift)
    main_transforms_list.append(sync_rs)
    main_transforms_list.append(sync_totensor)
    main_transforms_list.append(rn)

    train_dataset_dir = Path('/media/user/SP PHD U3/TSU_dataset/train')
    val_dataset_dir = Path('/media/user/SP PHD U3/TSU_dataset/test')
    train_mask_dataset_dir = Path('/media/user/SP PHD U3/TSU_dataset/train')
    val_mask_dataset_dir = Path('/media/user/SP PHD U3/TSU_dataset/test')
    train_mask_glob = train_mask_dataset_dir.glob("*.npy")
    val_mask_glob = val_mask_dataset_dir.glob("*.npy")
    t_masks = [x for x in train_mask_glob]
    v_masks = [x for x in val_mask_glob]
    train_arr = np.array(t_masks)
    val_arr = np.array(v_masks)

    print('Для обучения')

    train_dataset = Data_task2(train_arr, train_dataset_dir, main_transforms_list)



    print('Для тестирования')
    val_dataset = Data_task2(val_arr, val_dataset_dir, [sync_rs, sync_totensor])

    print('Датасет сформирован')


    return train_dataset, val_dataset
