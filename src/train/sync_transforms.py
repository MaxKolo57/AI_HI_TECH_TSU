import cv2
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode as IMode
from random import choice, random
from skimage.util import random_noise
import albumentations as A




rgba_imodes = [IMode.NEAREST, IMode.BILINEAR]
imodes = [IMode.NEAREST, IMode.BILINEAR, IMode.BICUBIC, IMode.BOX, IMode.HAMMING, IMode.LANCZOS]


def blend(img, mask):
    aimg = np.squeeze(np.array(img), axis=0)
    amask = np.squeeze(np.array(mask), axis=0)
    aimg[amask > 0] = 1

    return aimg


class RandomMirror():
    def __init__(self, p=0.25, resolution=512):
        self.p = p
        self.resolution = resolution

    # Высота и ширина должны быть чётными
    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:

            if self.resolution:
                resolution_x, resolution_y = self.resolution, self.resolution
            else:
                resolution_x, resolution_y = img.size

            if img.size != (resolution_x,resolution_y) or mask.size != (resolution_x,resolution_y):
                img = F.resize(img, [resolution_x, resolution_y])
                mask = F.resize(mask, [resolution_x, resolution_y], interpolation=IMode.NEAREST)
            b_img = np.array(img)
            b_mask = np.array(mask)
            new_image = np.zeros_like(b_img).astype(np.uint8)
            new_mask = np.zeros_like(b_mask).astype(np.uint8)
            size_x = np.random.randint(((resolution_x * 50) / 100), ((resolution_x * 93) / 100))
            size_y = np.random.randint(((resolution_y * 50) / 100), ((resolution_y * 93) / 100))
            cut_x = np.random.randint(0, resolution_x - size_x)
            cut_y = np.random.randint(0, resolution_y - size_y)
            b_img = b_img[cut_y:cut_y + size_y, cut_x:cut_x + size_x, ...]
            b_mask = b_mask[cut_y:cut_y + size_y, cut_x:cut_x + size_x, ...]
            cut_x = np.random.randint(0, resolution_x - size_x)
            cut_y = np.random.randint(0, resolution_y - size_y)
            new_image[cut_y:cut_y + size_y, cut_x:cut_x + size_x, ...] = b_img
            new_mask[cut_y:cut_y + size_y, cut_x:cut_x + size_x, ...] = b_mask

            if new_image[cut_y:cut_y + size_y, (cut_x + size_x):, ...].shape != b_img[:, ::-1][:, :resolution_x - (cut_x + size_x)].shape:
                print(new_image.shape, b_img.shape, cut_x, cut_y, size_x, size_y)

            if new_mask[cut_y:cut_y + size_y, (cut_x + size_x):, ...].shape != b_mask[:, ::-1][:, :resolution_x - (cut_x + size_x)].shape:
                print(new_mask.shape, b_mask.shape, cut_x, cut_y, size_x, size_y)

            new_image[cut_y:cut_y + size_y, (cut_x + size_x):, ...] = b_img[:, ::-1][:, :resolution_x - (cut_x + size_x)]
            new_mask[cut_y:cut_y + size_y, (cut_x + size_x):, ...] = b_mask[:, ::-1][:, :resolution_x - (cut_x + size_x)]

            new_image[cut_y:cut_y + size_y, :cut_x, ...] = b_img[:, ::-1][:, size_x - cut_x:]
            new_mask[cut_y:cut_y + size_y, :cut_x, ...] = b_mask[:, ::-1][:, size_x - cut_x:]

            new_image[:cut_y, :, ...] = new_image[cut_y:2 * cut_y, :][::-1, :]
            new_mask[:cut_y, :, ...] = new_mask[cut_y:2 * cut_y, :][::-1, :]

            new_image[cut_y + size_y:, :, ...] = new_image[2 * (cut_y + size_y) - resolution_y:cut_y + size_y, :][::-1, :]
            new_mask[cut_y + size_y:, :, ...] = new_mask[2 * (cut_y + size_y) - resolution_y:cut_y + size_y, :][::-1, :]

            new_image = Image.fromarray(new_image)
            new_mask = Image.fromarray(new_mask)
            return new_image, new_mask
        else:
            return img, mask

class RandomResizeCrop():
    def __init__(self, p1=0.5, p2=0.5, erode=True):
        self.p1 = p1
        self.p2 = p2
        self.erode = erode


    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p1:
            return img, mask
        else:
            w, h = img.size
            if np.random.rand() < self.p2:
                scale = np.random.randint(80, 100) / 100
                scale2 = np.random.randint(70, 100) / 100
                crop_size = [int(w * scale), int(h * scale)]
                indx = np.random.randint(0, 2)
                crop_size[indx] = int(crop_size[indx] * scale2)
                point = [np.random.randint(0, w - crop_size[0] + 1), np.random.randint(0, h - crop_size[1]) + 1]
                new_image = F.crop(img, point[1], point[0], crop_size[1], crop_size[0])
                new_image = F.resize(new_image, [w, h], interpolation=IMode.BILINEAR)
                new_mask = F.crop(mask, point[1], point[0], crop_size[1], crop_size[0])
                new_mask = F.resize(new_mask, [w, h], interpolation=IMode.NEAREST)
                return new_image, new_mask
            else:
                if self.erode:
                    scale = np.random.randint(80, 100) / 100
                    new_image = F.resize(img, [int(w * scale), int(h * scale)], interpolation=IMode.BILINEAR)
                    new_mask = F.resize(mask, [int(w * scale), int(h * scale)], interpolation=IMode.NEAREST)
                    l = np.random.randint(0, int(w - w * scale))
                    t = np.random.randint(0, int(h - h * scale))
                    r = int(w - w * scale) - l
                    d = int(h - h * scale) - t
                    new_image = F.pad(new_image, padding=(l, t, r, d))
                    new_mask = F.pad(new_mask, padding=(l, t, r, d))
                    return new_image, new_mask
                else:
                    return img, mask


class RandomShift():
    def __init__(self, p1=0.5, p2=0.5):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p1:
            w, h = img.size
            left = np.random.randint(0, int(w * 0.3))
            top = np.random.randint(0, int(h * 0.3))
            right = np.random.randint(0, int(w * 0.3))
            down = np.random.randint(0, int(h * 0.3))
            new_image = F.pad(img, padding=(left, top, right, down))
            new_mask = F.pad(mask, padding=(left, top, right, down))
            if np.random.rand() < self.p2:
                new_image = F.crop(new_image, 0, 0, h, w)
                new_mask = F.crop(new_mask, 0, 0, h, w)
            else:
                new_image = F.crop(new_image, top + down, left + right, h, w)
                new_mask = F.crop(new_mask, top + down, left + right, h, w)
            return new_image, new_mask

        else:
            return img, mask



class SyncRandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            return F.hflip(img), F.hflip(mask)

        return img, mask


class SyncRandomVerticalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            return F.vflip(img), F.vflip(mask)

        return img, mask


class TrickyResize_UpDwn():
    def __init__(self, resolution=512, minmax_size_up=[117, 200]):
        self.resolution = resolution
        self.minmax_size_up = minmax_size_up


    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < 0.8:
            if self.resolution:
                size_down = np.array([self.resolution, self.resolution])
            else:
                size_down = img.size

            up_l = np.random.randint(self.minmax_size_up[0], self.minmax_size_up[1])
            rand_size_up = np.array([int((size_down[0] * up_l) / 100), int((size_down[1] * up_l) / 100)])
            interpolation = choice(imodes)
            img = F.resize(img, tuple(rand_size_up), interpolation=interpolation, antialias=True)
            interpolation = choice(imodes)
            img = F.resize(img, tuple(size_down), interpolation=interpolation, antialias=True)
            return img, mask

        return img, mask


class SyncRotate360_plus():
    def __init__(self, p=0.8, p_c=0.7, resolution=512):
        self.p = p
        self.p_c = p_c
        self.resolution = resolution

    @staticmethod
    def _crop(X1, Y1, resolution_x, resolution_y, img_fc, mask_fc):
        X2, Y2 = X1 + resolution_x, Y1 + resolution_y
        new_img = img_fc.crop([X1, Y1, X2, Y2])
        new_mask = mask_fc.crop([X1, Y1, X2, Y2])

        return new_img, new_mask

    def __call__(self, img, mask, **kwargs):
        rnd_p = np.random.rand()

        if self.resolution:
            resolution_x, resolution_y = self.resolution, self.resolution
        else:
            resolution_x, resolution_y = img.size

        if rnd_p < self.p:
            resample = choice(rgba_imodes)
            angle = (random() * 90) # поменял * 45 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if np.random.rand() < self.p_c:
                expand = True
            else:
                expand = False
            img_r = F.rotate(img, -angle, resample, expand, None, list(np.zeros(len(img.getbands()))))
            mask_r = F.rotate(mask, -angle, resample, expand, None, list(np.zeros(len(mask.getbands()))))

            if expand:
                w, h = img_r.size
                if w < ((w * 127) / 100) and np.random.rand() < 0.5:
                    if np.random.rand() < 0.7:
                        lenimg = np.random.randint(88, 142)
                        rand_size_up = np.array([int((w * lenimg) / 100), int((h * lenimg) / 100)])
                        interpolation = choice(imodes)
                        img_r = F.resize(img_r, tuple(rand_size_up), interpolation=interpolation, antialias=True)
                        mask_r = F.resize(mask_r, tuple(rand_size_up), interpolation=interpolation, antialias=True)
                        w, h = img_r.size

                    delt_sz1 = (((w * 142) / 100) - w) // 2
                    delt_sz2 = (((h * 142) / 100) - h) // 2
                    pad_img = T.Pad(padding=(int(delt_sz1), int(delt_sz2)), fill=0)
                    img_r = pad_img(img_r)
                    mask_r = pad_img(mask_r)
                    w, h = img_r.size
                X1, Y1 = np.random.randint(0, w - resolution_x), np.random.randint(0, h - resolution_y)
                img_r, mask_r = self._crop(X1, Y1, resolution_x, resolution_y, img_r, mask_r)

            return img_r, mask_r

        elif rnd_p < 0.7: # корректно работает только для изображений с равными сторонами
            w, h = img.size
            if np.random.rand() < 0.7:
                lenimg = np.random.randint(88, 142)
                rand_size_up = np.array([int((w * lenimg) / 100), int((h * lenimg) / 100)])
                interpolation = choice(imodes)
                img = F.resize(img, tuple(rand_size_up), interpolation=interpolation, antialias=True)
                mask = F.resize(mask, tuple(rand_size_up), interpolation=interpolation, antialias=True)

            delt_sz1 = (((w * 142) / 100) - w) // 2
            delt_sz2 = (((h * 142) / 100) - h) // 2
            pad_img = T.Pad(padding=(int(delt_sz1), int(delt_sz2)), fill=0)
            img_r = pad_img(img)
            mask_r = pad_img(mask)
            X1, Y1 = np.random.randint(0, w - resolution_x), np.random.randint(0, h - resolution_y)
            img_r, mask_r = self._crop(X1, Y1, resolution_x, resolution_y, img_r, mask_r)
            return img_r, mask_r
        return img, mask


class SyncResize:
    def __init__(self, resolution=512):
        self.resolution = resolution

    def __call__(self, img, mask, **kwargs):
        if self.resolution:
            imsize = np.array([self.resolution, self.resolution])
            return F.resize(img, imsize), F.resize(mask, imsize, interpolation=IMode.NEAREST)
        else:
            return img, F.resize(mask, img.size, interpolation=IMode.NEAREST)



class RandomNoiseSP:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            amount = np.random.randint(0, 11) / 1000
            salt_img = torch.tensor(random_noise(img, mode='salt', amount=amount))
            return salt_img, mask
        elif np.random.rand() < self.p:
            amount = np.random.randint(0, 11) / 1000
            salt_img = torch.tensor(random_noise(img, mode='pepper', amount=amount))
            return salt_img, mask
        elif np.random.rand() < self.p:
            amount = np.random.randint(0, 11) / 1000
            salt_img = torch.tensor(random_noise(img, mode='s&p', amount=amount))
            return salt_img, mask
        return img, mask


class RandomGridDistortion:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            image = np.array(img)
            mask = np.array(mask)
            aug = A.GridDistortion(num_steps=np.random.randint(2,6), distort_limit=(-0.3, 0.3), normalized=(np.random.rand() < 0.5), p=1,
                                   border_mode=cv2.BORDER_CONSTANT,  # Заполнять константным значением
                                   )
            augmented = aug(image=image, mask=mask)
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            return image, mask
        else:
            return img, mask


class RandomElasticTransform:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            image = np.array(img)
            mask = np.array(mask)
            aug = A.ElasticTransform(p=1, alpha=np.random.randint(5,16), sigma=50, approximate=False,
                                     border_mode=cv2.BORDER_CONSTANT,  # Заполнять константным значением
                                     )
            augmented = aug(image=image, mask=mask)
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            return image, mask
        else:
            return img, mask


class SyncCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, **kwargs):
        for transform in self.transforms:
            img, mask = transform(img, mask, **kwargs)
        return img, mask


class SyncToTensor:
    def __init__(self):
        self.tt = ToTensor()

    def __call__(self, img, mask, **kwargs):
        return self.tt(img), self.tt(mask)
