from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):
        left = np.transpose(sample['left'], (2, 0, 1))  # [3, H, W]
        sample['left'] = torch.from_numpy(left) / 255.
        right = np.transpose(sample['right'], (2, 0, 1))
        sample['right'] = torch.from_numpy(right) / 255.

        # disp = np.expand_dims(sample['disp'], axis=0)  # [1, H, W]
        if 'disp' in sample.keys():
            disp = sample['disp']  # [H, W]
            sample['disp'] = torch.from_numpy(disp)

        if 'pseudo_disp' in sample.keys():
            disp = sample['pseudo_disp']  # [H, W]
            sample['pseudo_disp'] = torch.from_numpy(disp)

        return sample


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['left', 'right']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample


class RandomCrop(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        ori_height, ori_width = sample['left'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            sample['left'] = np.lib.pad(sample['left'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            sample['right'] = np.lib.pad(sample['right'],
                                         ((top_pad, 0), (0, right_pad), (0, 0)),
                                         mode='constant',
                                         constant_values=0)
            if 'disp' in sample.keys():
                sample['disp'] = np.lib.pad(sample['disp'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)

            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = np.lib.pad(sample['pseudo_disp'],
                                                   ((top_pad, 0), (0, right_pad)),
                                                   mode='constant',
                                                   constant_values=0)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            sample['left'] = self.crop_img(sample['left'])
            sample['right'] = self.crop_img(sample['right'])
            if 'disp' in sample.keys():
                sample['disp'] = self.crop_img(sample['disp'])
            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = self.crop_img(sample['pseudo_disp'])

        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]


class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            sample['left'] = np.copy(np.flipud(sample['left']))
            sample['right'] = np.copy(np.flipud(sample['right']))

            sample['disp'] = np.copy(np.flipud(sample['disp']))

            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = np.copy(np.flipud(sample['pseudo_disp']))

        return sample


class ToPILImage(object):

    def __call__(self, sample):
        sample['left'] = Image.fromarray(sample['left'].astype('uint8'))
        sample['right'] = Image.fromarray(sample['right'].astype('uint8'))

        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['left'] = np.array(sample['left']).astype(np.float32)
        sample['right'] = np.array(sample['right']).astype(np.float32)

        return sample


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)

            sample['left'] = F.adjust_contrast(sample['left'], contrast_factor)
            sample['right'] = F.adjust_contrast(sample['right'], contrast_factor)

        return sample


class RandomGamma(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['left'] = F.adjust_gamma(sample['left'], gamma)
            sample['right'] = F.adjust_gamma(sample['right'], gamma)

        return sample


class RandomBrightness(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)

            sample['left'] = F.adjust_brightness(sample['left'], brightness)
            sample['right'] = F.adjust_brightness(sample['right'], brightness)

        return sample


class RandomHue(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample['left'] = F.adjust_hue(sample['left'], hue)
            sample['right'] = F.adjust_hue(sample['right'], hue)

        return sample


class RandomSaturation(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)

            sample['left'] = F.adjust_saturation(sample['left'], saturation)
            sample['right'] = F.adjust_saturation(sample['right'], saturation)

        return sample


class RandomColor(object):

    def __call__(self, sample):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample
