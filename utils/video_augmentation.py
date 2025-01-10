# ----------------------------------------
# Written by Yuecong Min
# ----------------------------------------

import pdb
import PIL
import copy
import scipy.misc
import torch
import random
import numbers
import numpy as np
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class ToTensor(object):
    def __call__(self, video):
        if isinstance(video, list):
            video = np.array(video)
            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose((0, 3, 1, 2)))
        return video


class RandomCrop(object):
    """
    Extract random crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, clip):
        crop_h, crop_w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if crop_w > im_w:
            pad = crop_w - im_w
            clip = [np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            w1 = 0
        else:
            w1 = random.randint(0, im_w - crop_w)

        if crop_h > im_h:
            pad = crop_h - im_h
            clip = [np.pad(img, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            h1 = 0
        else:
            h1 = random.randint(0, im_h - crop_h)

        if isinstance(clip[0], np.ndarray):
            return [img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip]


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        try:
            im_h, im_w, im_c = clip[0].shape
        except ValueError:
            print(clip[0].shape)
        new_h, new_w = self.size
        new_h = im_h if new_h >= im_h else new_h
        new_w = im_w if new_w >= im_w else new_w
        top = int(round((im_h - new_h) / 2.))
        left = int(round((im_w - new_w) / 2.))
        return [img[top:top + new_h, left:left + new_w] for img in clip]


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, clip):
        # B, H, W, 3
        flag = random.random() < self.prob
        if flag:
            clip = np.flip(clip, axis=2)
            clip = np.ascontiguousarray(copy.deepcopy(clip))
        return np.array(clip)


class RandomRotation(object):
    """
    Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')
        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [scipy.ndimage.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return rotated


class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2):
        self.min_len = 32
        self.max_len = 230
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling
        # self.U = 1.0 + 0.2

    def __call__(self, clip):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return [clip[i] for i in index] if isinstance(clip, list) else clip[index]


class RandomResize(object):
    """
    Resize video bysoomingin and out.
    Args:
        rate (float): Video is scaled uniformly between
        [1 - rate, 1 + rate].
        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp='bilinear'):
        self.rate = rate
        self.interpolation = interp

    def __call__(self, clip):
        scaling_factor = random.uniform(1 - self.rate, 1 + self.rate)

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_h, new_w)
        if isinstance(clip[0], np.ndarray):
            return [np.array(Image.fromarray(img).resize(new_size, resample=self._get_PIL_interp(self.interpolation))) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=new_size, resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        if interp == 'nearest':
            return PIL.Image.NEAREST
        elif interp == 'lanczos':
            return PIL.Image.LANCZOS
        elif interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'cubic':
            return PIL.Image.CUBIC


class Resize(object):
    """
    Resize video bysoomingin and out.
    Args:
        rate (float): Video is scaled uniformly between
        [1 - rate, 1 + rate].
        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp='bilinear'):
        self.rate = rate
        self.interpolation = interp

    def __call__(self, clip):
        scaling_factor = self.rate

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        if isinstance(clip[0], np.ndarray):
            return [np.array(PIL.Image.fromarray(img).resize(new_size)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        if interp == 'nearest':
            return PIL.Image.NEAREST
        elif interp == 'lanczos':
            return PIL.Image.LANCZOS
        elif interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'cubic':
            return PIL.Image.CUBIC


if __name__ == "__main__":
    example_video = "../dataset/phoenix-2014-multisigner-T/features/fullFrame-256x256px/dev/01April_2010_Thursday_heute-6698"
    import glob
    img_list = sorted(glob.glob(f"{example_video}/*.png"))
    video = [cv2.imread(img_path) for img_path in img_list]
    augment =  Compose(
                [
                    # RandomCrop(128),
                    # RandomRotation(30),
                    # RandomHorizontalFlip(0.5),
                    # ToTensor(),
                    TemporalRescale(0.5),
                    # RandomResize(0.5)
                ]
            )
    print(len(video))
    video = augment(video)
    print(len(video))
    # # [0.0, 1.0)
    # random.random()
    # # [0.0, 1.0] size
    # np.random.random()
    # # [a, b]
    # random.randint()
    # # [low, high] -> a + (b-a)*random.random() 
    # random.uniform(low=0.0, high=1.0,size=None)
    # random.sample()
    # random.choices()
    # pdb.set_trace()
    cv2.imwrite("test.jpg", video[0])