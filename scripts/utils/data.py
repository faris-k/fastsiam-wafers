from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from lightly.data.collate import BaseCollateFunction, MultiViewCollateFunction
from lightly.transforms.rotation import RandomRotate
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode


# TODO: try making this work on PIL images instead of arrays
class DieNoise:
    """Adds noise to wafermap die by flipping pass to fail and vice-versa with probability p.

    Parameters
    ----------
    p : float, optional
        Probability of adding noise on a die-level basis, by default 0.03
    """

    def __init__(self, p: float = 0.03):
        self.p = p

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        def flip(item):
            """
            Given a wafermap die, flips pass to fail and vice-versa with probability p.
            Does nothing to non-die area (0's if 128's and 255's are passes/fails respectively).
            """
            prob = np.random.choice([False, True], p=[1 - self.p, self.p])
            if prob:
                if item == 128:
                    return 255
                elif item == 255:
                    return 128
                else:
                    return item
            return item

        vflip = np.vectorize(flip)
        out = vflip(sample)
        return torch.from_numpy(out)


def base_transforms(
    img_size: List[int] = [200, 200],
    die_noise_prob: float = 0.03,
    rr_prob: float = 0.5,
    hf_prob: float = 0.5,
    vf_prob: float = 0.5,
    rr_prob2: float = 0.25,
    to_tensor: bool = True,
) -> T.Compose:
    """Base image transforms for self-supervised training.
    Applies randomized die noise, converts to PIL Image, resizes, rotates, flips, and optionally converts to tensor.

    Parameters
    ----------
    img_size : List[int], optional
        Size of image, by default [200, 200]
    die_noise_prob : float, optional
        Probability of adding die noise, by default 0.03
    rr_prob : float, optional
        Probability of rotating image 90 degrees, by default 0.5
    hf_prob : float, optional
        Probability of flipping image horizontally, by default 0.5
    vf_prob : float, optional
        Probability of flipping image vertically, by default 0.5
    rr_prob2 : float, optional
        Probability of randomly rotating image between 0 and 90 degrees, by default 0.25
    to_tensor : bool, optional
        Whether to convert to tensor, by default True.
        Use False if you need further augmentations like global/patch cropping.
    """

    transforms = [
        # Add die noise before anything else
        DieNoise(die_noise_prob),
        # Convert to PIL Image, then perform all torchvision transforms except cropping
        T.ToPILImage(),
        T.Resize(img_size, interpolation=InterpolationMode.NEAREST),
        RandomRotate(rr_prob),
        T.RandomVerticalFlip(vf_prob),
        T.RandomHorizontalFlip(hf_prob),
        T.RandomApply(
            torch.nn.ModuleList(
                [T.RandomRotation(90, interpolation=InterpolationMode.NEAREST)]
            ),
            rr_prob2,
        ),
        # Finally, create a 3-channel image and convert to tensor
        T.Grayscale(num_output_channels=3),  # R == G == B
    ]

    # Optionally convert to tensor
    if to_tensor:
        transforms.append(T.ToTensor())

    return T.Compose(transforms)


class WaferImageCollateFunction(BaseCollateFunction):
    """Implements augmentations for self-supervised training on wafermaps.
    Works for "generic" joint-embedding methods like SimCLR, MoCo-v2, BYOL, SimSiam, etc.

    Parameters
    ----------
    img_size : List[int], optional
        Size of augmented images, by default [200, 200]
    die_noise_prob : float, optional
        Probability of applying die noise on a per-die basis, by default 0.03
    hf_prob : float, optional
        Probability of horizontally flipping the image, by default 0.5
    vf_prob : float, optional
        Probability of vertically flipping the image, by default 0.5
    rr_prob : float, optional
        Probability of rotating the image by 90 degrees, by default 0.5
    rr_prob2 : float, optional
        Probability of randomly rotating image between 0 and 90 degrees, by default 0.25
    """

    def __init__(
        self,
        img_size: List[int] = [200, 200],
        die_noise_prob: float = 0.03,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        rr_prob2: float = 0.25,
    ):

        transforms = base_transforms(
            img_size=img_size,
            die_noise_prob=die_noise_prob,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_prob2=rr_prob2,
            to_tensor=True,
        )
        super().__init__(transforms)


class WaferFastSiamCollateFunction(MultiViewCollateFunction):
    """Implements augmentations for FastSiam training on wafermaps.

    Parameters
    ----------
    img_size : List[int], optional
        Size of augmented images, by default [200, 200]
    die_noise_prob : float, optional
        Probability of applying die noise on a per-die basis, by default 0.03
    hf_prob : float, optional
        Probability of horizontally flipping the image, by default 0.5
    vf_prob : float, optional
        Probability of vertically flipping the image, by default 0.5
    rr_prob : float, optional
        Probability of rotating the image by 90 degrees, by default 0.5
    rr_prob2 : float, optional
        Probability of randomly rotating image between 0 and 90 degrees, by default 0.25
    """

    def __init__(
        self,
        img_size: List[int] = [200, 200],
        die_noise_prob: float = 0.03,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        rr_prob2: float = 0.25,
    ):

        transforms = base_transforms(
            img_size=img_size,
            die_noise_prob=die_noise_prob,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_prob2=rr_prob2,
            to_tensor=True,
        )
        super().__init__([transforms] * 4)


class WaferDINOCOllateFunction(MultiViewCollateFunction):
    """Custom collate function for DINO training on wafermaps."""

    def __init__(
        self,
        global_crop_size=224,
        global_crop_scale=(0.6, 1.0),
        local_crop_size=96,
        local_crop_scale=(0.1, 0.4),
        n_local_views=6,
        hf_prob=0.5,
        vf_prob=0.5,
        rr_prob=0.5,
    ):

        base_transforms = T.Compose(
            [
                # Add die noise before anything else
                DieNoise(),
                # Convert to PIL Image, then perform all torchvision transforms
                T.ToPILImage(),
                T.Resize(
                    [global_crop_size, global_crop_size],
                    interpolation=InterpolationMode.NEAREST,
                ),
                RandomRotate(rr_prob),
                T.RandomVerticalFlip(vf_prob),
                T.RandomHorizontalFlip(hf_prob),
                T.RandomApply(
                    torch.nn.ModuleList(
                        [T.RandomRotation(90, interpolation=InterpolationMode.NEAREST)]
                    ),
                    0.25,
                ),
                # Finally, create a 3-channel image and convert to tensor
                T.Grayscale(num_output_channels=3),  # R == G == B
                # T.ToTensor(),
            ]
        )

        global_crop = T.RandomResizedCrop(
            global_crop_size,
            scale=global_crop_scale,
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.NEAREST,
        )

        # first global crop
        global_transform_0 = T.Compose(
            [
                base_transforms,
                global_crop,
                T.ToTensor(),
            ]
        )

        # second global crop
        global_transform_1 = T.Compose(
            [
                base_transforms,
                global_crop,
                T.ToTensor(),
            ]
        )

        # transformation for the local small crops
        local_transform = T.Compose(
            [
                base_transforms,
                T.RandomResizedCrop(
                    local_crop_size,
                    scale=local_crop_scale,
                    ratio=(1.0, 1.0),
                    interpolation=InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
            ]
        )
        local_transforms = [local_transform] * n_local_views

        transforms = [global_transform_0, global_transform_1]
        transforms.extend(local_transforms)
        super().__init__(transforms)


class WaferMSNCollateFunction(MultiViewCollateFunction):
    """
    Implements MSN transformations for wafermaps.
    Modified from https://github.com/lightly-ai/lightly/blob/master/lightly/data/collate.py#L855

    Parameters
    ----------
    random_size : int, optional
        Size of the global/random image views, by default 224
    focal_size : int, optional
        Size of the focal image views, by default 96
    random_views : int, optional
        Number of global/random views to generate, by default 2
    focal_views : int, optional
        Number of focal views to generate, by default 10
    random_crop_scale : Tuple[float, float], optional
        Minimum and maximum size of the randomized crops relative to random_size,
        by default (0.3, 1.0)
    focal_crop_scale : Tuple[float, float], optional
        Minimum and maximum size of the focal crops relative to focal_size,
        by default (0.05, 0.3)
    die_noise_prob : float, optional
        Probability of adding randomized die noise at a die-level, by default 0.03
    rr_prob : float, optional
        Probability of rotating the image by 90 degrees, by default 0.5
    hf_prob : float, optional
        Probability that horizontal flip is applied, by default 0.5
    vf_prob : float, optional
        Probability that horizontal flip is applied, by default 0.0
    rr_prob2 : float, optional
        Probability of randomly rotating image between 0 and 90 degrees, by default 0.25
    """

    def __init__(
        self,
        random_size: int = 224,
        focal_size: int = 96,
        random_views: int = 2,
        focal_views: int = 10,
        random_crop_scale: Tuple[float, float] = (0.3, 1.0),
        focal_crop_scale: Tuple[float, float] = (0.05, 0.3),
        die_noise_prob: float = 0.03,
        rr_prob: float = 0.5,
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        rr_prob2: float = 0.25,
    ) -> None:

        base_transforms = [
            # Add die noise before anything else
            DieNoise(die_noise_prob),
            # Convert to PIL Image, then perform all torchvision transforms except cropping
            T.ToPILImage(),
            T.Resize(
                [random_size, random_size], interpolation=InterpolationMode.NEAREST
            ),
            RandomRotate(rr_prob),
            T.RandomVerticalFlip(vf_prob),
            T.RandomHorizontalFlip(hf_prob),
            T.RandomApply(
                torch.nn.ModuleList(
                    [T.RandomRotation(90, interpolation=InterpolationMode.NEAREST)]
                ),
                rr_prob2,
            ),
            # Finally, create a 3-channel image and convert to tensor
            T.Grayscale(num_output_channels=3),  # R == G == B
            # T.ToTensor(),  # Don't convert to tensor yet since we need to crop first
        ]

        # Create separate transforms for random and focal views
        random_crop = [
            T.RandomResizedCrop(
                size=random_size,
                scale=random_crop_scale,
                ratio=(1.0, 1.0),
                interpolation=InterpolationMode.NEAREST,
            ),
            T.ToTensor(),
        ]
        focal_crop = [
            T.RandomResizedCrop(
                size=focal_size,
                scale=focal_crop_scale,
                ratio=(1.0, 1.0),
                interpolation=InterpolationMode.NEAREST,
            ),
            T.ToTensor(),
        ]

        # Combine base transforms with random and focal crops
        transform = T.Compose(base_transforms + random_crop)
        focal_transform = T.Compose(base_transforms + focal_crop)

        # Put all transforms together
        transforms = [transform] * random_views
        transforms += [focal_transform] * focal_views
        super().__init__(transforms=transforms)


class WaferMapDataset(Dataset):
    """Dataset for wafermaps.

    Parameters
    ----------
    X : pd.Series
        Series of wafermaps
    y : pd.Series
        Series of labels
    transform : torchvision.transforms, optional
        Transformations to apply to each wafermap, by default None
    """

    def __init__(self, X, y, transform=None):
        self.data = pd.concat([X, y], axis="columns")
        # All resizing is done in augmentations, so we have tensors/arraays of different sizes
        # Because of this, just create a list of tensors
        self.X_list = [torch.tensor(ndarray) for ndarray in X]
        self.y_list = [torch.tensor(ndarray) for ndarray in y]
        self.transform = transform

    def __getitem__(self, index):
        x = self.X_list[index]
        y = self.y_list[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X_list)


def inference_transforms(img_size: List[int] = [224, 224]):
    """Image transforms for inference.
    Simply converts to PIL Image, resizes, and converts to tensor.

    Parameters
    ----------
    img_size : List[int], optional
        Size of image, by default [224, 224]
    """
    return T.Compose(
        [
            # Convert to PIL Image, then perform all torchvision transforms
            T.ToPILImage(),
            T.Resize(img_size, interpolation=InterpolationMode.NEAREST),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
        ]
    )
