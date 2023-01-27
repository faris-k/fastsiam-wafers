from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from lightly.data.collate import BaseCollateFunction, MultiViewCollateFunction
from lightly.transforms.rotation import RandomRotate
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode


# NEW VERSION 🚀
class DieNoise(object):
    """Adds noise to wafermap die by flipping pass to fail and vice-versa with probability p.

    Parameters
    ----------
    p : float, optional
        Probability of flipping on a die-level basis, by default 0.03
    """

    def __init__(self, p=0.03) -> None:
        self.p = p

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # Create a boolean mask of the 128's and 255's in the matrix
        mask = (sample == 128) | (sample == 255)
        # Create a tensor of random numbers between 0 and 1 with the same shape as the matrix
        rand = torch.rand(*sample.shape)
        # Use the mask and the random numbers to determine which elements to flip
        flip = ((rand < self.p) & mask).type(torch.bool)
        # Flip the elements
        sample[flip] = 383 - sample[flip]
        return sample


def get_base_transforms(
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

        transforms = get_base_transforms(
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

        transforms = get_base_transforms(
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
        global_crop_size: int = 224,
        global_crop_scale: Tuple[float, float] = (0.6, 1.0),
        local_crop_size: int = 96,
        local_crop_scale: Tuple[float, float] = (0.1, 0.4),
        n_local_views: int = 6,
        die_noise_prob: float = 0.03,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        rr_prob2: float = 0.25,
    ):
        """Implements augmentations for DINO training on wafermaps.

        Parameters
        ----------
        global_crop_size : int, optional
            Size of global crop, by default 224
        global_crop_scale : tuple, optional
            Minimum and maximum size of the global crops relative to global_crop_size,
            by default (0.6, 1.0)
        local_crop_size : int, optional
            Size of local crop, by default 96
        local_crop_scale : tuple, optional
            Minimum and maximum size of the local crops relative to global_crop_size,
            by default (0.1, 0.4)
        n_local_views : int, optional
            Number of generated local views, by default 6
        die_noise_prob : float, optional
            Probability of applying die noise on a per-die basis, by default 0.03
        hf_prob : float, optional
            Probability of horizontally flipping, by default 0.5
        vf_prob : float, optional
            Probability of vertically flipping, by default 0.5
        rr_prob : float, optional
            Probability of rotating by 90 degrees, by default 0.5
        rr_prob2 : float, optional
            Probability of randomly rotating image between 0 and 90 degrees, by default 0.25
        """

        base_transform = get_base_transforms(
            img_size=[global_crop_size, global_crop_size],
            die_noise_prob=die_noise_prob,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_prob2=rr_prob2,
            to_tensor=False,
        )

        global_crop = T.RandomResizedCrop(
            global_crop_size,
            scale=global_crop_scale,
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.NEAREST,
        )

        local_crop = T.RandomResizedCrop(
            local_crop_size,
            scale=local_crop_scale,
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.NEAREST,
        )

        global_transform = T.Compose(
            [
                base_transform,
                global_crop,
                T.ToTensor(),
            ]
        )

        local_transform = T.Compose(
            [
                base_transform,
                local_crop,
                T.ToTensor(),
            ]
        )

        # Create 2 global transforms and n_local_views local transforms
        global_transforms = [global_transform] * 2
        local_transforms = [local_transform] * n_local_views
        transforms = global_transforms + local_transforms

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
        random_crop_scale: Tuple[float, float] = (0.6, 1.0),
        focal_crop_scale: Tuple[float, float] = (0.1, 0.4),
        die_noise_prob: float = 0.03,
        rr_prob: float = 0.5,
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        rr_prob2: float = 0.25,
    ) -> None:

        base_transform = get_base_transforms(
            img_size=[random_size, random_size],
            die_noise_prob=die_noise_prob,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_prob2=rr_prob2,
            to_tensor=False,
        )

        # Create separate transforms for random and focal views
        random_crop = T.Compose(
            [
                T.RandomResizedCrop(
                    size=random_size,
                    scale=random_crop_scale,
                    ratio=(1.0, 1.0),
                    interpolation=InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
            ]
        )
        focal_crop = T.Compose(
            [
                T.RandomResizedCrop(
                    size=focal_size,
                    scale=focal_crop_scale,
                    ratio=(1.0, 1.0),
                    interpolation=InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
            ]
        )

        # Combine base transforms with random and focal crops
        transform = T.Compose([base_transform, random_crop])
        focal_transform = T.Compose([base_transform, focal_crop])

        # Put all transforms together
        transforms = [transform] * random_views
        transforms += [focal_transform] * focal_views
        super().__init__(transforms=transforms)


class WaferMAECollateFunction(MultiViewCollateFunction):
    """Implements the view augmentation for MAE.
    Unlike original paper, no cropping is performed, and we randomly rotate the image by 90 degrees.

    Parameters
    ----------
    img_size : List[int], optional
        Size of the image views, by default [200, 200]
    rr_prob : float, optional
        Probability of rotating the image by 90 degrees, by default 0.5
    hf_prob : float, optional
        Probability that horizontal flip is applied, by default 0.5
    """

    def __init__(
        self,
        img_size: List[int] = [200, 200],
        rr_prob: float = 0.5,
        hf_prob: float = 0.5,
    ):
        transforms = [
            T.ToPILImage(),
            T.Resize(img_size, interpolation=InterpolationMode.NEAREST),
            RandomRotate(rr_prob),
            T.RandomHorizontalFlip(hf_prob),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
        ]

        super().__init__([T.Compose(transforms)])

    def forward(self, batch: List[tuple]):
        views, labels, fnames = super().forward(batch)
        # Return only first view as MAE needs only a single view per image.
        return views[0], labels, fnames


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
        # All resizing is done in augmentations, so we have tensors/arrays of different sizes
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


def get_inference_transforms(img_size: List[int] = [224, 224]):
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
