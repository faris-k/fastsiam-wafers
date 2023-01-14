import numpy as np
import torch
import torchvision.transforms as T
from lightly.data.collate import MultiViewCollateFunction
from lightly.transforms.rotation import RandomRotate
from torchvision.transforms.functional import InterpolationMode


class DieNoise:
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        def flip(item, p=0.03):
            """
            Given a wafermap die, flips pass to fail and vice-versa with probability p.
            Does nothing to non-die area (0's if 128's and 255's are passes/fails respectively).
            """
            prob = np.random.choice([False, True], p=[1 - p, p])
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


base_transforms = T.Compose(
    [
        # Add die noise before anything else
        DieNoise(),
        # Convert to PIL Image, then perform all torchvision transforms
        T.ToPILImage(),
        T.Resize([224, 224], interpolation=InterpolationMode.NEAREST),
        RandomRotate(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomHorizontalFlip(0.5),
        T.RandomApply(
            torch.nn.ModuleList(
                [T.RandomRotation(90, interpolation=InterpolationMode.NEAREST)]
            ),
            0.25,
        ),
        # Finally, create a 3-channel image and convert to tensor
        T.Grayscale(num_output_channels=3),  # R == G == B
        T.ToTensor(),
    ]
)

FastSiamCollateFunction = MultiViewCollateFunction([base_transforms] * 4)


class WaferFastSiamCollateFunction(MultiViewCollateFunction):
    def __init__(
        self,
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
                T.Resize([224, 224], interpolation=InterpolationMode.NEAREST),
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
                T.ToTensor(),
            ]
        )
        super().__init__([base_transforms] * 4)


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
                T.Resize([224, 224], interpolation=InterpolationMode.NEAREST),
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
