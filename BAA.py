import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy
import torch
import torchvision.transforms.functional
from torch import Tensor

from torchvision.transforms import functional as F, InterpolationMode

__all__ = ["AutoAugmentPolicy", "AutoAugment", "RandAugment", "TrivialAugmentWide", "AugMix"]

import functools
import operator
import numpy as np
import PIL
from PIL import ImageOps
from PIL import Image
import math
import random

def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Cutout":
        img = Cutout(img,magnitude)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

#For DADA
def _apply_op_DADA(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = img.transform(img.size, PIL.Image.AFFINE, (1, magnitude, 0, 0, 1, 0))

    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, magnitude, 1, 0))

    elif op_name == "TranslateX":
        magnitude = magnitude * img.size[0]
        img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, magnitude, 0, 1, 0))

    elif op_name == "TranslateY":
        magnitude = magnitude * img.size[1]
        img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, magnitude))

    elif op_name == "Rotate":
        img = img.rotate(magnitude)

    elif op_name == "Brightness":
        img = PIL.ImageEnhance.Brightness(img).enhance(magnitude)
    elif op_name == "Color":
        img = PIL.ImageEnhance.Color(img).enhance(magnitude)
    elif op_name == "Contrast":
        img = PIL.ImageEnhance.Contrast(img).enhance(magnitude)
    elif op_name == "Sharpness":
        img = PIL.ImageEnhance.Sharpness(img).enhance(magnitude)
    elif op_name == "Posterize":
        img = PIL.ImageOps.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = PIL.ImageOps.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = PIL.ImageOps.autocontrast(img)
    elif op_name == "Equalize":
        img = PIL.ImageOps.equalize(img)
    # elif op_name == "Equalize":
    #     img = equalize2(img,magnitude)
    elif op_name == "Invert":
        img = PIL.ImageOps.invert(img)
    elif op_name == "Cutout":
        img = Cutout(img,magnitude)
    elif op_name == "Identity":
         pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    #assert 0.0 <= v <= 0.2
    # if v <= 0.:
    #     return img
    if v<0.:
        v=0.
    if v>2.0:
        v=2.0
    v = v * img.size[0]
    return CutoutAbs(img, v)

def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

class AutoAugmentPolicy(Enum):
    """AutoAugment policies learned on different datasets.
    Available policies are IMAGENET, CIFAR10 and SVHN.
    """

    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"

class DADAPolicy(Enum):
    """AutoAugment policies learned on different datasets.
    Available policies are IMAGENET, CIFAR10 and SVHN.
    """
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    SVHN = "svhn"
    IMAGENET = "imagenet"

class Fast_AutoAugment(torch.nn.Module):

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.policies = self._get_policies(policy)

    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [

            ]
        # in Fast AutoAugment
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                [["Contrast", 0.8320659688593578, 0.49884310562180767], ["TranslateX", 0.41849883971249136, 0.394023086494538]],
                [["Color", 0.3500483749890918, 0.43355143929883955], ["Color", 0.5120716140300229, 0.7508299643325016]],
                [["Rotate", 0.9447932604389472, 0.29723465088990375], ["Sharpness", 0.1564936149799504, 0.47169309978091745]],
                [["Rotate", 0.5430015349185097, 0.6518626678905443], ["Color", 0.5694844928020679, 0.3494533005430269]],
                [["AutoContrast", 0.5558922032451064, 0.783136004977799], ["TranslateY", 0.683914191471972, 0.7597025305860181]],
                [["TranslateX", 0.03489224481658926, 0.021025488042663354], ["Equalize", 0.4788637403857401, 0.3535481281496117]],
                [["Sharpness", 0.6428916269794158, 0.22791511918580576], ["Contrast", 0.016014045073950323, 0.26811312269487575]],
                [["Rotate", 0.2972727228410451, 0.7654251516829896], ["AutoContrast", 0.16005809254943348, 0.5380523650108116]],
                [["Contrast", 0.5823671057717301, 0.7521166301398389], ["TranslateY", 0.9949449214751978, 0.9612671341689751]],
                [["Equalize", 0.8372126687702321, 0.6944127225621206], ["Rotate", 0.25393282929784755, 0.3261658365286546]],
                [["Invert", 0.8222011603194572, 0.6597915864008403], ["Posterize", 0.31858707654447327, 0.9541013715579584]],
                [["Sharpness", 0.41314621282107045, 0.9437344470879956], ["Cutout", 0.6610495837889337, 0.674411664255093]],
                [["Contrast", 0.780121736705407, 0.40826152397463156], ["Color", 0.344019192125256, 0.1942922781355767]],
                [["Rotate", 0.17153139555621344, 0.798745732456474], ["Invert", 0.6010555860501262, 0.320742172554767]],
                [["Invert", 0.26816063450777416, 0.27152062163148327], ["Equalize", 0.6786829200236982, 0.7469412443514213]],
                [["Contrast", 0.3920564414367518, 0.7493644582838497], ["TranslateY", 0.8941657805606704, 0.6580846856375955]],
                [["Equalize", 0.875509207399372, 0.9061130537645283], ["Cutout", 0.4940280679087308, 0.7896229623628276]],
                [["Contrast", 0.3331423298065147, 0.7170041362529597], ["ShearX", 0.7425484291842793, 0.5285117152426109]],
                [["Equalize", 0.97344237365026, 0.4745759720473106], ["TranslateY", 0.055863458430295276, 0.9625142022954672]],
                [["TranslateX", 0.6810614083109192, 0.7509937355495521], ["TranslateY", 0.3866463019475701, 0.5185481505576112]],
                [["Sharpness", 0.4751529944753671, 0.550464012488733], ["Cutout", 0.9472914750534814, 0.5584925992985023]],
                [["Contrast", 0.054606784909375095, 0.17257080196712182], ["Cutout", 0.6077026782754803, 0.7996504165944938]],
                [["ShearX", 0.328798428243695, 0.2769563264079157], ["Cutout", 0.9037632437023772, 0.4915809476763595]],
                [["Cutout", 0.6891202672363478, 0.9951490996172914], ["Posterize", 0.06532762462628705, 0.4005246609075227]],
                [["TranslateY", 0.6908583592523334, 0.725612120376128], ["Rotate", 0.39907735501746666, 0.36505798032223147]],
            ]

        elif policy == AutoAugmentPolicy.SVHN:
            return [

            ]
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
            "Cutout": (torch.linspace(0.0, 0.2, num_bins), False),
        }

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(len(self.policies))

        op_meta = self._augmentation_space(10, (height, width))
        for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
            if probs[i] <= p:
                magnitudes, signed = op_meta[op_name]
                #magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                magnitude_id = round(magnitude_id * 9)
                magnitude = float(magnitudes[magnitude_id]) if op_name not in ["AutoContrast", "Equalize","Invert"] else 0.0
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(policy={self.policy}, fill={self.fill})"

class Fast_AutoAugment_BAA(torch.nn.Module):

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.policies = self._get_policies(policy)
        self.mixture_width=2 #default=2
    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [

            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                [["Contrast", 0.8320659688593578, 0.49884310562180767],["TranslateX", 0.41849883971249136, 0.394023086494538]],
                [["Color", 0.3500483749890918, 0.43355143929883955], ["Color", 0.5120716140300229, 0.7508299643325016]],
                [["Rotate", 0.9447932604389472, 0.29723465088990375],["Sharpness", 0.1564936149799504, 0.47169309978091745]],
                [["Rotate", 0.5430015349185097, 0.6518626678905443], ["Color", 0.5694844928020679, 0.3494533005430269]],
                [["AutoContrast", 0.5558922032451064, 0.783136004977799],["TranslateY", 0.683914191471972, 0.7597025305860181]],
                [["TranslateX", 0.03489224481658926, 0.021025488042663354],["Equalize", 0.4788637403857401, 0.3535481281496117]],
                [["Sharpness", 0.6428916269794158, 0.22791511918580576],["Contrast", 0.016014045073950323, 0.26811312269487575]],
                [["Rotate", 0.2972727228410451, 0.7654251516829896],["AutoContrast", 0.16005809254943348, 0.5380523650108116]],
                [["Contrast", 0.5823671057717301, 0.7521166301398389],["TranslateY", 0.9949449214751978, 0.9612671341689751]],
                [["Equalize", 0.8372126687702321, 0.6944127225621206],["Rotate", 0.25393282929784755, 0.3261658365286546]],
                [["Invert", 0.8222011603194572, 0.6597915864008403],["Posterize", 0.31858707654447327, 0.9541013715579584]],
                [["Sharpness", 0.41314621282107045, 0.9437344470879956],["Cutout", 0.6610495837889337, 0.674411664255093]],
                [["Contrast", 0.780121736705407, 0.40826152397463156],["Color", 0.344019192125256, 0.1942922781355767]],
                [["Rotate", 0.17153139555621344, 0.798745732456474], ["Invert", 0.6010555860501262, 0.320742172554767]],
                [["Invert", 0.26816063450777416, 0.27152062163148327],["Equalize", 0.6786829200236982, 0.7469412443514213]],
                [["Contrast", 0.3920564414367518, 0.7493644582838497],["TranslateY", 0.8941657805606704, 0.6580846856375955]],
                [["Equalize", 0.875509207399372, 0.9061130537645283],["Cutout", 0.4940280679087308, 0.7896229623628276]],
                [["Contrast", 0.3331423298065147, 0.7170041362529597],["ShearX", 0.7425484291842793, 0.5285117152426109]],
                [["Equalize", 0.97344237365026, 0.4745759720473106],["TranslateY", 0.055863458430295276, 0.9625142022954672]],
                [["TranslateX", 0.6810614083109192, 0.7509937355495521],["TranslateY", 0.3866463019475701, 0.5185481505576112]],
                [["Sharpness", 0.4751529944753671, 0.550464012488733],["Cutout", 0.9472914750534814, 0.5584925992985023]],
                [["Contrast", 0.054606784909375095, 0.17257080196712182],["Cutout", 0.6077026782754803, 0.7996504165944938]],
                [["ShearX", 0.328798428243695, 0.2769563264079157], ["Cutout", 0.9037632437023772, 0.4915809476763595]],
                [["Cutout", 0.6891202672363478, 0.9951490996172914],["Posterize", 0.06532762462628705, 0.4005246609075227]],
                [["TranslateY", 0.6908583592523334, 0.725612120376128],["Rotate", 0.39907735501746666, 0.36505798032223147]],
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [

            ]
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
            "Cutout": (torch.linspace(0.0, 0.2, num_bins), False),
        }

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        mix_temp = np.zeros((channels, height, width))
        mix = torch.from_numpy(mix_temp)

        op_meta = self._augmentation_space(10, (height, width))

        syn_weight= np.float32(np.random.dirichlet([1] * self.mixture_width))
        for j in range(self.mixture_width):
            transform_id, probs, signs = self.get_params(len(self.policies))
            img_ = img.copy()
            for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
                if probs[i] <= p:
                    magnitudes, signed = op_meta[op_name]
                    # magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                    magnitude_id = round(magnitude_id * 9)
                    magnitude = float(magnitudes[magnitude_id]) if op_name not in ["AutoContrast", "Equalize", "Invert"] else 0.0
                    if signed and signs[i] == 0:
                        magnitude *= -1.0
                    img_ = _apply_op(img_, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                #img_.show()
            img_ = torchvision.transforms.functional.to_tensor(img_)
            mix += syn_weight[j] * img_
        mix = torchvision.transforms.functional.to_pil_image(mix)
        #mix.show()
        return mix

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(policy={self.policy}, fill={self.fill})"

#image=policy1*weight1+policy2*weight2
#weight=random
class BAA(torch.nn.Module):

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.policies = self._get_policies(policy)
        self.mixture_width=2 #default=2
    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
                (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
                (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
                (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
                (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
                (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
                (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
                (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
                (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
                (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
                (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
                (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
                (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
                (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
                (("Color", 0.9, 9), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
                (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
                (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
                (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
                (("Equalize", 0.8, None), ("Invert", 0.1, None)),
                (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
        }

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        mix_temp = np.zeros((channels, height, width))
        mix = torch.from_numpy(mix_temp)

        op_meta = self._augmentation_space(10, (height, width))

        syn_weight= np.float32(np.random.dirichlet([1] * self.mixture_width))
        for j in range(self.mixture_width):
            transform_id, probs, signs = self.get_params(len(self.policies))
            img_ = img.copy()
            for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
                if probs[i] <= p:
                    magnitudes, signed = op_meta[op_name]
                    magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                    if signed and signs[i] == 0:
                        magnitude *= -1.0
                    img_ = _apply_op(img_, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                #img_.show()
            img_ = torchvision.transforms.functional.to_tensor(img_)
            mix += syn_weight[j] * img_
        mix = torchvision.transforms.functional.to_pil_image(mix)
        #mix.show()
        return mix

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(policy={self.policy}, fill={self.fill})"

class DADA(torch.nn.Module):

    def __init__(
        self,
        policy: DADAPolicy = DADAPolicy.CIFAR10,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.policies = self._get_policies(policy)

    def _get_policies(
        self, policy: DADAPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == DADAPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
        elif policy == DADAPolicy.CIFAR10: # DADA Search in paper
            return [
                [('TranslateX', 0.5183464288711548, 0.5752825736999512),('Rotate', 0.5693835616111755, 0.5274667739868164)],
                [('ShearX', 0.5028448104858398, 0.4595153331756592),('Sharpness', 0.5036740303039551, 0.5378073453903198)],
                [('Brightness', 0.5574088096618652, 0.5563607811927795), ('Sharpness', 0.5241265296936035, 0.4670485556125641)],
                [('ShearY', 0.6167989373207092, 0.4837495684623718), ('Brightness', 0.4740375578403473, 0.4566589295864105)],
                [('ShearX', 0.4406820833683014, 0.5844581723213196),('TranslateY', 0.3973384499549866, 0.5110136270523071)],
                [('Rotate', 0.39651215076446533, 0.517960250377655),('Equalize', 0.3761792480945587, 0.36424142122268677)],
                [('AutoContrast', 0.4399465024471283, 0.48347008228302), ('Cutout', 0.49450358748435974, 0.5)],
                [('AutoContrast', 0.5601025819778442, 0.479571133852005),('Color', 0.44876909255981445, 0.6113184690475464)],
                [('Rotate', 0.4231756627559662, 0.6356207132339478),('AutoContrast', 0.59876549243927, 0.5785224437713623)],
                [('Invert', 0.39854735136032104, 0.5), ('Color', 0.4968028664588928, 0.4379926025867462)],
                [('Posterize', 0.5603401064872742, 0.49880319833755493),('Brightness', 0.5293631553649902, 0.47918644547462463)],
                [('TranslateY', 0.4231869578361511, 0.5149744749069214),('AutoContrast', 0.3750160336494446, 0.5654526352882385)],
                [('ShearX', 0.3773101270198822, 0.5), ('Contrast', 0.485131174325943, 0.5186365842819214)],
                [('ShearY', 0.5420096516609192, 0.6018360257148743),('Rotate', 0.30701273679733276, 0.5576906800270081)],
                [('Posterize', 0.4173527657985687, 0.4971156716346741),('Color', 0.450246661901474, 0.5576846599578857)],
                [('TranslateX', 0.4143139123916626, 0.450955331325531),('TranslateY', 0.3599278926849365, 0.4812163710594177)],
                [('TranslateX', 0.574902355670929, 0.5), ('Brightness', 0.5378687381744385, 0.4751467704772949)],
                [('TranslateX', 0.5295567512512207, 0.5137100219726562),('Cutout', 0.6851200461387634, 0.4909016489982605)],
                [('ShearX', 0.4579184353351593, 0.44350510835647583),('Invert', 0.41791805624961853, 0.3984798192977905)],
                [('Rotate', 0.49958375096321106, 0.4244190752506256),('Contrast', 0.49455592036247253, 0.4244190752506256)],
                [('Rotate', 0.42886781692504883, 0.46792319416999817),('Solarize', 0.49862080812454224, 0.41575634479522705)],
                [('TranslateY', 0.7362872958183289, 0.5113809704780579),('Color', 0.3918609917163849, 0.5744326114654541)],
                [('Equalize', 0.42245715856552124, 0.5293998718261719),('Sharpness', 0.39708659052848816, 0.43052399158477783)],
                [('Solarize', 0.7260022759437561, 0.42120808362960815),('Cutout', 0.5075806379318237, 0.46120622754096985)],
                [('ShearX', 0.5757555961608887, 0.563892662525177),('TranslateX', 0.4761257469654083, 0.49035176634788513)]
            ]

        elif policy == DADAPolicy.CIFAR100:  # DADA Search in paper
            return [
                [('ShearY', 0.5558360815048218, 0.2755777835845947),('Sharpness', 0.4881928265094757, 0.21691159904003143)],
                [('Rotate', 0.3616902828216553, 0.18580017983913422), ('Contrast', 0.5613550543785095, 0.3083491921424866)],
                [('TranslateY', 0.0, 0.408305287361145),('Brightness', 0.4716355502605438, 0.5204870104789734)],
                [('AutoContrast', 0.7991832494735718, 0.44347289204597473),('Color', 0.4360397756099701, 0.36640283465385437)],
                [('Color', 0.9439005255699158, 0.25039201974868774),('Brightness', 0.6770737767219543, 0.44653841853141785)],
                [('TranslateY', 0.6285494565963745, 0.395266592502594),('Equalize', 0.8238864541053772, 0.2982426583766937)],
                [('Equalize', 0.46301358938217163, 0.7147184610366821),('Posterize', 0.5021865367889404, 0.7205064296722412)],
                [('Color', 0.5189914703369141, 0.482077032327652),('Sharpness', 0.19046853482723236, 0.39753425121307373)],
                [('Sharpness', 0.4223572611808777, 0.37934350967407227),('Cutout', 0.5508846044540405, 0.2419460117816925)],
                [('ShearX', 0.744318962097168, 0.5582589507102966),('TranslateX', 0.4841197729110718, 0.6740695834159851)],
                [('Invert', 0.3582773506641388, 0.5917970538139343),('Brightness', 0.5021751523017883, 0.2252121865749359)],
                [('TranslateX', 0.3551332652568817, 0.3603559732437134),('Posterize', 0.7993623614311218, 0.3243299722671509)],
                [('TranslateX', 0.47883617877960205, 0.356031209230423),('Cutout', 0.6418997049331665, 0.6689953207969666)],
                [('Posterize', 0.30889445543289185, 0.03791777789592743),('Contrast', 1.0, 0.0762958824634552)],
                [('Contrast', 0.4202372133731842, 0.25662222504615784),('Cutout', 0.0, 0.44168394804000854)],
                [('Equalize', 0.15656249225139618, 0.6881861686706543),('Brightness', 0.72562175989151, 0.177269846200943)],
                [('Contrast', 0.45150792598724365, 0.33532920479774475),('Sharpness', 0.589333176612854, 0.27893581986427307)],
                [('TranslateX', 0.12946638464927673, 0.5427500009536743),('Invert', 0.3318890929222107, 0.4762207269668579)],
                [('Rotate', 0.4988836646080017, 0.5838819742202759), ('Posterize', 1.0, 0.7428610324859619)],
                [('TranslateX', 0.5078659057617188, 0.4304893910884857),('Rotate', 0.4633696377277374, 0.48235252499580383)],
                [('ShearX', 0.5783067941665649, 0.455081969499588),('TranslateY', 0.32670140266418457, 0.30824044346809387)],
                [('Rotate', 1.0, 0.0), ('Equalize', 0.510175883769989, 0.37381383776664734)],
                [('AutoContrast', 0.25783753395080566, 0.5729542374610901),('Cutout', 0.3359143137931824, 0.34972965717315674)],
                [('ShearX', 0.5556561350822449, 0.5526643991470337),('Color', 0.5021030902862549, 0.5042688846588135)],
                [('ShearY', 0.4626863896846771, 0.08647401630878448),('Posterize', 0.5513173341751099, 0.3414877951145172)]

            ]
        # elif policy == DADAPolicy.CIFAR100: # DADA Search epoch=100
        #     return [
        #         (('Color', 0.5126565098762512, 0.44821491837501526), ('Sharpness', 0.5106418132781982, 0.44375038146972656)),
        #         (('TranslateX', 0.4912850260734558, 0.48561155796051025),('Brightness', 0.5101174712181091, 0.3431628346443176)),
        #         (('Contrast', 0.5098063349723816, 0.4405685067176819), ('Color', 0.507108747959137, 0.455096572637558)),
        #         (('Posterize', 0.5087332129478455, 0.46905896067619324),('Sharpness', 0.4941613972187042, 0.46371862292289734)),
        #         (('TranslateY', 0.5169646739959717, 0.4589107036590576), ('Color', 0.4764978289604187, 0.4636608958244324)),
        #         (('ShearX', 0.5039932131767273, 0.44715577363967896), ('Rotate', 0.516126811504364, 0.465667188167572)),
        #         (('TranslateY', 0.48406633734703064, 0.48456937074661255),('AutoContrast', 0.49910295009613037, 0.3520564138889313)),
        #         (('Contrast', 0.5029032826423645, 0.3411848545074463), ('Brightness', 0.5308189392089844, 0.4575544595718384)),
        #         (('Rotate', 0.4901990592479706, 0.4903252124786377), ('AutoContrast', 0.5092589855194092, 0.4285391569137573)),
        #         (('ShearX', 0.5183875560760498, 0.4481567144393921), ('Cutout', 0.4924670457839966, 0.46177756786346436)),
        #         (('Equalize', 0.5197846293449402, 0.4460393190383911),('Brightness', 0.5187625885009766, 0.44701293110847473)),
        #         (('TranslateX', 0.492715448141098, 0.4272593557834625), ('Equalize', 0.47879651188850403, 0.4710386097431183)),
        #         (('ShearX', 0.5028236508369446, 0.4491778016090393), ('ShearY', 0.49801355600357056, 0.4652498960494995)),
        #         (('TranslateY', 0.501209557056427, 0.4477178454399109), ('Rotate', 0.4738703668117523, 0.4499082863330841)),
        #         (('ShearX', 0.5322993993759155, 0.4046383798122406),('AutoContrast', 0.4714276194572449, 0.49322521686553955)),
        #         (('Brightness', 0.4805971682071686, 0.3745571970939636),('Sharpness', 0.5152804851531982, 0.3606525957584381)),
        #         (('Posterize', 0.5056659579277039, 0.4743610918521881), ('Color', 0.5147949457168579, 0.3432264029979706)),
        #         (('AutoContrast', 0.5024915337562561, 0.3729042410850525),('Cutout', 0.4904150664806366, 0.46811842918395996)),
        #         (('TranslateY', 0.5050394535064697, 0.40086692571640015),('Brightness', 0.5097253918647766, 0.421572208404541)),
        #         (('TranslateX', 0.48027822375297546, 0.436251163482666), ('Cutout', 0.49190807342529297, 0.43370574712753296)),
        #         (('Rotate', 0.4857220947742462, 0.4618608057498932), ('Contrast', 0.47429877519607544, 0.4226512908935547)),
        #         (('ShearY', 0.5313718914985657, 0.45189976692199707), ('AutoContrast', 0.5272999405860901, 0.466357946395874)),
        #         (('AutoContrast', 0.5089313983917236, 0.427597314119339),('Brightness', 0.5202492475509644, 0.39580386877059937)),
        #         (('Posterize', 0.5255964994430542, 0.40627235174179077),('Brightness', 0.5244666934013367, 0.41001251339912415)),
        #         (('ShearY', 0.5084934234619141, 0.45244520902633667), ('Posterize', 0.5109863877296448, 0.43440791964530945))
        #         ]
        # elif policy == DADAPolicy.CIFAR100:  # DADA Search epoch=200
        #     return [
        #              (('TranslateX', 0.49519214034080505, 0.4128504693508148),('Brightness', 0.5262572765350342, 0.25239863991737366)),
        #              (('Brightness', 0.5151030421257019, 0.28564244508743286),('Sharpness', 0.5393402576446533, 0.3677419424057007)),
        #              (('Contrast', 0.504188597202301, 0.3989821672439575),('Color', 0.49428367614746094, 0.39746755361557007)),
        #              (('Posterize', 0.47629088163375854, 0.39852914214134216),('Color', 0.5341065526008606, 0.2960209548473358)),
        #              (('Posterize', 0.5018191337585449, 0.410321444272995),('Sharpness', 0.5187022686004639, 0.4044341742992401)),
        #              (('Color', 0.489119291305542, 0.2532801628112793),('Brightness', 0.5558120608329773, 0.3568388521671295)),
        #              (('TranslateY', 0.5586605668067932, 0.40853041410446167),('Color', 0.48225992918014526, 0.34288835525512695)),
        #              (('Rotate', 0.508212149143219, 0.43404898047447205),('AutoContrast', 0.4522525370121002, 0.3632649779319763)),
        #              (('Posterize', 0.5278016328811646, 0.25486454367637634),('Contrast', 0.5423815846443176, 0.22771300375461578)),
        #              (('Color', 0.533390462398529, 0.4018372893333435),('Sharpness', 0.4842771887779236, 0.3928672671318054)),
        #              (('ShearX', 0.5329526662826538, 0.41086146235466003),('Rotate', 0.526030957698822, 0.42075538635253906)),
        #              (('AutoContrast', 0.5178370475769043, 0.37236708402633667),('Brightness', 0.5175914764404297, 0.3484806418418884)),
        #              (('Contrast', 0.4932439923286438, 0.3639858663082123),('Brightness', 0.5098784565925598, 0.4017108678817749)),
        #              (('TranslateY', 0.5622210502624512, 0.34088754653930664),('Brightness', 0.5204827785491943, 0.3742882311344147)),
        #              (('TranslateY', 0.4666820168495178, 0.42854899168014526),('AutoContrast', 0.4587632119655609, 0.327176958322525)),
        #              (('ShearX', 0.5505303144454956, 0.35762861371040344),('AutoContrast', 0.48612356185913086, 0.42598432302474976)),
        #              (('Equalize', 0.49401360750198364, 0.392402708530426),('Brightness', 0.5683338642120361, 0.3809775412082672)),
        #              (('Contrast', 0.47595784068107605, 0.3983074426651001),('Cutout', 0.5105708241462708, 0.34365829825401306)),
        #              (('Posterize', 0.5353227257728577, 0.41490015387535095),('Brightness', 0.5203555822372437, 0.4221963882446289)),
        #              (('Equalize', 0.510892927646637, 0.4216077923774719),('Color', 0.4980945289134979, 0.4014135003089905)),
        #              (('ShearY', 0.5340392589569092, 0.3680362403392792),('Brightness', 0.5174405574798584, 0.3540394604206085)),
        #              (('ShearY', 0.5151561498641968, 0.40351033210754395),('Cutout', 0.47045424580574036, 0.3731778562068939)),
        #              (('Contrast', 0.4996435344219208, 0.31362760066986084),('Sharpness', 0.5037422776222229, 0.21258455514907837)),
        #              (('Posterize', 0.45472291111946106, 0.352789044380188),('Cutout', 0.49143147468566895, 0.23919829726219177)),
        #              (('Brightness', 0.47342726588249207, 0.3653409779071808),('Cutout', 0.49691465497016907, 0.180319681763649))
        #         ]

        elif policy == DADAPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    # def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
    #     return {
    #         # op_name: (magnitudes, signed)
    #         "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
    #         "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
    #         "TranslateX": (torch.linspace(0.0, 0.45, num_bins), True),
    #         "TranslateY": (torch.linspace(0.0, 0.45, num_bins), True),
    #         "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
    #         "Brightness": (torch.linspace(0.1, 1.9, num_bins), True),
    #         "Color": (torch.linspace(0.1, 1.9, num_bins), True),
    #         "Contrast": (torch.linspace(0.1, 1.9, num_bins), True),
    #         "Sharpness": (torch.linspace(0.1, 1.9, num_bins), True),
    #         "Posterize": (torch.linspace(4, 8, num_bins).round().int(), False),
    #         "Solarize": (torch.linspace(0, 256, num_bins).round().int(), False),
    #         "AutoContrast": (torch.tensor(0.0), False),
    #         "Equalize": (torch.tensor(0.0), False),
    #         "Invert": (torch.tensor(0.0), False),
    #         "Cutout": (torch.linspace(0.0, 0.2, num_bins), False),
    #     }
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(-0.3, 0.3, num_bins), False),
            "ShearY": (torch.linspace(-0.3, 0.3, num_bins), False),
            "TranslateX": (torch.linspace(-0.45, 0.45, num_bins), False),
            "TranslateY": (torch.linspace(-0.45, 0.45, num_bins), False),
            "Rotate": (torch.linspace(-30.0, 30.0, num_bins), False),
            "Brightness": (torch.linspace(0.1, 1.9, num_bins), False),
            "Color": (torch.linspace(0.1, 1.9, num_bins), False),
            "Contrast": (torch.linspace(0.1, 1.9, num_bins), False),
            "Sharpness": (torch.linspace(0.1, 1.9, num_bins), False),
            "Posterize": (torch.linspace(4, 8, num_bins).round().int(), False),
            "Solarize": (torch.linspace(0, 256, num_bins).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
            "Cutout": (torch.linspace(0.0, 0.2, num_bins), False),
        }
    #범위 수정
    # def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
    #     return {
    #         # op_name: (magnitudes, signed)
    #         "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), #-0.5<=v<=0.5
    #         "ShearY": (torch.linspace(0.0, 0.5, num_bins), True), #-0.5<=v<=0.5
    #         "TranslateX": (torch.linspace(0.0, 0.5, num_bins), True), #-0.5<=v<=0.5
    #         "TranslateY": (torch.linspace(0.0, 0.5, num_bins), True), #-0.5<=v<=0.5
    #         "Rotate": (torch.linspace(0.0, 90.0, num_bins), True), #-90<=v<=90
    #         "Brightness": (torch.linspace(0.1, 2.0, num_bins), False), #0.1<=v<=2.0
    #         "Color": (torch.linspace(0, 10, num_bins), True), #-10<=v<=10.0
    #         "Contrast": (torch.linspace(0.1, 10.0, num_bins), False), #0.1<=v<=10.0
    #         "Sharpness": (torch.linspace(0.0, 100, num_bins), True), #-100<= v<=100
    #         "Posterize": (torch.linspace(1, 8, num_bins).round().int(), False), #1<=v<=8
    #         "Solarize": (torch.linspace(0, 256, num_bins).round().int(), False), #0<=v<=256
    #         #"AutoContrast": (torch.tensor(0.0), False),
    #         "Equalize": (torch.linspace(0.0, 7.0, num_bins), False),  #0<=v<=7
    #         #"Invert": (torch.tensor(0.0), False),
    #         "Cutout": (torch.linspace(0.0, 0.4, num_bins), False), #0<=v<=0.4
    #     }

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(len(self.policies))

        op_meta = self._augmentation_space(10, (height, width))
        # print("@@@")
        # print(op_meta)
        #print(transform_id)
        #print(probs)
        #print(signs)
        #print(self.policies[transform_id])

        for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
            if probs[i] <= p:
                magnitudes_range, signed = op_meta[op_name]
                #print('op_name:{0}, p:{1}, magnitude_id:{2}'.format(op_name,p,magnitude_id))
                #magnitude_id=round(magnitude_id*10) #magnitude_id=1.0 * 10.0 = 10 이면 magnitudes_range[10]이 되어 out of bound 에러가 뜸 magnitudes_range[]는 0~9 의 값을 같는 10단계
                magnitude_id = round(magnitude_id * 9)
                #print('op_name:{0}, p:{1}, magnitude_id:{2}'.format(op_name, p, magnitude_id))
                #print('magnitudes:{0}, signed:{1}'.format(magnitudes_range, signed))

                magnitude = float(magnitudes_range[magnitude_id]) if op_name not in ["AutoContrast", "Equalize", "Invert"]  else 0.0

                #magnitude = float(magnitudes_range[magnitude_id])
                # if signed and signs[i] == 0:
                #     magnitude *= -1.0

                #print('magnitudes:{0}, signed:{1}'.format(magnitudes_range, signed))
                #print('op_name:{0}, p:{1}, magnitude_id:{2}'.format(op_name, p, magnitude_id))
                img = _apply_op_DADA(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(policy={self.policy}, fill={self.fill})"

#image=policy1*weight1+policy2*weight2
#weight=random
class DADA_BAA(torch.nn.Module):

    def __init__(
        self,
        policy: DADAPolicy = DADAPolicy.CIFAR10,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.policies = self._get_policies(policy)
        self.mixture_width=2 #default=2
    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]

        elif policy == DADAPolicy.CIFAR10: # DADA Search in paper
            return [
                [('TranslateX', 0.5183464288711548, 0.5752825736999512),('Rotate', 0.5693835616111755, 0.5274667739868164)],
                [('ShearX', 0.5028448104858398, 0.4595153331756592),('Sharpness', 0.5036740303039551, 0.5378073453903198)],
                [('Brightness', 0.5574088096618652, 0.5563607811927795), ('Sharpness', 0.5241265296936035, 0.4670485556125641)],
                [('ShearY', 0.6167989373207092, 0.4837495684623718), ('Brightness', 0.4740375578403473, 0.4566589295864105)],
                [('ShearX', 0.4406820833683014, 0.5844581723213196),('TranslateY', 0.3973384499549866, 0.5110136270523071)],
                [('Rotate', 0.39651215076446533, 0.517960250377655),('Equalize', 0.3761792480945587, 0.36424142122268677)],
                [('AutoContrast', 0.4399465024471283, 0.48347008228302), ('Cutout', 0.49450358748435974, 0.5)],
                [('AutoContrast', 0.5601025819778442, 0.479571133852005),('Color', 0.44876909255981445, 0.6113184690475464)],
                [('Rotate', 0.4231756627559662, 0.6356207132339478),('AutoContrast', 0.59876549243927, 0.5785224437713623)],
                [('Invert', 0.39854735136032104, 0.5), ('Color', 0.4968028664588928, 0.4379926025867462)],
                [('Posterize', 0.5603401064872742, 0.49880319833755493),('Brightness', 0.5293631553649902, 0.47918644547462463)],
                [('TranslateY', 0.4231869578361511, 0.5149744749069214),('AutoContrast', 0.3750160336494446, 0.5654526352882385)],
                [('ShearX', 0.3773101270198822, 0.5), ('Contrast', 0.485131174325943, 0.5186365842819214)],
                [('ShearY', 0.5420096516609192, 0.6018360257148743),('Rotate', 0.30701273679733276, 0.5576906800270081)],
                [('Posterize', 0.4173527657985687, 0.4971156716346741),('Color', 0.450246661901474, 0.5576846599578857)],
                [('TranslateX', 0.4143139123916626, 0.450955331325531),('TranslateY', 0.3599278926849365, 0.4812163710594177)],
                [('TranslateX', 0.574902355670929, 0.5), ('Brightness', 0.5378687381744385, 0.4751467704772949)],
                [('TranslateX', 0.5295567512512207, 0.5137100219726562),('Cutout', 0.6851200461387634, 0.4909016489982605)],
                [('ShearX', 0.4579184353351593, 0.44350510835647583),('Invert', 0.41791805624961853, 0.3984798192977905)],
                [('Rotate', 0.49958375096321106, 0.4244190752506256),('Contrast', 0.49455592036247253, 0.4244190752506256)],
                [('Rotate', 0.42886781692504883, 0.46792319416999817),('Solarize', 0.49862080812454224, 0.41575634479522705)],
                [('TranslateY', 0.7362872958183289, 0.5113809704780579),('Color', 0.3918609917163849, 0.5744326114654541)],
                [('Equalize', 0.42245715856552124, 0.5293998718261719),('Sharpness', 0.39708659052848816, 0.43052399158477783)],
                [('Solarize', 0.7260022759437561, 0.42120808362960815),('Cutout', 0.5075806379318237, 0.46120622754096985)],
                [('ShearX', 0.5757555961608887, 0.563892662525177),('TranslateX', 0.4761257469654083, 0.49035176634788513)],
            ]
        elif policy == DADAPolicy.CIFAR100:  # DADA Search in paper
            return [
                [('ShearY', 0.5558360815048218, 0.2755777835845947),
                 ('Sharpness', 0.4881928265094757, 0.21691159904003143)],
                [('Rotate', 0.3616902828216553, 0.18580017983913422),
                 ('Contrast', 0.5613550543785095, 0.3083491921424866)],
                [('TranslateY', 0.0, 0.408305287361145), ('Brightness', 0.4716355502605438, 0.5204870104789734)],
                [('AutoContrast', 0.7991832494735718, 0.44347289204597473),
                 ('Color', 0.4360397756099701, 0.36640283465385437)],
                [('Color', 0.9439005255699158, 0.25039201974868774),
                 ('Brightness', 0.6770737767219543, 0.44653841853141785)],
                [('TranslateY', 0.6285494565963745, 0.395266592502594),
                 ('Equalize', 0.8238864541053772, 0.2982426583766937)],
                [('Equalize', 0.46301358938217163, 0.7147184610366821),
                 ('Posterize', 0.5021865367889404, 0.7205064296722412)],
                [('Color', 0.5189914703369141, 0.482077032327652),
                 ('Sharpness', 0.19046853482723236, 0.39753425121307373)],
                [('Sharpness', 0.4223572611808777, 0.37934350967407227),
                 ('Cutout', 0.5508846044540405, 0.2419460117816925)],
                [('ShearX', 0.744318962097168, 0.5582589507102966),
                 ('TranslateX', 0.4841197729110718, 0.6740695834159851)],
                [('Invert', 0.3582773506641388, 0.5917970538139343),
                 ('Brightness', 0.5021751523017883, 0.2252121865749359)],
                [('TranslateX', 0.3551332652568817, 0.3603559732437134),
                 ('Posterize', 0.7993623614311218, 0.3243299722671509)],
                [('TranslateX', 0.47883617877960205, 0.356031209230423),
                 ('Cutout', 0.6418997049331665, 0.6689953207969666)],
                [('Posterize', 0.30889445543289185, 0.03791777789592743), ('Contrast', 1.0, 0.0762958824634552)],
                [('Contrast', 0.4202372133731842, 0.25662222504615784), ('Cutout', 0.0, 0.44168394804000854)],
                [('Equalize', 0.15656249225139618, 0.6881861686706543),
                 ('Brightness', 0.72562175989151, 0.177269846200943)],
                [('Contrast', 0.45150792598724365, 0.33532920479774475),
                 ('Sharpness', 0.589333176612854, 0.27893581986427307)],
                [('TranslateX', 0.12946638464927673, 0.5427500009536743),
                 ('Invert', 0.3318890929222107, 0.4762207269668579)],
                [('Rotate', 0.4988836646080017, 0.5838819742202759), ('Posterize', 1.0, 0.7428610324859619)],
                [('TranslateX', 0.5078659057617188, 0.4304893910884857),
                 ('Rotate', 0.4633696377277374, 0.48235252499580383)],
                [('ShearX', 0.5783067941665649, 0.455081969499588),
                 ('TranslateY', 0.32670140266418457, 0.30824044346809387)],
                [('Rotate', 1.0, 0.0), ('Equalize', 0.510175883769989, 0.37381383776664734)],
                [('AutoContrast', 0.25783753395080566, 0.5729542374610901),
                 ('Cutout', 0.3359143137931824, 0.34972965717315674)],
                [('ShearX', 0.5556561350822449, 0.5526643991470337), ('Color', 0.5021030902862549, 0.5042688846588135)],
                [('ShearY', 0.4626863896846771, 0.08647401630878448),
                 ('Posterize', 0.5513173341751099, 0.3414877951145172)],
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(-0.3, 0.3, num_bins), False),
            "ShearY": (torch.linspace(-0.3, 0.3, num_bins), False),
            "TranslateX": (torch.linspace(-0.45, 0.45, num_bins), False),
            "TranslateY": (torch.linspace(-0.45, 0.45, num_bins), False),
            "Rotate": (torch.linspace(-30.0, 30.0, num_bins), False),
            "Brightness": (torch.linspace(0.1, 1.9, num_bins), False),
            "Color": (torch.linspace(0.1, 1.9, num_bins), False),
            "Contrast": (torch.linspace(0.1, 1.9, num_bins), False),
            "Sharpness": (torch.linspace(0.1, 1.9, num_bins), False),
            "Posterize": (torch.linspace(4, 8, num_bins).round().int(), False),
            "Solarize": (torch.linspace(0, 256, num_bins).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
            "Cutout": (torch.linspace(0.0, 0.2, num_bins), False),
        }

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        mix_temp = np.zeros((channels, height, width))
        mix = torch.from_numpy(mix_temp)

        op_meta = self._augmentation_space(10, (height, width))

        syn_weight= np.float32(np.random.dirichlet([1] * self.mixture_width))
        for j in range(self.mixture_width):
            transform_id, probs, signs = self.get_params(len(self.policies))
            img_ = img.copy()
            for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
                magnitude_id = round(magnitude_id * 9)
                if probs[i] <= p:
                    magnitudes, signed = op_meta[op_name]
                    #magnitude = float(magnitudes[magnitude_id].item())
                    magnitude = float(magnitudes[magnitude_id]) if op_name not in ["AutoContrast", "Equalize","Invert"] else 0.0
                    #magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                    # if signed and signs[i] == 0:
                    #     magnitude *= -1.0
                    img_ = _apply_op_DADA(img_, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                #img_.show()
            img_ = torchvision.transforms.functional.to_tensor(img_)
            mix += syn_weight[j] * img_
        mix = torchvision.transforms.functional.to_pil_image(mix)
        #mix.show()
        return mix

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(policy={self.policy}, fill={self.fill})"

class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        #논문에서는 14개의 augmentation을 사용하였고 파라미터 N은 적용할 augmentation개수, M은 적용할 augmentation의 magnitude(변환정도)를 의미한다.
        #각 기법마다 최소 최대 변환값을 정해놓고 magnitude를 통해 N개의 augmentation 기법들이 모두 같은 정도의 변환이 일어나게 한다.

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

#image=policy1*weight1+policy2*weight2
#weight=random
class RandAugment_BAA(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.mixture_width = 2  # default=2

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        #논문에서는 14개의 augmentation을 사용하였고 파라미터 N은 적용할 augmentation개수, M은 적용할 augmentation의 magnitude(변환정도)를 의미한다.
        #각 기법마다 최소 최대 변환값을 정해놓고 magnitude를 통해 N개의 augmentation 기법들이 모두 같은 정도의 변환이 일어나게 한다.

        mix_temp = np.zeros((channels, height, width))
        mix = torch.from_numpy(mix_temp)
        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        syn_weight = np.float32(np.random.dirichlet([1] * self.mixture_width))
        for j in range(self.mixture_width):
            img_ = img.copy()
            for i in range(self.num_ops):
                op_index = int(torch.randint(len(op_meta), (1,)).item())
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0
                img_ = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                #img_.show()
            img_ = torchvision.transforms.functional.to_tensor(img_)
            mix += syn_weight[j] * img_
        mix = torchvision.transforms.functional.to_pil_image(mix)
        #mix.show()
        return mix

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

class TrivialAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

class AugMix(torch.nn.Module):
    r"""AugMix data augmentation method based on
    `"AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty" <https://arxiv.org/abs/1912.02781>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        severity (int): The severity of base augmentation operators. Default is ``3``.
        mixture_width (int): The number of augmentation chains. Default is ``3``.
        chain_depth (int): The depth of augmentation chains. A negative value denotes stochastic depth sampled from the interval [1, 3].
            Default is ``-1``.
        alpha (float): The hyperparameter for the probability distributions. Default is ``1.0``.
        all_ops (bool): Use all operations (including brightness, contrast, color and sharpness). Default is ``True``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
        all_ops: bool = True,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self._PARAMETER_MAX = 10
        if not (1 <= severity <= self._PARAMETER_MAX):
            raise ValueError(f"The severity must be between [1, {self._PARAMETER_MAX}]. Got {severity} instead.")
        self.severity = severity
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth
        self.alpha = alpha
        self.all_ops = all_ops
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        s = {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, image_size[1] / 3.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, image_size[0] / 3.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Posterize": (4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
        if self.all_ops:
            s.update(
                {
                    "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                }
            )
        return s

    @torch.jit.unused
    def _pil_to_tensor(self, img) -> Tensor:
        return F.pil_to_tensor(img)

    @torch.jit.unused
    def _tensor_to_pil(self, img: Tensor):
        return F.to_pil_image(img)

    def _sample_dirichlet(self, params: Tensor) -> Tensor:
        # Must be on a separate method so that we can overwrite it in tests.
        return torch._sample_dirichlet(params)

    def forward(self, orig_img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(orig_img)
        if isinstance(orig_img, Tensor):
            img = orig_img
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]
        else:
            img = self._pil_to_tensor(orig_img)

        op_meta = self._augmentation_space(self._PARAMETER_MAX, (height, width))

        orig_dims = list(img.shape)
        batch = img.view([1] * max(4 - img.ndim, 0) + orig_dims)
        batch_dims = [batch.size(0)] + [1] * (batch.ndim - 1)

        # Sample the beta weights for combining the original and augmented image. To get Beta, we use a Dirichlet
        # with 2 parameters. The 1st column stores the weights of the original and the 2nd the ones of augmented image.
        m = self._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha], device=batch.device).expand(batch_dims[0], -1)
        )

        # Sample the mixing weights and combine them with the ones sampled from Beta for the augmented images.
        combined_weights = self._sample_dirichlet(
            torch.tensor([self.alpha] * self.mixture_width, device=batch.device).expand(batch_dims[0], -1)
        ) * m[:, 1].view([batch_dims[0], -1])

        mix = m[:, 0].view(batch_dims) * batch
        for i in range(self.mixture_width):
            aug = batch
            depth = self.chain_depth if self.chain_depth > 0 else int(torch.randint(low=1, high=4, size=(1,)).item())
            for _ in range(depth):
                op_index = int(torch.randint(len(op_meta), (1,)).item())
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                magnitude = (
                    float(magnitudes[torch.randint(self.severity, (1,), dtype=torch.long)].item())
                    if magnitudes.ndim > 0
                    else 0.0
                )
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0
                aug = _apply_op(aug, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            mix.add_(combined_weights[:, i].view(batch_dims) * aug)
        mix = mix.view(orig_dims).to(dtype=img.dtype)

        if not isinstance(orig_img, Tensor):
            return self._tensor_to_pil(mix)
        return mix

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"severity={self.severity}"
            f", mixture_width={self.mixture_width}"
            f", chain_depth={self.chain_depth}"
            f", alpha={self.alpha}"
            f", all_ops={self.all_ops}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s





