import os
from typing import Callable, Optional

from torchvision.datasets.folder import ImageFolder


class RESISC(ImageFolder):
    """RGB version of the `EuroSAT <https://github.com/phelber/eurosat>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``root/eurosat`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        train: bool = True
    ) -> None:
        self.root = os.path.expanduser(root)
        self._base_folder = os.path.join(self.root, "RESISC")
        super().__init__(self._base_folder, transform=transform, target_transform=target_transform)
        self.root = os.path.expanduser(root)

    def __len__(self) -> int:
        return len(self.samples)