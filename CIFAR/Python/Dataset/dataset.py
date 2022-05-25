import os
from abc import ABC

import torch
from torch.utils.data import Dataset
import PIL.Image as PILImage
from torchvision.transforms import transforms
from random import shuffle
from typing import Tuple, List, Dict


class CIFARTorchDataset(Dataset, ABC):
    def __init__(self, path: str):
        self.sample_table, self.target2id = self.__directory_traversal(path=path)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        sample = self.sample_table[index]
        target = sample[0]
        image = self.__get_image(path=sample[1])

        return target, image

    @staticmethod
    def __directory_traversal(path: str) -> Tuple[List[Tuple[int, str]], Dict[int, str]]:
        sample_table, target2id = list(), dict()
        image_dirs = os.listdir(path)

        target_id = -1
        for image_dir in image_dirs:

            image_dir_path = os.path.join(path, image_dir)
            if not os.path.isdir(image_dir_path):
                continue

            target_id += 1
            target2id[target_id] = image_dir

            image_files_path = os.listdir(image_dir_path)
            for image_file_path in image_files_path:
                image_file_path_ = os.path.join(image_dir_path, image_file_path)
                sample_table.append((target_id, image_file_path_))

        shuffle(sample_table)
        return sample_table, target2id

    def __len__(self) -> int:
        return len(self.sample_table)

    def __get_image(self, path) -> torch.Tensor:
        image = PILImage.open(path)
        image_tensor = self.__image_to_tensor(image)
        return image_tensor

    __image_to_tensor = staticmethod(transforms.Compose([transforms.ToTensor()]))
