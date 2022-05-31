import os.path
import random
from typing import Union, List, Tuple
import math
import torch


class BlockConvModel(torch.nn.Module):
    def __init__(
        self,
        image_sizes: Union[List[int], Tuple[int, int]],
        channels: List[int],
        kernel_size_conv: List[int],
        strides_conv: List[int],
        dilation_conv: List[int],
        kernel_size_pool: List[int],
        dilation_pool: List[int],
        strides_pool: List[int],
        dropout: float,
        num_classes: int,
    ):
        super(BlockConvModel, self).__init__()
        self.num_blocks = len(channels) - 1
        assert len(image_sizes) == 2
        assert (
            self.num_blocks
            == len(kernel_size_conv)
            == len(strides_conv)
            == len(dilation_conv)
            == len(kernel_size_pool)
            == len(dilation_pool)
            == len(strides_pool)
        )

        self.conv_submodules = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                    kernel_size=kernel_size_conv[idx],
                    dilation=dilation_conv[idx],
                    stride=strides_conv[idx],
                )
                for idx in range(self.num_blocks)
            ]
        )

        self.maxpool_submodules = torch.nn.ModuleList(
            [
                torch.nn.MaxPool2d(
                    kernel_size=kernel_size_pool[idx],
                    dilation=dilation_pool[idx],
                    stride=strides_pool[idx],
                )
                for idx in range(self.num_blocks)
            ]
        )

        sizes = self._get_interim(
            image_sizes,
            kernel_size_conv,
            strides_conv,
            dilation_conv,
            kernel_size_pool,
            dilation_pool,
            strides_pool,
        )

        self.relu = torch.nn.ReLU()

        self.flatten_submodule = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.dropout_submodules = torch.nn.Dropout(dropout)
        self.linear_submodule = torch.nn.Linear(
            in_features=sizes[0] * sizes[1] * channels[-1], out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, (conv_submodule, maxpool_submodule) in enumerate(
            zip(self.conv_submodules, self.maxpool_submodules)
        ):
            x = conv_submodule(x)
            x = self.relu(x)
            x = maxpool_submodule(x)

        x = self.flatten_submodule(x)
        x = self.dropout_submodules(x)
        x = self.linear_submodule(x)

        return x

    def save_model(self, path: str) -> None:
        path = self.__handle_path(path)

        model_checkpoint = self.state_dict()
        torch.save(model_checkpoint, path)

    def load_model(self, path: str) -> None:
        path = self.__handle_path(path)

        model_checkpoint = torch.load(path)
        self.load_state_dict(model_checkpoint)

    def export_to_jit(self, path: str, sample: Tuple[torch.Tensor]) -> None:
        path = self.__handle_path(path)
        traced = torch.jit.trace(self, example_inputs=sample)
        torch.jit.save(traced, path)

    def load_from_jit(self, path: str) -> None:
        path = self.__handle_path(path)
        traced = torch.jit.load(path)
        torch.jit.save(traced, path)

    def _get_interim(
        self,
        image_sizes: Union[List[int], Tuple[int, int]],
        kernel_size_conv: List[int],
        strides_conv: List[int],
        dilation_conv: List[int],
        kernel_size_pool: List[int],
        dilation_pool: List[int],
        strides_pool: List[int],
    ) -> Union[List[int], Tuple[int]]:

        if isinstance(image_sizes, tuple):
            image_sizes = list(image_sizes)

        for idx in range(self.num_blocks):
            for idx_ in range(2):
                image_sizes[idx_] = self.__get_interim_helper(
                    image_sizes[idx_],
                    kernel_size=kernel_size_conv[idx],
                    dilation=dilation_conv[idx],
                    stride=strides_conv[idx],
                )

                image_sizes[idx_] = self.__get_interim_helper(
                    image_sizes[idx_],
                    kernel_size=kernel_size_pool[idx],
                    dilation=dilation_pool[idx],
                    stride=strides_pool[idx],
                )

        return image_sizes

    @staticmethod
    def __get_interim_helper(
        curr_size: int, kernel_size: int, dilation: int, stride: int
    ) -> int:
        size = math.floor(((curr_size - dilation * (kernel_size - 1) - 1) / stride) + 1)

        return size

    @staticmethod
    def __handle_path(path: str) -> str:
        if os.path.isdir(path):
            path = path + "BlockConvNet.pth"
        if ".pth" not in path:
            path = path + ".pth"
        return path
