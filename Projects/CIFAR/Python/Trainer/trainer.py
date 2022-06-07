import logging
import os.path
import random

import torch
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict, Union, TypeVar

ModelClass = TypeVar("ModelClass")
TorchDatasetClass = TypeVar("TorchDatasetClass")
TorchDataLoaderClass = TypeVar("TorchDataLoaderClass")
OptimizerClass = TypeVar("OptimizerClass")
LossClass = TypeVar("LossClass")


class Trainer:
    def __init__(
            self,
            model: ModelClass,
            train_dataset: TorchDatasetClass,
            eval_dataset: TorchDatasetClass,
            batch_size: int,
            optimizer: OptimizerClass,
            loss: LossClass,
    ):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss = loss

        self.train_dataloader = self.setup_data_loader(
            train_dataset, batch_size=batch_size
        )
        self.eval_dataloader = self.setup_data_loader(eval_dataset, batch_size=1)

    def train_loop(self) -> Tuple[float, float]:
        losses = list()
        accuracies = list()

        self.model.train()
        for batch in self.train_dataloader:
            batch = tuple([x.to(self.device) for x in batch])
            target, image = batch

            logits = self.model(image)
            predicted_classes = torch.argmax(logits, dim=-1)

            self.optimizer.zero_grad()
            loss = self.loss(logits, target)
            accuracy = self.__accuracy(predicted_classes, target)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            accuracies.append(accuracy)

        return self.__get_batch_metrics(losses, accuracies)

    def eval_loop(self) -> Tuple[float, float]:
        losses = list()
        accuracies = list()

        self.model.eval()
        for batch in self.eval_dataloader:
            batch = tuple([x.to(self.device) for x in batch])
            target, image = batch

            logits = self.model(image)
            predicted_classes = torch.argmax(logits, dim=-1)

            loss = self.loss(logits, target)
            accuracy = self.__accuracy(predicted_classes, target)

            losses.append(loss.item())
            accuracies.append(accuracy)

        return self.__get_batch_metrics(losses, accuracies)

    def fit(self, epochs: int) -> None:
        for idx in range(epochs):
            logging.info(f"Epoch:{idx}")
            logging.info("Training Phase")
            train_metrics = self.train_loop()
            logging.info(f"Loss: {train_metrics[0]}, Accuracy: {train_metrics[1]}")
            logging.info("Evaluation Phase")
            eval_metrics = self.eval_loop()
            logging.info(f"Loss: {eval_metrics[0]}, Accuracy: {eval_metrics[1]}\n")

    def save_optimizer(self, path: str) -> None:
        path = self.__handle_path(path)
        checkpoint_opt = self.optimizer.state_dict()
        torch.save(checkpoint_opt, path)

    def load_optimizer(self, path) -> None:
        path = self.__handle_path(path)
        checkpoint_opt = torch.load(path)
        self.optimizer.load_state_dict(checkpoint_opt)

    @torch.no_grad()
    def export_model_to_jit(self, path: str) -> None:
        self.model.eval()
        self.model.cpu()
        traced_model = torch.jit.script(self.model)
        logging.info(traced_model.graph)
        torch.jit.save(traced_model, path)

    def export_model_to_onnx(
            self,
            path: str,
            input_names: List[str],
            output_names: List[str],
            dynamic_axes: Union[Dict[str, Dict[int, str]], Dict[str, List[int]]],
            opset_version=12,
    ) -> None:
        self.model.eval()
        if self.eval_dataloader is not None:
            sample_idx = random.randint(0, self.eval_dataloader.__len__())
        else:
            sample_idx = random.randint(0, self.train_dataloader.__len__())

        sample = self.train_dataloader.dataset[sample_idx]
        sample_inputs = (sample[1].unsqueeze(0),)
        sample_outputs = (sample[0], )
        torch.onnx.export(
            self.model,
            sample_inputs,
            path,
            input_names=input_names,
            example_outputs=sample_outputs,
            opset_version=opset_version,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    @staticmethod
    def setup_data_loader(
            dataset: TorchDatasetClass, batch_size: int
    ) -> TorchDataLoaderClass:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
        return dataloader

    @staticmethod
    def __accuracy(
            predicted_classes: torch.Tensor, true_classes: torch.Tensor
    ) -> float:
        eq_indices = torch.eq(predicted_classes, true_classes)
        return predicted_classes[eq_indices].size(0) / predicted_classes.size(0)

    @staticmethod
    def __get_batch_metrics(
            losses: List[float], accuracies: List[float]
    ) -> Tuple[float, float]:
        avg_loss = sum(losses) / len(losses)
        avg_accuracy = sum(accuracies) / len(accuracies)

        return avg_loss, avg_accuracy

    @staticmethod
    def __handle_path(path: str) -> str:
        if os.path.isdir(path):
            path = path + "Optimizer.pth"
        if ".pth" not in path:
            path = path + ".pth"
        return path
