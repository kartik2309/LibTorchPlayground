import logging
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List, TypeVar

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
        self.model = model
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
            target, image = batch
            target_ = self.model(image)

            predicted_classes = torch.argmax(target_, dim=-1)

            self.optimizer.zero_grad()
            loss = self.loss(target_, target)
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
            target, image = batch
            target_ = self.model(image)

            predicted_classes = torch.argmax(target_, dim=-1)

            loss = self.loss(target_, target)
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
