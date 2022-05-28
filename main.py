import os.path

import torch.optim
from Projects.CIFAR.Python import CIFARTorchDataset, BlockConvModel
from Trainer.Python.trainer import Trainer
from torchvision.models import resnet18


def driver():
    cifar_path_train = (
        "/path/to/CIFAR-10-images/train"
    )
    cifar_path_test = (
        "/path/to/CIFAR-10-images/test"
    )

    model_path = "/path/to/Models/"

    train_dataset = CIFARTorchDataset(cifar_path_train)
    eval_dataset = CIFARTorchDataset(cifar_path_test)

    model = resnet18(pretrained=False)

    adamw = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss = torch.nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=32,
        optimizer=adamw,
        loss=loss,
    )

    trainer.save_optimizer(path=model_path)

    trainer.export_model_to_jit(os.path.join(model_path, "ResNet-18.pt"))


if __name__ == "__main__":
    driver()
