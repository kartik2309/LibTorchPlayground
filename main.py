import torch.optim
from CIFAR.Python import CIFARTorchDataset, BlockConvModel, Trainer


def driver():
    cifar_path_train = (
        "/Users/kartikrajeshwaran/CodeSupport/CPP/Datasets/CIFAR-10-images/train"
    )
    cifar_path_test = (
        "/Users/kartikrajeshwaran/CodeSupport/CPP/Datasets/CIFAR-10-images/test"
    )

    train_dataset = CIFARTorchDataset(cifar_path_train)
    eval_dataset = CIFARTorchDataset(cifar_path_test)

    image_dims = (32, 32)
    channels = [3, 32, 64, 128]
    kernel_sizes_conv = [3, 3, 3]
    strides_conv = [1, 1, 1]
    dilation_conv = [1, 1, 1]

    kernel_sizes_pool = [3, 3, 3]
    dilation_pool = [1, 1, 1]
    strides_pool = [1, 1, 1]
    dropout = 0.5
    num_classes = 10

    model = BlockConvModel(
        image_sizes=image_dims,
        channels=channels,
        kernel_size_conv=kernel_sizes_conv,
        strides_conv=strides_conv,
        dilation_conv=dilation_conv,
        kernel_size_pool=kernel_sizes_pool,
        dilation_pool=dilation_pool,
        strides_pool=strides_pool,
        dropout=dropout,
        num_classes=num_classes,
    )

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

    trainer.fit(2)


if __name__ == "__main__":
    driver()
