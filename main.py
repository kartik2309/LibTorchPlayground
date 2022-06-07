from torchvision.models import resnet18
from TorchScriptUtilities.ExportJIT.export_jit import export_model_to_jit


def driver():

    model = resnet18(pretrained=False)
    export_model_to_jit(model, "/content/BlockConvNet/Resnet-18.pt")


if __name__ == "__main__":
    driver()
