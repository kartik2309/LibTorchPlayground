from .Dataset import CIFARTorchDataset
from .Models import BlockConvModel
from .Trainer import Trainer
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
