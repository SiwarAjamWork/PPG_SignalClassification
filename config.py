import dataclasses
from enum import Enum
from typing import Dict, Union

from augmentations import Compose, RandomNoise, RandomShift


class Mode(Enum):
    train = "train"
    eval = "eval"


@dataclasses.dataclass()
class DatasetConfig:
    def __init__(self):
        self.batch_size: int = 116
        self.num_workers: int = 8
        self.path: Dict[Mode, str] = {
            Mode.train: "C:/Users/USER/DataspellProjects/AttentionIsAllYouNeed/MatToCsv/CsvFiles/trainseg1pic.csv",
            Mode.eval: "C:/Users/USER/DataspellProjects/AttentionIsAllYouNeed/MatToCsv/CsvFiles/testseg1pic.csv"

        }
        self.transforms: Dict[Mode, Union[Compose, callable]] = {
            Mode.train: Compose([RandomNoise(0.05, 0.33), RandomShift(10, 0.33)]),
            Mode.eval: self.identity_transform
        }
        
    def identity_transform(self, x):
        return x        




@dataclasses.dataclass()
class ModelConfig:
    num_layers: int = 12
    signal_length: int = 93
    num_classes: int = 3
    input_channels: int = 1
    embed_size: int = 96 #range 128-1024
    num_heads: int = 8
    expansion: int = 16


@dataclasses.dataclass()
class PPGConfig:
   # dataset: DatasetConfig = DatasetConfig()
   # model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    device: Union[int, str] = "cuda"
    lr: float = 2e-4
    num_epochs: int = 40
    validation_frequency: int = 2
