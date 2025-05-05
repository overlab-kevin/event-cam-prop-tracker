from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional
from omegaconf import OmegaConf


@dataclass
class DatasetCfg:
    labelled_root: str = "./data"
    time_bin_us: int = 1000

@dataclass
class ModelCfg:
    # output feature channels of the sparse UNet
    out_channels: int = 1
    # base width of the UNet (first conv); kept small for embedded GPUs
    base_channels: int = 4
    # whether to keep BatchNorm layers (handy switch for inference‐only builds)
    use_batch_norm: bool = True

@dataclass
class TrainCfg:
    batch_size: int = 16
    num_workers: int = 8
    epochs: int = 100
    lr: float = 1e-3
    lr_milestones: List[int] = (60, 85)

@dataclass
class Cfg:
    output_dir: str = "./runs/debug"
    dataset: DatasetCfg = DatasetCfg()
    model: ModelCfg = ModelCfg()
    train: TrainCfg = TrainCfg()

    # handy wrappers
    def save(self, path: Union[str, Path]):
        OmegaConf.save(OmegaConf.structured(self), Path(path) / "config.yaml")

    @staticmethod
    def load(path: Optional[Union[str, Path]] = None) -> "Cfg":
        base = OmegaConf.structured(Cfg)
        if path:
            return OmegaConf.merge(base, OmegaConf.load(path))
        return OmegaConf.to_container(base, resolve=True)
