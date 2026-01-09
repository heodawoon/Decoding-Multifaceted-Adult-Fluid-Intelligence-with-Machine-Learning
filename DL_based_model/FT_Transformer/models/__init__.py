import typing as ty
from omegaconf import DictConfig
import torch
from .ft_transformer import FTTransformer


def model_factory(config: DictConfig,
                  device: torch.device,
                  categories: ty.Optional[ty.List[int]]):
    d_numerical = getattr(config.model, 'd_numerical', 64)

    # Get d_numerical from dataset or use default
    default_kwargs = FTTransformer.get_default_kwargs()

    return eval(config.model.name)(
        n_cont_features=d_numerical,
        cat_cardinalities=categories,
        d_out=int(config.model.d_out),
        **default_kwargs
    ).to(device)

