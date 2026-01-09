import gc
import os
from datetime import datetime
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from custom_utils.dataloader import build_dataloaders
from custom_utils.utils import Logger
from models import model_factory
from training import Train


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    training_seed = 42

    # Save seed to config
    with open_dict(cfg):
        cfg.seed = training_seed

    # Setup reproducible training (seeds)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('training_seed: ', cfg.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unique_id = datetime.now().strftime("%m-%d-%H-%M")
    setting = 'FTTF_E{}B{}LR{}LD{}'.format(cfg.training.train_epochs,
                                           cfg.training.batch_size,
                                           cfg.optimizer.lr,
                                           cfg.optimizer.weight_decay)
    path = os.path.join(cfg.save_path, unique_id + '_' + setting)
    with open_dict(cfg):
        cfg.unique_id = unique_id
        cfg.path = path
    os.makedirs(path, exist_ok=True)

    logger = Logger(cfg)
    logger.init_logging()

    all_results = []
    all_results_val = []

    iterations = cfg.n_iterations
    folds = cfg.n_fold
    for iteration in range(iterations):
        for fold in range(folds):
            with open_dict(cfg):
                cfg.dataset.fold = fold + 1
                cfg.dataset.iteration = iteration

            print(f"<<<<<<<<<<<<< Iteration[{iteration + 1}/{cfg.n_iterations}]; Fold[{fold + 1}/{cfg.n_fold}] >>>>>>>>>>>>>")
            if cfg.wandb:
                wandb.init(project=cfg.project,
                           group=unique_id,
                           name=f"{unique_id}:Iter{iteration}:F{cfg.dataset.fold}",
                           config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                           reinit=True,
                           tags=[f"Iter{iteration}", f"Fold{cfg.dataset.fold}"]
                           )

            dataloaders = build_dataloaders(cfg=cfg, fold=fold)
            print(f"Continuous variables: {cfg.model.d_numerical}")
            
            model = model_factory(config=cfg, device=device, categories=dataloaders['categories'])

            train_model = Train(
                cfg=cfg,
                model=model,
                dataloaders=dataloaders,
                fold=fold,
                logger=logger,
                device=device
            )

            final_result, final_result_val = train_model.train()

            cf_save_pth = os.path.join(path, "config.yaml")
            with open(cf_save_pth, "w") as f:
                OmegaConf.save(config=cfg, f=f)

            all_results.append(final_result)
            all_results_val.append(final_result_val)

            if cfg.wandb:
                wandb.finish()

            gc.collect()
            torch.cuda.empty_cache()


    logger.log_results(all_results=all_results, all_results_val=all_results_val)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
