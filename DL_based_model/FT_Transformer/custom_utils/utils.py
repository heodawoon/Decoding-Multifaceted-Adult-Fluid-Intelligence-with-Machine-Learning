import os
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import Optional, List, Tuple


class Logger:
    def __init__(self, cfg: DictConfig, verbose: bool = True) -> None:
        self.cfg = cfg
        self.unique_id = cfg.unique_id
        self.path = cfg.path
        self.log_filepath = os.path.join(self.path, 'test_result.csv')
        self.verbose = verbose
        self.wandb = self.cfg.wandb

    def init_logging(self) -> None:
        header = 'Iteration,Fold,Phase,Accuracy,Sensitivity,Specificity,AUC,F1,Recall,Precision\n'
        if not os.path.exists(self.log_filepath):
            with open(self.log_filepath, 'w') as f:
                f.write(header)

    def log_results(
            self,
            iteration: Optional[int] = None,
            fold: Optional[int] = None,
            results: Optional[Tuple[float, ...]] = None,
            all_results: Optional[List[Tuple[float, ...]]] = None,
            all_results_val: Optional[List[Tuple[float, ...]]] = None,
            phase: Optional[str] = None,
    ) -> None:
        if (all_results is not None) and (all_results_val is not None):
            mean_result_val = np.mean(all_results_val, axis=0)
            std_result_val = np.std(all_results_val, axis=0)
            mean_result = np.mean(all_results, axis=0)
            std_result = np.std(all_results, axis=0)
            with open(self.log_filepath, 'a') as f:
                if mean_result is not None:
                    f.write(f'Avg.,Valid ACC,Valid SEN,Valid SPC,Valid AUC,Test ACC,Test SEN,Test SPC,Test AUC\n')
                    f.write(f'Avg.,{mean_result_val[0]},{mean_result_val[1]},{mean_result_val[2]},{mean_result_val[3]},'
                            f'{mean_result[0]},{mean_result[1]},{mean_result[2]},{mean_result[3]}\n')
                    f.write(f'STD,{std_result_val[0]},{std_result_val[1]},{std_result_val[2]},{std_result_val[3]},'
                            f'{std_result[0]},{std_result[1]},{std_result[2]},{std_result[3]}\n')
            if self.wandb:
                wandb.init(
                    project=self.cfg.project,
                    group=self.unique_id,
                    name=f"{self.unique_id}:Avg_Results",
                    config=OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True),
                    reinit=True,
                    tags=f"Iter{iteration}:F{self.cfg.dataset.fold}"
                )
                wandb.log({
                    "Avg. Valid ACC": mean_result_val[0],
                    "Avg. Valid SEN": mean_result_val[1],
                    "Avg. Valid SPC": mean_result_val[2],
                    "Avg. Valid AUC": mean_result_val[3],
                    "Avg. Valid F1": mean_result_val[4],
                    "Avg. Valid recall": mean_result_val[5],
                    "Avg. Valid precision": mean_result_val[6],
                    "Avg. ACC": mean_result[0],
                    "Avg. SEN": mean_result[1],
                    "Avg. SPC": mean_result[2],
                    "Avg. AUC": mean_result[3],
                    "Avg. F1": mean_result[4],
                    "Avg. recall": mean_result[5],
                    "Avg. precision": mean_result[6],
                })
            if self.verbose:
                print(f"Avg. Valid ACC: {mean_result_val[0]:.5f}, Avg. Valid AUC: {mean_result_val[3]:.5f} "
                      f"Avg. Valid SEN: {mean_result_val[1]:.5f}, Avg. Valid SPC: {mean_result_val[2]:.5f}\n"
                      f"Avg. ACC: {mean_result[0]:.5f}, Avg. AUC: {mean_result[3]:.5f} "
                      f"Avg. SEN: {mean_result[1]:.5f}, Avg. SPC: {mean_result[2]:.5f} ")

        if (all_results is not None):
            mean_result = np.mean(all_results, axis=0)
            std_result = np.std(all_results, axis=0)
            with open(self.log_filepath, 'a') as f:
                if mean_result is not None:
                    f.write(f'Avg.,Test ACC,Test SEN,Test SPC,Test AUC\n')
                    f.write(f'Avg.,{mean_result[0]},{mean_result[1]},{mean_result[2]},{mean_result[3]}\n')
                    f.write(f'STD,{std_result[0]},{std_result[1]},{std_result[2]},{std_result[3]}\n')
            if self.wandb:
                wandb.init(
                    project=self.cfg.project,
                    group=self.unique_id,
                    name=f"{self.unique_id}:Avg_Results",
                    config=OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True),
                    reinit=True,
                    tags=f"Iter{iteration}:F{self.cfg.dataset.fold}"
                )
                wandb.log({
                    "Avg. ACC": mean_result[0],
                    "Avg. SEN": mean_result[1],
                    "Avg. SPC": mean_result[2],
                    "Avg. AUC": mean_result[3],
                    "Avg. F1": mean_result[4],
                    "Avg. recall": mean_result[5],
                    "Avg. precision": mean_result[6],
                })
            if self.verbose:
                print(f"Avg. ACC: {mean_result[0]:.5f}, Avg. AUC: {mean_result[3]:.5f} "
                      f"Avg. SEN: {mean_result[1]:.5f}, Avg. SPC: {mean_result[2]:.5f} ")

        else:
            with open(self.log_filepath, 'a') as f:
                f.write(f'{iteration},{fold},{phase},{results[0]},{results[1]},{results[2]},{results[3]},'
                        f'{results[4]},{results[5]},{results[6]}\n')
            if self.wandb:
                wandb.log({
                    f"Final {phase} ACC": results[0],
                    f"Final {phase} SEN": results[1],
                    f"Final {phase} SPC": results[2],
                    f"Final {phase} AUC": results[3],
                    f"Final {phase} F1": results[4],
                    f"Final {phase} recall": results[5],
                    f"Final {phase} precision": results[6],
                })
            if self.verbose:
                print(f"Iter_{iteration} | Fold_{fold} | Phase_{phase} | Final ACC: {results[0]:.5f}, Final AUC: {results[3]:.5f} "
                      f"Final SEN: {results[1]:.5f}, Final SPC: {results[2]:.5f} ")


def mixup_data(
    x_cat: Optional[torch.Tensor],
    x_con: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    cat_swap_prob: float = 0.2,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Apply mixup to both continuous and categorical data.

    Continuous features are mixed via convex interpolation (standard mixup).
    Categorical features are mixed using a random swap strategy.

    Args:
        x_cat: Categorical features [B, n_categorical] or None
        x_con: Continuous features [B, n_continuous]
        y: Labels [B, n_classes] (one-hot expected)
        alpha: Mixup strength for Beta(alpha, alpha)
        cat_swap_prob: Swap probability for categorical features

    Returns:
        Mixed categorical, continuous features, and labels.
    """

    assert alpha > 0, "alpha must be > 0"
    assert 0 <= cat_swap_prob <= 1, "cat_swap_prob must be in [0, 1]"
    assert y.dim() == 2, "y must be one-hot encoded [B, C]"

    batch_size = x_con.size(0)

    # Generate mixup parameters
    lam = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(x_con.device)
    lam = lam.view(-1, 1)
    indices = torch.randperm(batch_size, device=x_cat.device)

    # ------------ Mixup continuous variables
    # Mix continuous features (standard mixup)
    x_con_mixed = lam * x_con + (1 - lam) * x_con[indices]

    # ------------ Mixup categorical variables
    # Mix categorical features (simple swap strategy)
    if x_cat is not None:
        x_cat_mixed = x_cat.clone()
        mixup_mask = torch.rand(batch_size, device=x_cat.device) < cat_swap_prob
        x_cat_mixed[mixup_mask] = x_cat[indices][mixup_mask]
    else:
        x_cat_mixed = None

    # Mix labels
    y_mixed = lam * y.float() + (1 - lam) * y[indices].float()

    return x_cat_mixed, x_con_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()