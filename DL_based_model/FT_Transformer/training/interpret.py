import os
from typing import Dict, Any

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from omegaconf import DictConfig
from captum.attr import GradientShap
from models import FTTransformer
from custom_utils.metric import confusion
from custom_utils.utils import Logger
from custom_utils.optimizer_utils import get_parameter_stats


class Eval_Interpret:
    def __init__(
            self, cfg: DictConfig,
            fold: int,
            model: nn.Module,
            dataloaders: Dict[str, Any],
            logger: Logger,
            device: torch.device,
    ) -> None:
        self.device = device
        self.current_step = 0
        self.cfg = cfg
        self.fold = fold
        self.iteration = cfg.dataset.iteration
        self.model = model
        self.train_dataloader = dataloaders['train']
        self.valid_dataloader = dataloaders['valid']
        self.test_dataloader = dataloaders['test']
        self.Categories = dataloaders['categories']
        self.epochs = cfg.training.train_epochs
        self.logger = logger
        self.path = cfg.path
        self.save_path = os.path.join(self.path, f'Iteration_{self.iteration + 1}', f'Fold_{self.fold + 1}')

        os.makedirs(self.save_path, exist_ok=True)

    def get_feature_importance(self, dataloader, criterion):
        self.model.eval()

        # Forward from token (start at token-level representation)
        def f_from_tokens(tok):
            return self.model.forward_from_tokens(tok)

        gs = GradientShap(f_from_tokens)

        total, loss = 0., 0.
        pred, true = [], []

        all_attr_cont = []
        all_attr_cat = []
        all_eid = []

        for batch in tqdm(dataloader, leave=True):
            total += 1
            batch_eid = batch["eid"]
            batch_con = batch["x_con"].to(self.device)  # Float
            batch_y = batch["y"].float().to(self.device)

            with torch.no_grad():
                if self.cfg.dataset.variable_type != 'brain':
                    batch_cat = batch["x_cat"].to(self.device)
                    outputs = self.model(batch_con, batch_cat)
                elif self.cfg.dataset.variable_type == 'brain':
                    outputs = self.model(batch_con, None)

                loss += criterion(outputs, batch_y).item()
                pred.append(outputs.detach().cpu())
                true.append(batch_y.detach().cpu())

            if self.cfg.dataset.variable_type != 'brain':
                tokens = self.model.token_embed(batch_con, batch_cat)  # [B, 1+d_num+num_cat, d_token]
            elif self.cfg.dataset.variable_type == 'brain':
                tokens = self.model.token_embed(batch_con, None)

            # Use the positive class index for binary classification (single logit or two-class logits)
            ig_target = 1

            attr_tok = gs.attribute(
                inputs=tokens,  # token-level features (3D tensor)
                baselines=torch.zeros_like(tokens),
                n_samples=50,  # number of baseline samples
                stdevs=0.1,  # noise scale
                target=ig_target,
            )  # [B, 1 + d_num + num_cat, d_token]

            # Compute token positions
            B, d_num = batch_con.shape  # number of continuous features
            n_tokens = tokens.size(1)  # [B, 1 + d_num + n_cat, d_token]
            n_cat = n_tokens - 1 - d_num

            if self.cfg.dataset.variable_type != 'brain':
                # Slice token attributions excluding the CLS token
                attr_num = attr_tok[:, 1:1 + d_num, :]  # [B, d_num, d_token]
                attr_cat_vec = attr_tok[:, 1 + d_num:1 + d_num + n_cat, :]  # [B, n_cat, d_token]
                all_attr_cont.append(attr_num.detach().cpu())
                all_attr_cat.append(attr_cat_vec.detach().cpu())
                batch_eid = torch.asarray(batch_eid)

                all_eid.append(batch_eid)
            elif self.cfg.dataset.variable_type == 'brain':
                # Slice token attributions excluding the CLS token (continuous features only)
                attr_num = attr_tok[:, 1:1 + d_num, :]  # [B, d_num, d_token]
                all_attr_cont.append(attr_num.detach().cpu())
                batch_eid = torch.asarray(batch_eid)
                all_eid.append(batch_eid)
                all_attr_cat = None

        loss /= total
        final_result = confusion(true, pred)
        return loss, final_result, true, pred, all_attr_cont, all_attr_cat, all_eid

    def interprete(self, dataloader, criterion):
        print('Loading best model...')
        # Get feature importance
        loss, final_result, true, pred, attr_cont, attr_cat, all_eid = self.get_feature_importance(dataloader, criterion)
        # Log results
        self.logger.log_results(iteration=self.iteration + 1, fold=self.fold + 1, results=final_result)

        # Save attributions
        torch.save({
            'eid': all_eid,
            'attr_cont': attr_cont,
            'attr_cat': attr_cat,
            'true': true,
            'pred': pred
        }, os.path.join(self.save_path, 'attributions.pt'))

        return final_result

    def inference(self):
        """
        Run model inference on the test set
        and perform interpretation by extracting feature importance using Captum GradientShap.
        """

        self.current_step = 0

        # Print model parameter statistics
        param_stats = get_parameter_stats(self.model)
        print(f"Model Parameters: {param_stats['total']:,} total, {param_stats['trainable']:,} trainable")

        # Create criterion for evaluation
        criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)

        # Interpret Model
        final_result = self.interprete(self.test_dataloader, criterion)

        return final_result

