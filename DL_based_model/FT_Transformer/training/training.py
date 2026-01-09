import os
import pickle
from typing import Dict, Any

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
from omegaconf import DictConfig

from models import FTTransformer
from custom_utils.utils import mixup_data, soft_cross_entropy
from custom_utils.lr_scheduler import LRScheduler
from custom_utils.metric import confusion
from custom_utils.utils import Logger
from custom_utils.optimizer_utils import create_optimizer, get_parameter_stats


class Train:
    def __init__(
            self, cfg: DictConfig,
            fold: int,
            model: nn.Module,
            dataloaders: Dict[str, Any],
            logger: Logger,
            device: torch.device,
    ) -> None:
        self.current_step = 0
        self.device = device
        self.cfg = cfg
        self.fold = fold
        self.iteration = cfg.dataset.iteration
        self.model = model.to(self.device)
        self.train_dataloader = dataloaders['train']
        self.valid_dataloader = dataloaders['valid']
        self.test_dataloader = dataloaders['test']
        self.Categories = dataloaders['categories']
        self.epochs = cfg.training.train_epochs
        self.logger = logger
        self.path = cfg.path
        self.save_path = os.path.join(self.path, f'Iteration_{self.iteration + 1}', f'Fold_{self.fold + 1}')
        os.makedirs(self.save_path, exist_ok=True)

    def train_per_epoch(self, optimizer, criterion, lr_scheduler):
        total_train, train_loss = 0., 0.
        train_pred, train_true = [], []

        self.model.train()
        for batch in tqdm(self.train_dataloader, leave=True):
            total_train += 1
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            batch_eid, batch_con, batch_y = batch["eid"], batch["x_con"].to(self.device), batch["y"].to(self.device)
            batch_cat = batch["x_cat"].to(self.device) if batch["x_cat"] is not None else None

            if self.cfg.training.mixup_data:
                batch_cat, batch_con, batch_y = mixup_data(x_cat=batch_cat, x_con=batch_con, y=batch_y)

            outputs = self.model(batch_con, batch_cat)

            if self.cfg.training.mixup_data:
                loss = soft_cross_entropy(outputs, batch_y)  # batch_y: float soft labels
            else:
                y_idx = batch_y.argmax(dim=1)
                loss = criterion(outputs, y_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pred.append(outputs.detach().cpu())
            train_true.append(batch_y.detach().cpu())

            if self.cfg.wandb:
                wandb.log({"LR": lr_scheduler.lr, "Iter loss": loss.item()})

        train_loss /= total_train
        train_acc, _, _, _, _, _, _ = confusion(train_true, train_pred)
        return train_loss, train_acc

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total, loss = 0., 0.
        pred, true = [], []

        with torch.no_grad():
            for batch in dataloader:
                total += 1

                batch_eid, batch_con, batch_y = batch["eid"], batch["x_con"].to(self.device), batch["y"].to(self.device)
                batch_cat = batch["x_cat"].to(self.device) if batch["x_cat"] is not None else None

                outputs = self.model(batch_con, batch_cat)

                y_idx = batch_y.argmax(dim=1)
                loss += criterion(outputs, y_idx).item()

                pred.append(outputs.detach().cpu())
                true.append(batch_y.detach().cpu())

        loss /= total
        final_result = confusion(true, pred)
        return loss, final_result

    def test(self, dataloader, criterion, phase):
        print('Loading best model...')
        checkpoint = torch.load(os.path.join(self.save_path, 'train_model.pt'), weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        _, final_result = self.evaluate(dataloader, criterion)
        self.logger.log_results(iteration=self.iteration + 1, fold=self.fold + 1, phase=phase, results=final_result)

        return final_result

    def train(self):
        self.current_step = 0

        # Print model parameter statistics
        param_stats = get_parameter_stats(self.model)
        print(f"Model Parameters: {param_stats['total']:,} total, {param_stats['trainable']:,} trainable")
        
        optimizer = create_optimizer(self.model, self.cfg)
        criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        lr_schedulers = LRScheduler(cfg=self.cfg, optimizer_cfg=self.cfg.optimizer)

        best_valid_auc = 0.
        training_process = []
        for epoch in range(self.epochs):
            print(f"Epoch[{epoch + 1}/{self.cfg.training.train_epochs}] ========================")
            train_loss, train_acc = self.train_per_epoch(optimizer, criterion, lr_schedulers)
            valid_loss, valid_result = self.evaluate(self.valid_dataloader, criterion)

            print(f"| Train Loss: {train_loss:.5f} Train ACC: {train_acc:.5f} "
                  f"| Valid Loss: {valid_loss:.5f} Valid ACC: {valid_result[0]:.5f}")

            if self.cfg.wandb:
                wandb.log({
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Valid Loss": valid_loss,
                    "Valid Accuracy": valid_result[0],
                    "Valid AUC": valid_result[3],
                })
            training_process.append({
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Accuracy": train_acc,
                "Valid Loss": valid_loss,
                "Valid Accuracy": valid_result[0],
                "Valid AUC": valid_result[3],
            })

            if best_valid_auc < valid_result[3]:
                best_valid_auc = valid_result[3]
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_schedulers.state_dict() if hasattr(lr_schedulers, 'state_dict') else None,
                    'epoch': epoch,
                    'best_valid_auc': best_valid_auc,
                }, os.path.join(self.save_path, 'train_model.pt'))
                pickle.dump(training_process, open(os.path.join(self.save_path, 'training_process.pkl'), 'wb'))

            # Check if learning rate has reached minimum (cosine annealing completion)
            if lr_schedulers.min_lr_reached:
                print("Learning rate reached minimum - cosine annealing complete")
                break

        best_ckpt = os.path.join(self.save_path, 'train_model.pt')
        if not os.path.exists(best_ckpt):
            torch.save({'model': self.model.state_dict()}, best_ckpt)

        # Test Model
        final_result_val = self.test(self.valid_dataloader, criterion, 'valid')
        final_result = self.test(self.test_dataloader, criterion, 'test')

        return final_result, final_result_val
