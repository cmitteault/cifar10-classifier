import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from torch.utils.data import DataLoader, random_split, Subset

#Compatibility Hydra Python
import argparse
orig_expand_help = argparse.HelpFormatter._expand_help

def patched_expand_help(self, action):
    if action.help is not None and type(action.help).__name__ == 'LazyCompletionHelp':
        action.help = str(action.help)
    return orig_expand_help(self, action)

argparse.HelpFormatter._expand_help = patched_expand_help

from model.cnn import CNNClassif
from data.dataloader import build_dataloaders
from losses.build_loss import build_loss
from optimizers.build_optimizer import build_optimizer
from schedulers.build_schedulers import build_scheduler
from utils.early_stopping import train_val_classifier

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
   
    #Initialization wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,     
        name=cfg.wandb.run_name,       
        config=config_dict             
    )
   
    #Dataloaders and model instantiation with Hydra
    train_loader, val_loader, test_loader = build_dataloaders(cfg.data, cfg.augmentation)
    model = CNNClassif(**cfg.model.param)

    criterion = build_loss(cfg.loss)
    optimizer = build_optimizer(cfg.optimizer, model)
    scheduler = build_scheduler(cfg.scheduler, optimizer)

    #Training
    train_val_classifier(
        model,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        num_epochs=cfg.training.epochs,
        loss_fn=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=cfg.training.patience,
    )

    wandb.finish()

if __name__ == "__main__":
    main()