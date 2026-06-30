import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
import torch
import wandb

# Comptabilité Hydra Python
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

best_global_accuracy = 0.0

def objective(trial, cfg: DictConfig):
    global best_global_accuracy

    # Hyperparamètres à optimiser
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    cfg.optimizer.param.lr = lr
    cfg.optimizer.param.weight_decay = weight_decay

    # Initialisation de wandb
    run_name = f"optuna_T{trial.number}_lr{lr:.4f}_wd{weight_decay:.6f}"
    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True 
    )

    # Instanciation des dataloaders et du modele depuis les yaml
    train_loader, val_loader, test_loader = build_dataloaders(cfg.data, cfg.augmentation)
    model = CNNClassif(**cfg.model.param)
    criterion = build_loss(cfg.loss)
    optimizer = build_optimizer(cfg.optimizer, model)
    scheduler = build_scheduler(cfg.scheduler, optimizer)

    # Entrainement 
    best_model, train_losses, list_acc = train_val_classifier(
        model_tr=model,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        num_epochs=cfg.training.epochs,
        loss_fn=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=cfg.training.patience,
        verbose=False 
    )

    wandb.finish()

    best_accuracy = max(list_acc) if list_acc else 0

    # Sauvegarde du meilleur modele
    if best_accuracy > best_global_accuracy:
        best_global_accuracy = best_accuracy
        torch.save(best_model, 'best_optuna_model.pt')

    return best_accuracy


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("\n Optimisation des hyperparamètres avec Optuna")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, cfg), n_trials=5)

    print(" Fin optimisation ")
    print(f"Meilleur essai  : Trial #{study.best_trial.number}")
    print(f"Meilleur score  : {study.best_value:.2f}%")
    print(f"Meilleurs paramètres : {study.best_params}")
    print("Modèle sauvegardé dans : 'best_optuna_model.pt'\n")

if __name__ == "__main__":
    main()