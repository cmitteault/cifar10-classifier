# CIFAR-10 Classifier from scratch
This project is part of the Advanced Programming course I took at school.

## DESCRIPTION
The goal of this project is to design and implement a minimal, configurable, and expandable training framework in PyTorch that allows the quick testing of different deep learning configurations.
The focus is on software quality, code structuring, separation of responsibilities, and the reproducibility of experiments.

The benchmark task is image classification on the CIFAR-10 dataset.

This framework uses PyTorch while integrating :
- Hydra for dynamic management of YAML configurations and instantiation.
- Weights & Biases (W&B) for tracking and saving training curves.
- Optuna for automatic hyperparameter optimization.

## PROJECT STRUCTURE
The software architecture is modular and easily expandable.

```
cifar10-classifier/
|-- config/                     - Hydra configurations
    `-- augmentation/           - basic.yaml, none.yaml
    `-- loss/                   - cross_entropy.yaml, mse.yaml
    `-- model/                  - cnn.yaml
    `-- optimizer/              - adam.yaml, sgd.yaml
    `-- scheduler/              - cosine.yaml, step_lr.yaml
    `-- config.yaml             - Default configuration
|-- data/                       - dataloader.py
|-- losses/                     - build_loss.py
|-- model/                      - cnn.py, build_model.py
|-- optimizers/                 - build_optimizer.py
|-- schedulers/                 - build_schedulers.py
|-- utils/                      - early_stopping.py
`-- optuna_opti.py              - Optimisation script
`-- train.py                    - Main training script
`-- README.md
```

## DEFAULT CONFIGURATION
The default configuration (defined in the `config/config.yaml` file) runs training :
- in debug mode (`debug_mode: true`)
- on a basic CNN model (2 hidden layers, 16 initial channels).

Training runs for 15 epochs with batches of 64 images, using the Adam optimizer (learning rate of 0.001), the cross-entropy loss function, and the StepLR scheduler (learning rate reduction every 10 steps). 

No data augmentation is applied by default, and early stopping will be triggered if no improvement in accuracy is observed for 10 consecutive epochs.


## USAGE AND CONFIGURATION
With Hydra, all framework settings can be modified directly from the terminal via the command line. The default configuration is defined in `config/config.yaml`.

**WARNING** : By default, the `data.debug_mode` parameter is set to `True` to allow for quick testing on a subset of the data. For actual, full training, you must set it to `False`.   

### 1. Standard Training
- To start training with the default configuration (CNN, Adam, CrossEntropy, StepLR, debug mode enabled) :
`python train.py`

- To start full training on the entire dataset :
`python train.py data.debug_mode=false`

### 2. Changing an Entire Module (Hydra)
You can replace a YAML configuration by specifying the YAML filename (without the .yaml extension).

- Change the optimizer (Adam → SGD) :
`python train.py optimizer=sgd`

- Change the scheduler (StepLR → Cosine) :
`python train.py scheduler=cosine`

- Enable “basic” data augmentation :
`python train.py augmentation=basic`

- Combine multiple changes :
`python train.py optimizer=sgd scheduler=cosine augmentation=basic`

### 3. Modifying Specific Hyperparameters on the Fly
You can modify a specific value in the YAML files using the syntax `group.parameter=value`.

- Modify training parameters (Epochs, Patience, Batch_size) :
`python train.py training.epochs=50 training.patience=15 data.batch_size=128`

- Modify the model architecture (e.g., 3 hidden layers, 32 channels) :
`python train.py model.param.nb_hidden_layers=3 model.param.num_channels1=32`

- Modify the optimizer’s hyperparameters (e.g., learning rate for Adam) :
`python train.py optimizer.param.lr=0.005 optimizer.param.weight_decay=0.001`

- Switch modules AND modify their parameters at the same time :
`python train.py optimizer=sgd optimizer.param.lr=0.05 optimizer.param.momentum=0.95`

### 4. Weights & Biases
To easily identify your runs on the W&B interface, you can rename the experiment on the fly :
`python train.py wandb.run_name=“SGD_Test_LR0.05_BasicAugmentation”`


## AUTOMATIC OPTIMIZATION WITH OPTUNA
To automatically search for the best hyperparameters (the learning rate and weight decay are explored by default), run the dedicated script :
`python optuna_opti.py data.debug_mode=false`

The best model found during the trials will be automatically saved as `best_optuna_model.pt.`