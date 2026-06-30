from hydra.utils import instantiate

def build_optimizer(cfg_optimizer, model):
    """
    Inputs : Configuration dictionary cfg_optimizer, network model
    Outputs : Optimizer
    Function : Build the optimizer using Hydra with _target_.
    """
    #Hydra instantiates the optimizer by passing it the model parameters and configuration hyperparameters.
    optimizer = instantiate(cfg_optimizer.param, params=model.parameters())
    
    return optimizer