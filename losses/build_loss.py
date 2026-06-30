from hydra.utils import instantiate

def build_loss(cfg_loss):
    """
    Inputs : Configuration dictionary cfg_loss
    Outputs : Loss function
    Function : Builds the loss function using Hydra's dynamic instantiation.
    """
    #We directly instantiate the object referenced by _target_ in the YAML with all the arguments 
    loss_fct = instantiate(cfg_loss.param)
    
    return loss_fct