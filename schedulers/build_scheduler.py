from hydra.utils import instantiate

def build_scheduler(cfg_scheduler, optimizer):
    """
    Inputs : Dictionary cfg_scheduler, Optimizer
    Ouputs : Scheduler
    Function : Builds the learning rate scheduler. 
    """
    if cfg_scheduler is None or cfg_scheduler.get("name") == "none":
        return None
 
    #Hydra instantiates the class, along with the YAML arguments.
    scheduler = instantiate(cfg_scheduler.param, optimizer=optimizer)
    
    return scheduler