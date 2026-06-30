import torch
from .cnn import CNNClassif 

def build_model(cfg_model):
    """
    Inputs : Configuration dictionary cfg_model
    Outputs : Neural network model (CNN)
    Function : Build the CNNClassif model based on the configuration.
    """
    name = cfg_model.get("name", "cnn_classif").lower()
    params = cfg_model.get("params", {})

    #We instantiate the model using the YAML parameters
    model = CNNClassif(**params)

    #Initializing nn.LazyLinear
    #Passing an empty tensor (Batch=1, Channels=3, 32x32 for CIFAR-10)
    dummy_input = torch.randn(1, 3, 32, 32)
    model(dummy_input)
    
    return model