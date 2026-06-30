import torch
import torch.nn as nn

class CNNClassif(nn.Module):

    def __init__(self, nb_hidden_layers, num_channels1=16, num_classes=10):
        super(CNNClassif, self).__init__()
        self.nb_hidden_layers = nb_hidden_layers
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(3, num_channels1, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm2d(num_channels1), nn.MaxPool2d(kernel_size=2))

        out_nb_channels = num_channels1
        self.hidden_layers = nn.ModuleList()

        for i in range(0,nb_hidden_layers):
            layers = nn.Sequential(nn.Conv2d(out_nb_channels, out_nb_channels * 2, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm2d(out_nb_channels * 2), nn.MaxPool2d(kernel_size=2))
            self.hidden_layers.append(layers)
            out_nb_channels = out_nb_channels * 2

        self.linear_layer = nn.LazyLinear(num_classes)

    def forward(self, x):
        y = self.cnn_layer1(x)

        for i in range(0, self.nb_hidden_layers):
            y = self.hidden_layers[i](y)

        y = y.reshape(y.shape[0], -1)
        out = self.linear_layer(y)

        return out