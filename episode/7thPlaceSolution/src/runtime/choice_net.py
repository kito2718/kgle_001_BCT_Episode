from torch import nn
class Net(nn.Sequential):

    def __init__(self, d):
        super().__init__(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 1))
