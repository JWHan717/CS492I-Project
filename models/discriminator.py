import torch
import torch.nn as nn
import torchvision.transforms as transforms

from functools import reduce
from operator import mul

import models.utils as utils

class Discriminator(nn.Module):
    def __init__(self, lr):
        super(Discriminator, self).__init__()
        nc = 3
        ndf = 64

        #self.init_shape = (filters, shapes[0][0], shapes[0][1])
        #self.preprocess = nn.Sequential(
        #    nn.Linear(z_size, reduce(mul, self.init_shape, 1)),
        #    nn.ReLU(True))

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        self.real_label = 1.
        self.fake_label = 0.

    def forward(self, level):
        return self.main(level)