from torch import nn
from networks.encoder import Encoder
from networks.decoders import FPNDecoder

class FPNSeg(nn.Module):
    def __init__(self, args):
        super(FPNSeg, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = FPNDecoder(args)
        print(self)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

