from torch import nn
from networks.encoder import Encoder
from networks.decoders import FPNDecoder


class FPNSeg(nn.Module):
    def __init__(self, args, load_pretrained=True):
        super(FPNSeg, self).__init__()
        self.encoder = Encoder(args, load_pretrained)
        self.decoder = FPNDecoder(args)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
