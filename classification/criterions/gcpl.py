import torch
import torch.nn.functional as F

class Criterion:
    def __init__(self, args):
        self.loss_type = args.loss_type
        self.use_pl = args.use_pl
        self.w_pl = args.w_pl

    # distance between prototype and feature
    @staticmethod
    def distance(emb, prototypes):
        # emb: b x 2, prototypes: n_classes x 2
        emb_sq = emb.pow(exponent=2).sum(dim=1, keepdim=True)  # b x 1
        prototypes_sq = prototypes.pow(exponent=2).sum(dim=1, keepdim=True)  # n_classes x 1
        dist = emb_sq - 2 * torch.matmul(emb, prototypes.t()) + prototypes_sq.t()  # b x n_classes
        return dist

    # distance-based cross-entropy loss
    def _dce(self, emb, prototypes, labels, temp=1.0):
        distance = self.distance(emb, prototypes)
        logits = -distance / temp
        return F.cross_entropy(logits, labels)

    # prototype loss
    @staticmethod
    def _pl(emb, prototypes, labels):
        prototypes_class = torch.index_select(prototypes, dim=0, index=labels)
        assert prototypes_class.shape == emb.shape

        return F.mse_loss(emb, prototypes_class)

    def __call__(self, dict_outputs, prototypes, labels):
        dict_loss = dict()

        if self.loss_type == "dce":
            loss_dce = self._dce(dict_outputs["emb"], prototypes, labels)
            dict_loss.update({"dce": loss_dce})

        if self.use_pl:
            loss_pl = self._pl(dict_outputs["emb"], prototypes, labels)
            dict_loss.update({"pl": self.w_pl * loss_pl})

        return dict_loss
