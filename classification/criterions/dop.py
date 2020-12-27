import torch
import torch.nn.functional as F

class Criterion:
    def __init__(self, args):
        self.loss_type = args.loss_type
        self.use_pl = args.use_pl
        self.w_pl = args.w_pl

        self.use_repl = args.use_repl
        self.w_repl = args.w_repl

    # distance between prototype and feature
    @staticmethod
    def distance(emb, prototypes, non_isotropic=False):
        # emb: b x 2, prototypes: n_classes x 2
        if non_isotropic:
            b, n_classes = emb.shape[0], prototypes.shape[0]
            emb = emb.unsqueeze(dim=1).repeat((1, n_classes, 1))  # b x n_classes x 2
            prototypes = prototypes.unsqueeze(dim=0).repeat((b, 1, 1))  # b x n_classes x 2
            dist = (emb - prototypes).abs()  # b x n_classes x 2

        else:
            emb_sq = emb.pow(exponent=2).sum(dim=1, keepdim=True)  # b x 1
            prototypes_sq = prototypes.pow(exponent=2).sum(dim=1, keepdim=True)  # n_classes x 1
            dist = emb_sq - 2 * torch.matmul(emb, prototypes.t()) + prototypes_sq.t()  # b x n_classes
        return dist

    # distance-based cross-entropy loss
    def _dce(self, emb, prototypes, labels, temp=1.0, non_isotropic=False):
        dist = self.distance(emb, prototypes, non_isotropic)
        if non_isotropic:
            dist = -dist.mean(dim=2)
        logits = -dist /temp
        return F.cross_entropy(logits, labels)

    # prototype loss
    @staticmethod
    def _pl(emb, prototypes, labels):
        prototypes_class = torch.index_select(prototypes, dim=0, index=labels)
        assert prototypes_class.shape == emb.shape

        return F.mse_loss(emb, prototypes_class)

    @staticmethod
    def _repl(prototypes):
        # print(prototypes.shape)
        prototypes_sq = prototypes.pow(exponent=2).sum(dim=1, keepdim=True)  # n_classes x 1

        dist = prototypes_sq - 2 * torch.matmul(prototypes, prototypes.t()) + prototypes_sq.t()  # b x n_classes

        # dist_ = 1 / dist
        # ind = torch.arange(0, n_classes)
        # dist_[ind, ind] = 0
        # return dist_.sum() / 2

        return torch.exp(- 10 * dist).sum() / 2.

    def __call__(self, dict_outputs, prototypes, labels, non_isotropic=False):
        dict_loss = dict()

        if self.loss_type == "dce":
            loss_dce = self._dce(dict_outputs["emb"], prototypes, labels, non_isotropic)
            dict_loss.update({"dce": loss_dce})

        if self.use_pl:
            loss_pl = self._pl(dict_outputs["emb"], prototypes, labels)
            dict_loss.update({"pl": self.w_pl * loss_pl})

        if self.use_repl:
            loss_repl = self._repl(prototypes)
            dict_loss.update({"repl": self.w_repl * loss_repl})

        return dict_loss
