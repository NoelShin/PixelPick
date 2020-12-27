import torch
import torch.nn.functional as F

class Criterion:
    def __init__(self, args):
        self.ignore_index = args.ignore_index

        self.loss_type = args.loss_type
        self.use_pl = args.use_pl
        self.w_pl = args.w_pl

        self.use_repl = args.use_repl
        self.w_repl = args.w_repl

    @staticmethod
    def distance(emb, prototypes, non_isotropic=False):
        # emb: b x n_emb_dims x h x w, prototypes: n_classes x n_emb_dims
        b, n_emb_dims, h, w = emb.shape
        if non_isotropic:
            n_classes = prototypes.shape[0]
            emb = emb.unsqueeze(dim=1).repeat((1, n_classes, 1, 1, 1))  # b x n_classes x n_emb_dims x h x w
            prototypes = prototypes.view((1, n_classes, n_emb_dims, 1, 1)).repeat((b, 1, 1, h, w))  # b x n_classes x n_emb_dims x h x w

            dist = (emb - prototypes).abs()
            confidence_per_class = dist.mean(dim=2)  # b x n_classes x h x w
            return confidence_per_class

        else:
            emb_sq = emb.pow(exponent=2).sum(dim=1, keepdim=True)  # b x 1 x h x w
            emb_sq = emb_sq.transpose(1, 0).contiguous().view(1, -1).transpose(1, 0)  # (b x h x w) x 1

            emb = emb.transpose(1, 0).contiguous().view(n_emb_dims, -1).transpose(1, 0)  # (b * h * w) x n_emb_dims

            prototypes_sq = prototypes.pow(exponent=2).sum(dim=1, keepdim=True)  # n_classes x 1

            dist = emb_sq - 2 * torch.matmul(emb, prototypes.t()) + prototypes_sq.t()  # (b x h x w) x n_classes

            dist = dist.transpose(1, 0).view(-1, b, h, w).transpose(1, 0)  # b x n_classes h x w

            return dist

    # distance-based cross-entropy loss
    def _dce(self, emb, prototypes, labels, temp=1.0, non_isotropic=False):
        distance = self.distance(emb, prototypes, non_isotropic=non_isotropic)  # b x n_classes h x w
        logits = -distance / temp

        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)

    # prototype loss
    @staticmethod
    def _pl(dict_label_emb, prototypes):
        # emb: m x n_emb_dims, prototypes: n_classes x n_emb_dims, labels: b x h x w
        loss = 0.
        for label, emb in dict_label_emb.items():
            prototype_label = prototypes[label].unsqueeze(dim=0).repeat((emb.shape[0], 1))
            loss += F.mse_loss(emb, prototype_label)
        return loss

    @staticmethod
    def _repl(prototypes, non_isotropic=False):
        # prototypes: n_classes x n_emb_dims
        if non_isotropic:
            prototypes = prototypes.transpose(1, 0)  # n_classes x n_emb_dims
            prototypes_1 = prototypes.unsqueeze(dim=1)  # n_emb_dims x 1 x n_classes
            prototypes_2 = prototypes.unsqueeze(dim=2)  # n_emb_dims x n_classes x 1
            dist = (prototypes_1 - prototypes_2).abs()  # n_emb_dims x n_classes x n_classes
            dist = dist.sum(dim=2)  # n_emb_dims x n_classes
            dist = torch.exp(-dist).sum(dim=1)  # n_emb_dims
            dist = dist.mean()
            return dist

        else:
            prototypes_sq = prototypes.pow(exponent=2).sum(dim=1, keepdim=True)  # n_classes x 1
            dist = prototypes_sq - 2 * torch.matmul(prototypes, prototypes.t()) + prototypes_sq.t()  # b x n_classes
            return torch.exp(-dist).sum() / 2.

    def __call__(self, dict_outputs, prototypes, labels, non_isotropic=False):
        dict_loss = dict()

        if self.loss_type == "dce":
            loss_dce = self._dce(dict_outputs["emb"], prototypes, labels, non_isotropic=non_isotropic)
            dict_loss.update({"dce": loss_dce})

        if self.use_pl:
            loss_pl = self._pl(dict_outputs["dict_label_emb"], prototypes)
            dict_loss.update({"pl": self.w_pl * loss_pl})

        if self.use_repl:
            loss_repl = self._repl(prototypes, non_isotropic=non_isotropic)
            dict_loss.update({"repl": self.w_repl * loss_repl})
        return dict_loss
