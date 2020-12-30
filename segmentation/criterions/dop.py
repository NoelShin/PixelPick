import torch
import torch.nn.functional as F


class Criterion:
    def __init__(self, args, device):
        self.device = device
        self.ignore_index = args.ignore_index

        self.loss_type = args.loss_type
        self.use_pl = args.use_pl
        self.w_pl = args.w_pl

        self.use_repl = args.use_repl
        self.w_repl = args.w_repl

        self.use_vl = args.use_vl
        self.w_vl = args.w_vl

    @staticmethod
    def distance(emb, prototypes, non_isotropic=False, dict_label_counts=None):
        # emb: b x n_emb_dims x h x w, prototypes: n_classes x n_emb_dims
        # prototypes: n_classes x n_prototypes x n_emb_dims
        b, n_emb_dims, h, w = emb.shape
        n_classes = prototypes.shape[0]

        if non_isotropic:
            emb = emb.unsqueeze(dim=1).repeat((1, n_classes, 1, 1, 1))  # b x n_classes x n_emb_dims x h x w
            prototypes = prototypes.view((1, n_classes, n_emb_dims, 1, 1)).repeat((b, 1, 1, h, w))  # b x n_classes x n_emb_dims x h x w

            dist = (emb - prototypes).abs()  # b x n_classes x n_emb_dims x h x w
            dist = dist.sum(dim=2)  # b x n_classes x h x w
            return dist

        else:
            n_prototypes = prototypes.shape[1]
            emb_sq = emb.pow(exponent=2).sum(dim=1, keepdim=True)  # b x 1 x h x w
            emb_sq = emb_sq.transpose(1, 0).contiguous().view(1, -1).transpose(1, 0)  # (b x h x w) x 1

            prototypes_sq = prototypes.pow(exponent=2).sum(dim=2, keepdim=True)  # n_classes x n_prototypes x 1
            prototypes_sq = prototypes_sq.view(n_classes * n_prototypes, 1)  # (n_classes * n_prototypes) x 1

            emb = emb.transpose(1, 0).contiguous().view(n_emb_dims, -1).transpose(1, 0)  # (b * h * w) x n_emb_dims
            prototypes = prototypes.view(n_classes * n_prototypes, n_emb_dims)  # (n_classes * n_prototypes) x n_emb_dims

            # emb: (b * h * w) x n_emb_dims, prototypes.t(): n_emb_dims x (n_classes x n_prototypes)
            dist = emb_sq - 2 * torch.matmul(emb, prototypes.t()) + prototypes_sq.t()  # (b x h x w) x (n_classes * n_prototypes)
            dist = dist.view(b * h * w, n_classes, n_prototypes).sum(dim=-1)  # (b x h x w) x n_classes
            dist = dist.transpose(1, 0).view(-1, b, h, w).transpose(1, 0)  # b x n_classes h x w
            return dist

    # distance-based cross-entropy loss
    def _dce(self, emb, prototypes, labels, temp=1.0, non_isotropic=False, dict_label_counts=None):
        distance = self.distance(emb, prototypes, non_isotropic=non_isotropic, dict_label_counts=dict_label_counts)  # b x n_classes h x w
        logits = -distance / temp  # b x n_classes h x w

        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)

    # prototype loss
    def _pl(self, dict_label_emb, prototypes):
        # emb: m x n_emb_dims, prototypes: n_classes x n_prototypes x n_emb_dims, labels: b x h x w
        loss = torch.tensor(0., device=self.device)
        for label, emb in dict_label_emb.items():
            # prototype_label = prototypes[label].unsqueeze(dim=0).repeat((emb.shape[0], 1))
            prototype_label = prototypes[label]  # n_prototypes x n_emb_dims
            emb = emb.mean(dim=0).unsqueeze(0).repeat((prototype_label.shape[0], 1))  # n_prototypes x n_emb_dims
            l, _ = F.mse_loss(emb, prototype_label, reduction='none').min(dim=0)  # n_emb_dims
            loss += l.mean()
        return loss

    # variance loss
    def _vl(self, dict_label_emb):
        # emb: m x n_emb_dims, prototypes: n_classes x n_emb_dims, labels: b x h x w
        loss = torch.tensor(0., device=self.device)
        n_classes = 0
        for label, emb in dict_label_emb.items():
            if len(emb) > 1:
                emb = emb.var(dim=0)
                loss += torch.exp(-emb).mean()
                n_classes += 1
        return loss / n_classes

    @staticmethod
    def _repl(prototypes, non_isotropic=False):
        # prototypes: n_classes x n_emb_dims
        # if non_isotropic:
        #     prototypes = prototypes.transpose(1, 0)  # n_classes x n_emb_dims
        #     prototypes_1 = prototypes.unsqueeze(dim=1)  # n_emb_dims x 1 x n_classes
        #     prototypes_2 = prototypes.unsqueeze(dim=2)  # n_emb_dims x n_classes x 1
        #     dist = (prototypes_1 - prototypes_2).abs()  # n_emb_dims x n_classes x n_classes
        #     dist = dist.sum(dim=2)  # n_emb_dims x n_classes
        #     dist = torch.exp(-dist).sum(dim=1)  # n_emb_dims
        #     dist = dist.mean()
        #     return dist
        #
        # else:
        #     prototypes_sq = prototypes.pow(exponent=2).sum(dim=1, keepdim=True)  # n_classes x 1
        #     dist = prototypes_sq - 2 * torch.matmul(prototypes, prototypes.t()) + prototypes_sq.t()  # b x n_classes
        #     return torch.exp(-dist).sum() / 2.
        return torch.exp(-prototypes.var(dim=0)).mean()

    def __call__(self, dict_outputs, prototypes, labels, non_isotropic=False, dict_label_counts=None):
        dict_loss = dict()

        if self.loss_type == "dce":
            loss_dce = self._dce(dict_outputs["emb"], prototypes, labels, non_isotropic=non_isotropic,
                                 dict_label_counts=dict_label_counts)
            dict_loss.update({"dce": loss_dce})

        if self.use_pl:
            loss_pl = self._pl(dict_outputs["dict_label_emb"], prototypes)
            dict_loss.update({"pl": self.w_pl * loss_pl})

        if self.use_vl:
            loss_vl = self._vl(dict_outputs["dict_label_emb"])  # , prototypes.shape[0])
            dict_loss.update({"vl": self.w_vl * loss_vl})

        if self.use_repl:
            loss_repl = self._repl(prototypes, non_isotropic=non_isotropic)
            dict_loss.update({"repl": self.w_repl * loss_repl})
        return dict_loss
