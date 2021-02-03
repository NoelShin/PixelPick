import numpy as np
import torch
import torch.nn.functional as F


def prediction(emb, prototypes, non_isotropic=False, return_distance=False):
    # emb: b x n_emb_dims x h x w
    # prototypes: n_classes x n_emb_dims
    b, n_emb_dims, h, w = emb.shape
    n_classes = prototypes.shape[0]

    if non_isotropic:
        emb = emb.unsqueeze(dim=1).repeat((1, n_classes, 1, 1, 1))  # b x n_classes x n_emb_dims x h x w
        prototypes = prototypes.view((1, n_classes, n_emb_dims, 1, 1)).repeat((b, 1, 1, h, w))  # b x n_classes x n_emb_dims x h x w

        dist = (emb - prototypes) ** 2  # .abs()

        return dist.sum(dim=2).argmin(dim=1)

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

        if return_distance:
            return dist.argmin(dim=1), dist
        else:
            return dist.argmin(dim=1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


def batch_pix_accuracy(predict, target):
    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target) * (target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union(predict, target, num_class):
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


def eval_metrics(prediction, target, num_classes, ignore_index):
    target = target.clone()
    target[target == ignore_index] = -1
    correct, labeled = batch_pix_accuracy(prediction, target)
    inter, union = batch_intersection_union(prediction, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    @staticmethod
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Pixel Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))