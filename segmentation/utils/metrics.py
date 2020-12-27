import numpy as np
import torch

def prediction(emb, prototypes, non_isotropic=False):
    # emb: b x n_emb_dims x h x w, prototypes: n_classes x n_emb_dims
    b, n_emb_dims, h, w = emb.shape

    if non_isotropic:
        n_classes = prototypes.shape[0]
        emb = emb.unsqueeze(dim=1).repeat((1, n_classes, 1, 1, 1))  # b x n_classes x n_emb_dims x h x w
        prototypes = prototypes.view((1, n_classes, n_emb_dims, 1, 1)).repeat((b, 1, 1, h, w))  # b x n_classes x n_emb_dims x h x w

        dist = (emb - prototypes).abs()
        confidence = torch.exp(-dist)
        confidence_per_class = confidence.mean(dim=2)  # b x n_classes x h x w
        return confidence_per_class.argmax(dim=1)  # b x h x w

    else:
        emb_sq = emb.pow(exponent=2).sum(dim=1, keepdim=True)  # b x 1 x h x w
        emb_sq = emb_sq.transpose(1, 0).contiguous().view(1, -1).transpose(1, 0)  # (b x h x w) x 1

        emb = emb.transpose(1, 0).contiguous().view(n_emb_dims, -1).transpose(1, 0)  # (b * h * w) x n_emb_dims

        prototypes_sq = prototypes.pow(exponent=2).sum(dim=1, keepdim=True)  # n_classes x 1

        dist = emb_sq - 2 * torch.matmul(emb, prototypes.t()) + prototypes_sq.t()  # (b x h x w) x n_classes
        dist = dist.transpose(1, 0).view(-1, b, h, w).transpose(1, 0)  # b x n_classes h x w
        return dist.argmin(dim=1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
