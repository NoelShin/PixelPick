import numpy as np
import torch
import torch.nn.functional as F


def get_dict_label_projection(dataloader_query, model, arr_masks, ignore_index,
                              region_contrast=False, device=torch.device("cuda:0")):
    dict_label_projection = dict()

    # collect embeddings and labels
    model.eval()
    with torch.no_grad():
        for batch_ind, dict_data in enumerate(dataloader_query):
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            mask = arr_masks[batch_ind]

            dict_output = model(x)

            projection, logits = dict_output['projection'], dict_output["pred"]

            # l2 norm
            projection = projection / torch.linalg.norm(projection, ord=2, dim=1, keepdims=True)  # b x n_emb x h x w
            projection = torch.transpose(projection, 1, 0)  # n_emb x b x h x w
            projection = projection.contiguous().view(projection.shape[0], -1)  # n_emb x (b x h x w)
            projection = torch.transpose(projection, 1, 0)  # (b x h x w) x n_emb

            masked_projection = projection[mask.flatten()]  # m x n_emb
            masked_labels = y.flatten()[mask.flatten()]  # m

            set_unique_labels = set(masked_labels.cpu().numpy().flatten())
            assert ignore_index not in set_unique_labels
            for ul in set_unique_labels:
                positives = masked_projection[masked_labels == ul].cpu()  # m x n_emb

                try:
                    dict_label_projection[ul].append(positives)
                except KeyError:
                    dict_label_projection.update({ul: [positives]})

                if region_contrast:
                    dict_label_projection[ul].append(positives.mean(dim=0, keepdim=True))

        for k, v in dict_label_projection.items():
            dict_label_projection[k] = torch.cat(v, dim=0)
    return dict_label_projection


def select_samples(projections, n_upper_bound,
                   is_positive=False, mode="random", projection_cur=None, device=torch.device("cuda:0")):
    n_projections = len(projections)
    if n_projections < n_upper_bound:
        return projections

    if mode == "random":
        ind = np.random.choice(range(len(projections)), n_upper_bound, replace=False)
        bool_ = np.zeros_like(range(len(projections)), dtype=np.bool)
        bool_[ind] = True
        samples = projections[torch.tensor(bool_).to(device)]

    elif mode == "hard":
        assert projection_cur is not None, "For mode 'hard' projection_cur should be given."
        projection_cur = projection_cur.unsqueeze(dim=0).repeat(n_projections, 1)
        sim = F.cosine_similarity(projections, projection_cur, dim=1)  # n_projections

        sim_sorted = sim.sort(descending=not is_positive).values
        if 2 * n_upper_bound <= n_projections:
            thres = sim_sorted[2 * n_upper_bound - 1]

        else:
            thres = sim_sorted[n_projections - 1]

        ind = torch.where(sim <= thres)[0] if is_positive else torch.where(sim >= thres)[0]
        assert len(ind) >= n_upper_bound, f"{len(ind)} {n_upper_bound}"

        ind = np.random.choice(ind.cpu().numpy(), n_upper_bound, replace=False)
        bool_ = np.zeros_like(range(len(projections)), dtype=np.bool)
        bool_[ind] = True
        samples = projections[torch.tensor(bool_).to(device)]

    else:
        raise ValueError(f"Invalid mode {mode}")
    return samples


def compute_contrastive_loss(projection, masked_y, mask,
                             dict_label_projection=None,
                             selection_mode="hard",
                             temperature=0.1,
                             device=torch.device("cuda:0")):
    # l2-normalisation
    projection = projection / torch.linalg.norm(projection, ord=2, dim=1, keepdims=True)

    masked_projection = torch.transpose(projection, 1, 0)  # n_emb x b x h x w
    masked_projection = masked_projection.contiguous().view(masked_projection.shape[0], -1)  # n_emb x (b x h x w)
    masked_projection = torch.transpose(masked_projection, 1, 0)  # (b x h x w) x n_emb
    masked_projection = masked_projection[mask.flatten()]  # m x n_emb

    masked_y = masked_y.flatten()[mask.flatten()]  # m

    for i, projection in enumerate(masked_projection):
        label = masked_y[i]
        if dict_label_projection is not None:
            # get the pre-computed positives
            positives = dict_label_projection[label.cpu().item()].to(device)

            # get negatives from the pre-computed positives
            list_negatives = list()
            for l in set(dict_label_projection.keys()) - {label.cpu().item()}:
                list_negatives.append(dict_label_projection[l].to(device))
            negatives = torch.cat(list_negatives, dim=0)

        else:
            positives = masked_projection[masked_y == label]
            negatives = masked_projection[masked_y != label]

        # select among the positives and negatives
        positives = select_samples(positives, n_upper_bound=1024, is_positive=True, projection_cur=projection,
                                   mode=selection_mode)
        negatives = select_samples(negatives, n_upper_bound=2048, is_positive=False, projection_cur=projection,
                                   mode=selection_mode)

        # compute contrastive loss
        negative_term = (projection.unsqueeze(0).repeat(negatives.shape[0], 1) * negatives).sum(dim=1)  # n_negatives
        negative_term = torch.exp(negative_term / temperature).sum()  # n_negatives

        positive_terms = (projection.unsqueeze(0).repeat(positives.shape[0], 1) * positives).sum(dim=1)  # n_positives
        positive_terms = torch.exp(positive_terms / temperature)  # n_positives

        loss_contrastive = (- torch.log(positive_terms / (positive_terms + negative_term))).mean()  # scalar
    return loss_contrastive

    # if dict_label_projection is not None:
    #     for i, projection in enumerate(masked_projection):
    #         label = masked_y[i].cpu().item()
    #         positives = dict_label_projection[label]
    #
    #         list_negatives = list()
    #         for l in set(dict_label_projection.keys()) - {label}:
    #             list_negatives.append(dict_label_projection[l])
    #         negatives = torch.cat(list_negatives, dim=0)
    #
    #         positives = select_samples(positives, n_upper_bound=1024, is_positive=True, projection_cur=projection,
    #                                    mode=selection_mode)
    #         negatives = select_samples(negatives, n_upper_bound=2048, is_positive=False, projection_cur=projection,
    #                                    mode=selection_mode)
    #
    #         negative_term = (projection.unsqueeze(0).repeat(negatives.shape[0], 1) * negatives).sum(dim=1)
    #         negative_term = torch.exp(negative_term / temperature).sum()
    #
    #         positive_terms = (projection.unsqueeze(0).repeat(positives.shape[0], 1) * positives).sum(dim=1)  # m
    #         positive_terms = torch.exp(positive_terms / temperature)  # m
    #
    #         loss = (- torch.log(positive_terms / (positive_terms + negative_term))).mean()
    #
    # else:
    #     for i, projection in enumerate(masked_projection):
    #         label = masked_y[i]
    #         positives = masked_projection[masked_y == label]
    #         negatives = masked_projection[masked_y != label]
    #
    #         negative_term = (projection.unsqueeze(0).repeat(negatives.shape[0], 1) * negatives).sum(dim=1)
    #         negative_term = torch.exp(negative_term / temperature).sum()
    #
    #         positive_terms = (projection.unsqueeze(0).repeat(positives.shape[0], 1) * positives).sum(dim=1)  # m
    #         positive_terms = torch.exp(positive_terms / temperature)  # m
    #
    #         loss = (- torch.log(positive_terms / (positive_terms + negative_term))).mean()