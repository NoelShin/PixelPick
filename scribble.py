import torch
import numpy as np
import cv2
from PIL import Image


def get_edges(instance_tensor):
    edge = torch.ByteTensor(instance_tensor.shape).zero_()
    edge[:, :, 1:] = edge[:, :, 1:] | (instance_tensor[:, :, 1:] != instance_tensor[:, :, :-1])
    edge[:, :, :-1] = edge[:, :, :-1] | (instance_tensor[:, :, 1:] != instance_tensor[:, :, :-1])
    edge[:, 1:, :] = edge[:, 1:, :] | (instance_tensor[:, 1:, :] != instance_tensor[:, :-1, :])
    edge[:, :-1, :] = edge[:, :-1, :] | (instance_tensor[:, 1:, :] != instance_tensor[:, :-1, :])

    return edge.float()


annot = Image.open('abcd.png')
t = torch.from_numpy(np.array(annot)).unsqueeze(dim=0)
print(t.shape)
edge = get_edges(t).squeeze().numpy().astype(np.uint8)
edge *= 255
print(edge.shape)

dilated_edges = cv2.dilate(edge, np.ones((5, 5), np.uint8), iterations=1)

Image.fromarray(edge).show()
Image.fromarray(dilated_edges).show()

