import numpy as np
import cv2
from PIL import Image


class CED:
    def __init__(self, kernel_size, threshold1, threshold2):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def img_to_edge(self, pil_img):
        assert isinstance(pil_img, Image.Image)

        print(self.threshold1, self.threshold2)
        edges = cv2.Canny(np.asarray(pil_img), self.threshold1, self.threshold2)
        dilated_edges = cv2.dilate(edges, self.kernel, iterations=1)
        return dilated_edges

    def __call__(self, pil_img):
        # assert isinstance(np_img, np.ndarray)
        # list_dilated_edges = list()
        # for t in np_img:
        #     img = self._adjust_range(np.transpose(t, (1, 2, 0)))

        edges = cv2.Canny(np.array(pil_img), self.threshold1, self.threshold2)
        dilated_edges = cv2.dilate(edges, self.kernel, iterations=1)
        dilated_edges = Image.fromarray(dilated_edges)
        return dilated_edges
        # return np.expand_dims(np.array(list_dilated_edges), axis=1)

    @staticmethod
    def _adjust_range(np_array, eps=1e-5):
        np_array -= np_array.min()
        np_array = np_array / (np_array.max() + eps)
        np_array *= 255.0
        np_array = np.clip(np_array, 0, 255)
        np_array = np_array.astype(np.uint8)
        return np_array