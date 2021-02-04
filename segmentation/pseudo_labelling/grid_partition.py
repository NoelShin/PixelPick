import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def partition(grid, points, efficient=True):
    ind_points = np.where(points)
    print(ind_points)
    if efficient:
        arr_h = np.expand_dims(np.array(range(360), np.int32), axis=1).repeat(480, axis=1)
        arr_w = np.expand_dims(np.array(range(480), np.int32), axis=0).repeat(360, axis=0)

        arr = np.stack((arr_h, arr_w), axis=0)

        list_distance_maps = list()
        for num_p, (i_p, j_p) in enumerate(zip(*ind_points)):
            arr_p = np.empty_like(arr)
            arr_p[0].fill(i_p)
            arr_p[1].fill(j_p)

            distance_map = ((arr_p - arr) ** 2).sum(axis=0)
            list_distance_maps.append(distance_map)

        distance_maps = np.array(list_distance_maps)
        grid = distance_maps.argmin(axis=0).squeeze()

    else:
        h, w = grid.shape
        for i in tqdm(range(h)):
            for j in range(w):
                d = np.inf
                for num_p, (i_p, j_p) in enumerate(zip(*ind_points)):
                    distance = ((i - i_p) ** 2 + (j - j_p) ** 2)
                    if d > distance:
                        d = distance
                        grid[i, j] = num_p
    return grid


if __name__ == '__main__':
    np.random.seed(0)
    N_PIXELS = 100

    grid = np.empty((360, 480), dtype=np.uint8)
    points = np.zeros((360 * 480), dtype=np.bool)
    ind = np.random.choice(range(len(points)), N_PIXELS, replace=False)
    points[ind] = True

    points = points.reshape((360, 480))

    grid = partition(grid, points)

    grid = grid.astype(np.float32)
    grid = grid / grid.max()
    grid *= 255
    grid = grid.astype(np.uint8)

    Image.fromarray(grid).show()
