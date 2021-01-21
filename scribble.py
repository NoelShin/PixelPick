import numpy as np
from tqdm import tqdm
from PIL import Image

np.random.seed(0)

h, w = 360, 480
grid = np.zeros((h, w))
grid.fill(np.inf)
grid_flat = grid.flatten()

grid_loc = list()
for i in range(360 * 480):
    grid_loc.append([i // w, i % w])
grid_loc = np.array(grid_loc)

N = 100
list_ind = np.random.choice(range(len(grid_flat)), N, replace=False)
list_ind_2d = {(ind // w, ind % w) for ind in list_ind}

for ind, (i, j) in tqdm(enumerate(list_ind_2d)):
    dist = ((grid_loc - np.expand_dims(np.array([i, j]), axis=0)) ** 2).sum(axis=1).squeeze()
    grid_flat = np.where(dist < grid_flat, dist, grid_flat)

print(np.exp(- 0.00001 * grid_flat).min())
grid = np.exp(- 0.1 * grid_flat).reshape(h, w) * 255
grid = grid.astype(np.uint8)
Image.fromarray(grid).show()
