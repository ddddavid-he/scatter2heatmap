import numpy as np
from matplotlib import pyplot as plt
from heatmap import heat_map


data = np.load("./test_data.npy") # shape = [20000, 2]
print(f"Data shape = {data.shape}")


dense_mat = heat_map(data, grid_size=100)

plt.figure(figsize=(6,6))
plt.scatter(data[:,0], data[:,1], marker='.', s=2, alpha=.3)
plt.savefig("./scatter.png", bbox_inches="tight")

plt.cla()
plt.figure(figsize=(6,6))
plt.imshow(dense_mat)
plt.savefig("./heatmap.png", bbox_inches="tight")


