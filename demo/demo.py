import sys
sys.path.append("../")
import numpy as np
from matplotlib import pyplot as plt
from HeatMap import heat_map

data = np.load("./ramachandran.npy") # shape = [20000, 2]
print(data.shape)


plt.figure(figsize=(6,6))
plt.scatter(data[:,0], data[:,1], marker='.', s=2, alpha=.3)
plt.savefig("./ra-scatter.png", bbox_inches="tight")

heat_mat = heat_map(data, grid_size=50)

plt.cla()
plt.figure(figsize=(6,6))
plt.imshow(heat_mat)
plt.savefig("./ra-heatmap.png", bbox_inches="tight")



# sin(x^2+y^2)
height, width = 50, 75
apprx_dots_per_grid = 50
origin = np.zeros((2, height, width))
samples = np.array([[0, 0]])
xy = np.zeros((2, height, width))
xy[0] = np.linspace(-width/2, width/2, width).reshape([1,-1])
xy[1] = np.linspace(-height/2, height/2, height).reshape([-1,1])
function_value = np.power(
        np.sin(
            np.pi/height*2 * np.sqrt((np.power(xy, 2).sum(axis=0)))
        ), 2
    ) * apprx_dots_per_grid

origin = xy
for i in range(apprx_dots_per_grid):
    shift = np.power(
        2*np.random.random(size=[2, height, width]) - 1
        , 3
    ) * 1
    shifted = origin.copy()
    shifted[0] += shift[0] * (function_value>0)
    shifted[1] += shift[1] * (function_value>0)
    shifted = shifted.transpose([1, 2, 0])[function_value>0, :]
    samples = np.vstack([samples, shifted])
    function_value -= 1

samples = samples[1:]
print(samples.shape)

plt.figure(figsize=(width/10, height/10))
plt.scatter(samples[:,0], samples[:,1], marker='.', s=2, alpha=.1)
plt.savefig("./sin-scatter.png", bbox_inches="tight")

heat_mat = heat_map(samples, grid_size=50)
plt.cla()
plt.figure(figsize=(width/10, height/10))
plt.imshow(heat_mat)
plt.savefig("./sin-heatmap.png", bbox_inches="tight")

