import numpy as np


def heat_map(samples: np.ndarray, grid_size=20) -> np.ndarray:
    """
    samples: array of (x, y) data, in format of np.ndarray([[x,y],..,[x,y]])
    grid_lt: coordination of left-top corner of each grid 
    grid_rb: coordination of right-bottom corner of each grid 
    return: 
        count_mat: 2-dim ndarray in the following order 
            y
            ^
            |
            |
            o--------> x   
    """
    grid_lt = np.zeros((grid_size, grid_size, 2), dtype=np.uint16)
    x = np.arange(0, grid_size).reshape((1, -1))
    grid_lt[:,:,0] = x
    grid_lt[:,:,1] = x.T
    grid_rb = grid_lt.copy() + 1  # right shift and down shift by 1
    norm_samples = samples.copy()
    
    # normalize the samples' coordinations and map them to [0, grid_size]
    norm_samples[:,0] = (
            samples[:,0] - samples[:,0].min()
        ) / (
            samples[:,0].max() - samples[:,0].min()
        ) * grid_size
    norm_samples[:,1] = (
            samples[:,1] - samples[:,1].min()
        ) / (
            samples[:,1].max() - samples[:,1].min()
        ) * grid_size

    # create a matrix for each sample, and combine them into an array
    sample_mat = np.zeros([samples.shape[0], grid_size, grid_size, 2])
    for i in range(samples.shape[0]):
        sample_mat[i,:,:] = norm_samples[i]
    # (x, y) > (x0, y0) --> (x, y) is in the right-hand side of (x0, y0)
    count_mat = (sample_mat > grid_lt) * (sample_mat <= grid_rb) # matrix of Bool type
    count_mat = count_mat[:,:,:,0] * count_mat[:,:,:,1]  # find those with both x and y True
    count_mat = count_mat.sum(axis=0) # count each grids
    # make those grids with smaller y in lower rows
    count_mat = count_mat[::-1]  
    return count_mat






