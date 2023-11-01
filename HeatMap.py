import numpy as np
import psutil


def heat_map(samples: np.ndarray, grid_size=20) -> np.ndarray:
    """
    samples: array of (x, y) data, in format of np.ndarray([[x,y],..,[x,y]])
    grid_size: int, number of grids in the shorter edge
    return: 
        count_mat: 2-dim ndarray in the following order 
            y
            ^
            |
            |
            o--------> x   
    """

    ymin = samples[:, 1].min()
    ymax = samples[:, 1].max()
    xmin = samples[:, 0].min()
    xmax = samples[:, 0].max()
    
    dy = ymax - ymin
    dx = xmax - xmin
    
    if dy > dx:
        grid_size = [round(dy/dx*grid_size), grid_size]
    else:
        grid_size = [grid_size, round(dx/dy*grid_size)]
    
    
    grid_lt = np.empty(grid_size + [2], dtype=np.uint)
    grid_lt[:,:,1] = np.arange(0, grid_size[0]).reshape((-1, 1)) # y 
    grid_lt[:,:,0] = np.arange(0, grid_size[1]).reshape((1, -1)) # x
    grid_rb = grid_lt.copy() + 1  # right shift and down shift by 1
    mapped_samples = np.array(samples.copy(), dtype=np.float128)
    
    # normalize the samples' coordinations and map them to [0, grid_size]
    mapped_samples[:,0] = (
            mapped_samples[:,0] - xmin
        ) / dx * grid_size[1]
    mapped_samples[:,1] = (
            mapped_samples[:,1] - ymin
        ) / dy * grid_size[0]
    
    
    max_mem_size = 0.9 * psutil.virtual_memory().available # Bytes
    frame_size = grid_size[0] * grid_size[1] * mapped_samples.itemsize * 2 # Bytes
    if frame_size > max_mem_size:
        raise MemoryError(f"grid_size={tuple(grid_size)} is too large, cannot allocate memory for a single frame.")


    if frame_size * samples.shape[0] < max_mem_size:
        sample_mat = np.empty([samples.shape[0], grid_size[0], grid_size[1], 2], dtype=mapped_samples.dtype)
        sample_mat[:,:,:] = mapped_samples.reshape([-1,1,1,2])
        # (x, y) > (x0, y0) --> (x, y) is in the right-below side of (x0, y0)
        count_mat = (sample_mat >= grid_lt) * (sample_mat < grid_rb) # matrix of Bool type
        count_mat = count_mat[:,:,:,0] * count_mat[:,:,:,1]  # find those with both x and y True
        count_mat = count_mat.sum(axis=0) # count each grids
    else:
        print(f"W: Sample size too large, performance will decrease")
        count_mat = np.zeros(grid_size)
        sample_mat = np.empty([grid_size[0], grid_size[1], 2], dtype=mapped_samples.dtype)
        for sample in mapped_samples:
            sample_mat[:, :] = sample
            bool_mat = (sample_mat >= grid_lt) * (sample_mat < grid_rb)
            bool_mat = bool_mat[:,:,0] * bool_mat[:,:,1]
            count_mat += bool_mat
        
        
    # have those grids with smaller y in lower rows
    return count_mat[::-1]  







