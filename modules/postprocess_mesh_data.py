# import json
# import os
# import glob
# import time

from scipy.interpolate import griddata
import numpy as np


def postprocess_mesh_data(output_path, filename_base, mesh, values, domain, bcs, nx_grid, ny_grid, aabb):

    x = np.linspace(aabb[0][0], aabb[0][1], nx_grid)
    y = np.linspace(aabb[1][0], aabb[1][1], ny_grid)
    xv, yv = np.meshgrid(x, y)

    inv_dom = np.invert(np.array(domain, dtype=bool))
    inv_dom = np.array(inv_dom, dtype=float)
    field_grid_bcs = inv_dom*bcs

    # to real grid
    field_grid = np.zeros((values.shape[0], y.shape[0], x.shape[0]))

    for i in range(values.shape[0]):
        # field_grid[i, :, :] = griddata(mesh, values[i, :], (xv, yv), method='nearest')
        field_grid[i, :, :] = griddata(mesh, values[i, :], (xv, yv), method='linear')
        # field_grid[i, :, :] = griddata(mesh, values[i, :], (xv, yv), method='cubic')
        field_grid[i, :, :] = np.nan_to_num(field_grid[i, :, :])*domain + field_grid_bcs

    np.save(output_path+'field_'+filename_base+'.npy', field_grid)

    return [field_grid]

