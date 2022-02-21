from dolfin import *
from fenics import *
import numpy as np

def process_domain_data(output_path, filename_base, nx_grid, ny_grid, aabb, domain_list, no_domain_bc_expression, no_domain_material_expression):

    x = np.linspace(aabb[0][0], aabb[0][1], nx_grid)
    y = np.linspace(aabb[1][0], aabb[1][1], ny_grid)
    xv, yv = np.meshgrid(x, y)
    dx = x[1]-x[0]
    dy = y[1]-y[0]

    print('dx: '+str(dx))
    print('dy: '+str(dy))

    bc_index = -1*np.ones_like(xv, dtype=int)
    bc_surf_index = -1*np.ones_like(xv, dtype=int)
    material_index = -1*np.ones_like(xv, dtype=int)

    domain_mask = np.zeros_like(xv, dtype=bool)
    domain_surf_mask = np.zeros_like(xv, dtype=bool)
    surface_mask = np.zeros_like(xv, dtype=bool)

    for i in range(nx_grid):
        for j in range(ny_grid):
            for k in range(len(domain_list)):

                tmp_domain_info = domain_list[k].coord_is_in_extended_domain(xv[j,i], yv[j,i], 1.0)
                tmp_domain_surf_info = domain_list[k].coord_is_in_extended_domain(xv[j,i], yv[j,i], -1.0)

                if not domain_list[k].bc_expression is None:
                    domain_mask[j,i] |= not tmp_domain_info
                    domain_surf_mask[j,i] |= not tmp_domain_surf_info

                if tmp_domain_info == False and not domain_list[k].bc_expression is None:
                    bc_index[j, i] = k
                
                if tmp_domain_surf_info == False and not domain_list[k].bc_expression is None:
                    bc_surf_index[j, i] = k
                
                if tmp_domain_info == True and not domain_list[k].material_expression is None:
                    material_index[j, i] = k

    domain = np.zeros_like(xv)
    domain_surf = np.zeros_like(xv)
    surface = np.zeros_like(xv)
    bcs = np.zeros_like(xv)
    bcs_surf = np.zeros_like(xv)
    material = np.zeros_like(xv)

    for i in range(nx_grid):
        for j in range(ny_grid):

            domain[j,i] = float(~domain_mask[j,i])
            domain_surf[j,i] = float(~domain_surf_mask[j,i])
            surface[j,i] = float(surface_mask[j,i])

            if material_index[j, i] >= 0 and material_index[j, i] < len(domain_list) and not domain_list[material_index[j, i]].material_expression is None:
                material[j,i] = domain_list[material_index[j, i] ].eval_material_expression(xv[j,i], yv[j,i])
            else:
                values = np.array([0.0])
                coords = np.array([xv[j,i], yv[j,i]])
                no_domain_material_expression.eval(values,coords)
                material[j,i] = values[0]
            
            if bc_index[j, i] >= 0 and bc_index[j, i] < len(domain_list) and not domain_list[bc_index[j, i]].bc_expression is None:
                bcs[j,i] = domain_list[bc_index[j, i] ].eval_bc_expression(xv[j,i], yv[j,i])
            else:
                values = np.array([0.0])
                coords = np.array([xv[j,i], yv[j,i]])
                no_domain_bc_expression.eval(values,coords)
                bcs[j,i] = values[0]
            
            if bc_surf_index[j, i] >= 0 and bc_surf_index[j, i] < len(domain_list) and not domain_list[bc_surf_index[j, i]].bc_expression is None:
                bcs_surf[j,i] = domain_list[bc_surf_index[j, i] ].eval_bc_expression(xv[j,i], yv[j,i])
            else:
                values = np.array([0.0])
                coords = np.array([xv[j,i], yv[j,i]])
                no_domain_bc_expression.eval(values,coords)
                bcs_surf[j,i] = values[0]
    
    np.save(output_path+'domain_'+filename_base+'.npy', domain)
    np.save(output_path+'domain_surf_'+filename_base+'.npy', domain_surf)
    np.save(output_path+'bcs_'+filename_base+'.npy', bcs)
    np.save(output_path+'bcs_surf_'+filename_base+'.npy', bcs_surf)
    np.save(output_path+'material_'+filename_base+'.npy', material)

    return [domain, domain_surf, surface, bcs, bcs_surf, material]

