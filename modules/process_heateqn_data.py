from dolfin import *
from fenics import *
from mshr import *

import numpy as np
import json
from datetime import datetime
import time
import os
import os.path

from modules.process_config_data import process_config_data
from modules.process_domain_data import process_domain_data
from modules.process_heateqn_fem_data import process_heateqn_fem_data
from modules.postprocess_mesh_data import postprocess_mesh_data


def process_heateqn_data(abs_path, config):

    print('nx_grid: '+str(config['nx_grid']))
    print('ny_grid: '+str(config['ny_grid']))
    print('dx: '+str(config['dx']))

    print('time_end: '+str(config['time_end']))
    print('num_steps_fem: '+str(config['num_steps_fem']))
    print('skip_fem_steps_for_output: '+str(config['skip_fem_steps_for_output']))

    if os.path.isfile(abs_path+"/data/"+config['output_folder']+"/"+'config_'+config['filename_base']+'.json'):
        with open(abs_path+"/data/"+config['output_folder']+"/"+'config_'+config['filename_base']+'.json') as json_file:
            tmp_config = json.load(json_file)
            if  ("processed" in tmp_config) and (tmp_config['processed'] == True):
                print(config['filename_base'] + " already processed")
                return True
            else:
                print(config['filename_base'] + " exists bot not processed")


    [output_path, domain_list, no_domain_bc_expression, no_domain_material_expression] = process_config_data(abs_path, config['output_folder'], config)

    [domain, domain_surf, surface, bcs, bcs_surf, material] = process_domain_data(output_path, config['filename_base'], config['nx_grid'], config['ny_grid'], config['aabb'], domain_list, no_domain_bc_expression, no_domain_material_expression)
    print(config['filename_base'])

    tic = time.perf_counter()
    [mesh, values] = process_heateqn_fem_data(output_path, config['filename_base'], config['time_end'], config['num_steps_fem'], config['skip_fem_steps_for_output'], config['num_fem_cells'], domain_list, no_domain_bc_expression, no_domain_material_expression)
    toc = time.perf_counter()
    print(f"FEM processing in {toc - tic:0.4f} seconds")
    # print(config['filename_base'])

    tic = time.perf_counter()
    [field] = postprocess_mesh_data(output_path,config['filename_base'], mesh, values, domain, bcs, config['nx_grid'], config['ny_grid'], config['aabb'])
    toc = time.perf_counter()
    print(f"Post processing in {toc - tic:0.4f} seconds")

    with open(output_path+'config_'+config['filename_base']+'.json') as json_file:
        config = json.load(json_file)

    config['processed'] = True
    with open(output_path+'config_'+config['filename_base']+'.json', 'w') as json_file:
        json.dump(config, json_file, sort_keys=True, indent=4)
        print(config['filename_base'] + " processed")

    return True

