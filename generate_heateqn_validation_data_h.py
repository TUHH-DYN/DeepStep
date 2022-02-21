import sys, os
import math
import time
from datetime import datetime
import numpy as np

from modules.process_heateqn_data import process_heateqn_data

##
## Settings for random field values
##

# f_low = 0
# f_high = 1


##
## Init Configuration
##

config = {}


##
## Temporal and Spatial Settings
##

config['delta_time_fem'] = 5.0/1000.0
config['time_end'] = 0.6
config['num_steps_fem'] = int(config['time_end']/config['delta_time_fem'])
config['skip_fem_steps_for_output'] = 1
config['nx_grid'] = 256
config['ny_grid'] = 128
config['padding'] = 22 #smallest padding is = config['padding']-1
config['dx'] = 2.0/128.0
config['aabb'] = [[0, (config['nx_grid']-1)*config['dx']],[0, (config['ny_grid']-1)*config['dx']]] #[[x-limits] [y-limits]]
config['num_fem_cells_factor'] = 1.25
config['num_fem_cells'] = int((max(config['nx_grid'], config['ny_grid']) - 2*config['padding'] )/config['num_fem_cells_factor'])

##
## File/Path Settings
##

config['output_folder'] = 'datasets_test_heatequation'
config['filename_base'] = 'dataset_validation_h'


##
## General Settings
##

# base expressions
config['no_domain_material_expression_value'] = 0.75
config['no_domain_bc_expression_value'] = 0.0


##
## Root Domain and BC Settings
##

config['root'] = {}

# root material parameter
config['root']['k'] = config['no_domain_material_expression_value']

# default temperature root domain
config['root']['f'] = 1.0

# root domain
config['root']['width'] = (config['nx_grid']-2*config['padding'])*config['dx']
config['root']['height'] = (config['ny_grid']-2*config['padding'])*config['dx']
config['root']['center_x'] = (config['nx_grid']/2.0)*config['dx']
config['root']['center_y'] = (config['ny_grid']/2.0)*config['dx']

ll_root_domain = np.array([config['root']['center_x']-config['root']['width']/2.0,config['root']['center_y']-config['root']['height']/2.0])
ur_root_domain = np.array([config['root']['center_x']+config['root']['width']/2.0,config['root']['center_y']+config['root']['height']/2.0])
cl_root_domain = np.array([config['root']['center_x']-config['root']['width']/2.0,config['root']['center_y']])
cr_root_domain = np.array([config['root']['center_x']+config['root']['width']/2.0,config['root']['center_y']])


# 0=const | 1=linear| 2=dual linear
config['root']['bc_type'] = 1

# surface eps/ext
config['root']['surface_eps'] = config['dx']/10.0
config['root']['surface_ext'] = config['dx']/10.0


if config['root']['bc_type'] != 0:

    config['root']['bc_dir'] = 0 # 0=horizontal | 1=diagonal

if config['root']['bc_type'] == 0:
    
    assert 0
    config['root']['f_0'] = 0

elif config['root']['bc_type'] == 1:

    config['root']['f_0'] = 0.25
    config['root']['f_1'] = 0.0

elif config['root']['bc_type'] == 2:
    
    assert 0
    config['root']['f_0'] = 0
    config['root']['f_1'] = 1
    config['root']['f_2'] = 0.5
else:
    assert 0



##
## Inclusion Domain and BC Settings
##

config['inclusions'] = []


inclusionConfig = {}

# 0=no inclusion | 1=material inclusion | 2=bc inclusion
inclusionConfig['type'] = 2

# 0=rectangle | 1=ellipse
inclusionConfig['shape'] = 1
inclusionConfig['shape_name'] = 'RectangleDomain' if inclusionConfig['shape'] == 0 else 'EllipseDomain'

# inclusion rotation
inclusionConfig['rot_deg'] = 0

# surface eps
inclusionConfig['surface_eps'] = config['dx']/5.0
inclusionConfig['surface_ext'] = config['dx']/5.0

# temperatures for constant bc of inclusion domain
inclusionConfig['f'] = 0.0

inclusionConfig['width'] = config['dx']*2*12
inclusionConfig['height'] = config['dx']*2*12
inclusionConfig['center_x'] = config['dx']*(128-60)
inclusionConfig['center_y'] = config['dx']*(64-14)

config['inclusions'].append(inclusionConfig)


inclusionConfig = {}

# 0=no inclusion | 1=material inclusion | 2=bc inclusion
inclusionConfig['type'] = 2

# 0=rectangle | 1=ellipse
inclusionConfig['shape'] = 1
inclusionConfig['shape_name'] = 'RectangleDomain' if inclusionConfig['shape'] == 0 else 'EllipseDomain'

# inclusion rotation
inclusionConfig['rot_deg'] = 0

# surface eps
inclusionConfig['surface_eps'] = config['dx']/5.0
inclusionConfig['surface_ext'] = config['dx']/5.0

# temperatures for constant bc of inclusion domain
inclusionConfig['f'] = 0.75

inclusionConfig['width'] = config['dx']*2*18
inclusionConfig['height'] = config['dx']*2*18
inclusionConfig['center_x'] = config['dx']*(128+52)
inclusionConfig['center_y'] = config['dx']*(64+10)

config['inclusions'].append(inclusionConfig)

##
## Initial Condition Settings
##

config['initial_condition'] = {}

# 0=none | 1=gauss | 2=shape
config['initial_condition']['type'] = 0


path_name = os.path.dirname(sys.argv[0])   
abs_path = os.path.abspath(path_name) 

tic = time.perf_counter()
success = process_heateqn_data(abs_path, config)
toc = time.perf_counter()

print(f"Generated data in {toc - tic:0.4f} seconds")


