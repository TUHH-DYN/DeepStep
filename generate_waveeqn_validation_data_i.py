import sys, os
import math
import time
from datetime import datetime
import numpy as np

from modules.process_waveeqn_data import process_waveeqn_data

##
## Settings for path
##

path_name = os.path.dirname(sys.argv[0])   
abs_path = os.path.abspath(path_name) 


##
## Settings for random field values
##

f_bc = 0
f_base = f_bc
f_low = -1
f_high = 1

k_base = 1
k_low = 0.5
k_high = 1.5

##
## Init Configuration
##

config = {}



##
## Temporal and Spatial Settings
##

config['delta_time_fem'] = 1.0/100.0
config['time_end'] = 8.0
config['num_steps_fem'] = int(config['time_end']/config['delta_time_fem'])
config['skip_fem_steps_for_output'] = 1
config['nx_grid'] = 256
config['ny_grid'] = 128
config['padding'] = 22 #smallest padding is = config['padding']-1
config['dx'] = 4.0/128.0
config['aabb'] = [[0, (config['nx_grid']-1)*config['dx']],[0, (config['ny_grid']-1)*config['dx']]] #[[x-limits] [y-limits]]
config['num_fem_cells_factor'] = 1.25
config['num_fem_cells'] = int((max(config['nx_grid'], config['ny_grid']) - 2*config['padding'] )/config['num_fem_cells_factor'])
config['cfl_fem'] = k_high*config['delta_time_fem']/config['dx']
config['cfl'] = k_high*config['skip_fem_steps_for_output']*config['delta_time_fem']/config['dx']

##
## File/Path Settings
##

config['output_folder'] = 'datasets_test_waveequation'
config['filename_base'] = 'dataset_validation_i'
   
##
## General Settings
##
config['root'] = {}

# root material parameter (0.5-2.0)
config['root']['k'] = 1.0

# default temperature root domain
config['root']['f'] = f_base

# base expressions
config['no_domain_material_expression_value'] = config['root']['k']
config['no_domain_bc_expression_value'] = f_base

##
## Root Domain and BC Settings
##

# root domain
config['root']['width'] = (config['nx_grid']-2*config['padding'])*config['dx']
config['root']['height'] = (config['ny_grid']-2*config['padding'])*config['dx']
config['root']['center_x'] = (config['nx_grid']/2.0)*config['dx']
config['root']['center_y'] = (config['ny_grid']/2.0)*config['dx']

ll_root_domain = np.array([config['root']['center_x']-config['root']['width']/2.0,config['root']['center_y']-config['root']['height']/2.0])
ur_root_domain = np.array([config['root']['center_x']+config['root']['width']/2.0,config['root']['center_y']+config['root']['height']/2.0])
cl_root_domain = np.array([config['root']['center_x']-config['root']['width']/2.0,config['root']['center_y']])
cr_root_domain = np.array([config['root']['center_x']+config['root']['width']/2.0,config['root']['center_y']])

print("ll_root_domain: {} ur_root_domain: {}".format(ll_root_domain, ur_root_domain))

# 0=const | 1=linear| 2=dual linear
# Waveequation -> Neumann BCs -> const -> 0
config['root']['bc_type'] = 0
 
# surface eps/xt
config['root']['surface_eps'] = config['dx']/5.0
config['root']['surface_ext'] = config['dx']/5.0

if config['root']['bc_type'] == 0:
    config['root']['f_0'] = f_bc
else:
    assert 0


##
## Inclusion Domain and BC Settings
##

config['inclusions'] = []


# left side
# left side
# left side

inclusionConfig = {}

# 0=no inclusion | 1=material inclusion | 2=bc inclusion
inclusionConfig['type'] = 2

# 0=rectangle | 1=ellipse
inclusionConfig['shape'] = 0
inclusionConfig['shape_name'] = 'RectangleDomain' if inclusionConfig['shape'] == 0 else 'EllipseDomain'

# inclusion rotation
inclusionConfig['rot_deg'] = 45

# surface eps/ext
inclusionConfig['surface_eps'] = config['dx']/5.0
inclusionConfig['surface_ext'] = config['dx']/5.0

# constant bc of inclusion domain
inclusionConfig['f'] = f_bc

inclusionConfig['width'] = config['dx']*2*20
inclusionConfig['height'] = config['dx']*2*20
inclusionConfig['center_x'] = config['dx']*(22)
inclusionConfig['center_y'] = config['dx']*(22)

config['inclusions'].append(inclusionConfig)


inclusionConfig = {}

# 0=no inclusion | 1=material inclusion | 2=bc inclusion
inclusionConfig['type'] = 2

# 0=rectangle | 1=ellipse
inclusionConfig['shape'] = 0
inclusionConfig['shape_name'] = 'RectangleDomain' if inclusionConfig['shape'] == 0 else 'EllipseDomain'

# inclusion rotation
inclusionConfig['rot_deg'] = 45

# surface eps/ext
inclusionConfig['surface_eps'] = config['dx']/5.0
inclusionConfig['surface_ext'] = config['dx']/5.0

# constant bc of inclusion domain
inclusionConfig['f'] = f_bc

inclusionConfig['width'] = config['dx']*2*25
inclusionConfig['height'] = config['dx']*2*25
inclusionConfig['center_x'] = config['dx']*(22)
inclusionConfig['center_y'] = config['dx']*(22+35)

config['inclusions'].append(inclusionConfig)


inclusionConfig = {}

# 0=no inclusion | 1=material inclusion | 2=bc inclusion
inclusionConfig['type'] = 2

# 0=rectangle | 1=ellipse
inclusionConfig['shape'] = 0
inclusionConfig['shape_name'] = 'RectangleDomain' if inclusionConfig['shape'] == 0 else 'EllipseDomain'

# inclusion rotation
inclusionConfig['rot_deg'] = 45

# surface eps/ext
inclusionConfig['surface_eps'] = config['dx']/5.0
inclusionConfig['surface_ext'] = config['dx']/5.0

# constant bc of inclusion domain
inclusionConfig['f'] = f_bc

inclusionConfig['width'] = config['dx']*2*30
inclusionConfig['height'] = config['dx']*2*30
inclusionConfig['center_x'] = config['dx']*(22)
inclusionConfig['center_y'] = config['dx']*(22+70)

config['inclusions'].append(inclusionConfig)



# right side
# right side
# right side


inclusionConfig = {}

# 0=no inclusion | 1=material inclusion | 2=bc inclusion
inclusionConfig['type'] = 2

# 0=rectangle | 1=ellipse
inclusionConfig['shape'] = 0
inclusionConfig['shape_name'] = 'RectangleDomain' if inclusionConfig['shape'] == 0 else 'EllipseDomain'

# inclusion rotation
inclusionConfig['rot_deg'] = 30

# surface eps/ext
inclusionConfig['surface_eps'] = config['dx']/5.0
inclusionConfig['surface_ext'] = config['dx']/5.0

# constant bc of inclusion domain
inclusionConfig['f'] = f_bc

inclusionConfig['width'] = config['dx']*2*20
inclusionConfig['height'] = config['dx']*2*20
inclusionConfig['center_x'] = config['dx']*(234)
inclusionConfig['center_y'] = config['dx']*(22)

config['inclusions'].append(inclusionConfig)


inclusionConfig = {}

# 0=no inclusion | 1=material inclusion | 2=bc inclusion
inclusionConfig['type'] = 2

# 0=rectangle | 1=ellipse
inclusionConfig['shape'] = 0
inclusionConfig['shape_name'] = 'RectangleDomain' if inclusionConfig['shape'] == 0 else 'EllipseDomain'

# inclusion rotation
inclusionConfig['rot_deg'] = 30

# surface eps/ext
inclusionConfig['surface_eps'] = config['dx']/5.0
inclusionConfig['surface_ext'] = config['dx']/5.0

# constant bc of inclusion domain
inclusionConfig['f'] = f_bc

inclusionConfig['width'] = config['dx']*2*25
inclusionConfig['height'] = config['dx']*2*25
inclusionConfig['center_x'] = config['dx']*(234)
inclusionConfig['center_y'] = config['dx']*(22+35)

config['inclusions'].append(inclusionConfig)


inclusionConfig = {}

# 0=no inclusion | 1=material inclusion | 2=bc inclusion
inclusionConfig['type'] = 2

# 0=rectangle | 1=ellipse
inclusionConfig['shape'] = 0
inclusionConfig['shape_name'] = 'RectangleDomain' if inclusionConfig['shape'] == 0 else 'EllipseDomain'

# inclusion rotation
inclusionConfig['rot_deg'] = 30

# surface eps/ext
inclusionConfig['surface_eps'] = config['dx']/5.0
inclusionConfig['surface_ext'] = config['dx']/5.0

# constant bc of inclusion domain
inclusionConfig['f'] = f_bc

inclusionConfig['width'] = config['dx']*2*30
inclusionConfig['height'] = config['dx']*2*30
inclusionConfig['center_x'] = config['dx']*(234)
inclusionConfig['center_y'] = config['dx']*(22+70)

config['inclusions'].append(inclusionConfig)




##
## Initial Condition Settings
##

config['initial_condition'] = {}

# 0=none | 1=gauss | 2=shape(shouldn't be used for wave eqn.)
config['initial_condition']['type'] = 1

# initial condition rotation (fenics implementation ??)
config['initial_condition']['rot'] = 0

config['initial_condition']['width'] = config['dx']*2*3
config['initial_condition']['height'] = config['dx']*2*3
config['initial_condition']['center_x'] = config['dx']*(110)
config['initial_condition']['center_y'] = config['dx']*(63)

config['initial_condition']['f'] = f_high

tic = time.perf_counter()
success = process_waveeqn_data(abs_path, config)
toc = time.perf_counter()

print(f"Generated data in {toc - tic:0.4f} seconds")


