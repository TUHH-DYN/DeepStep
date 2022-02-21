import sys, os
import math
import time
from datetime import datetime
import numpy as np

from modules.process_waveeqn_data import process_waveeqn_data
from modules.helpers import coord_is_in_rot_rect_domain

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
f_ic_low = 0.5
f_ic_high = 1

k_base = 1
k_low = 0.5
k_high = 1.5

##
## Loop Settings
##

counter = 0
num_i = 5 # num base materials
num_j = 2 # num gaussian max
num_k = 5 # type: range(0, num_k)
          # 0 = const material
          # 1 = inhomog. material
          # 2 = inner bc
          # 3 = inhomog. material (rotated)
          # 4 = inner bc (rotated)
if num_k == 5:
    num_k_reps = [2, 2, 2, 2, 2] # num simulations per type
elif num_k ==3:
    num_k_reps = [2, 2, 2] # num simulations per type
else:
    assert 0

assert len(num_k_reps) == num_k
assert num_i%2 == 1
assert num_j%2 == 0

range_i_min = 0
range_i_max = num_i
range_j_min = 0
range_j_max = num_j
range_k_min = 0
range_k_max = num_k

single = False
if single:
    range_i_min = 2
    range_i_max = range_i_min+1
    range_j_min = 0
    range_j_max = range_j_min+1
    range_k_min = 1
    range_k_max = range_k_min+1
    for i in range(num_k):
        num_k_reps[i] = 0
    num_k_reps[range_k_min] = 1

num_total = np.sum(np.array(num_k_reps).dot((range_i_max-range_i_min)*(range_j_max-range_j_min)))

print("Total number of simulations: {}".format(num_total))

for i in range(range_i_min, range_i_max):
    for j in range(range_j_min, range_j_max):
        for k in range(range_k_min, range_k_max):
            for l in range(num_k_reps[k]):

                config = {}

                ##
                ## Temporal and Spatial Settings
                ##

                config['delta_time_fem'] = 1.0/100.0
                config['time_end'] = 8.0
                config['num_steps_fem'] = int(config['time_end']/config['delta_time_fem'])
                config['skip_fem_steps_for_output'] = 1
                config['nx_grid'] = 128
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
                config['output_folder'] = 'datasets_train_waveequation'
                config['filename_base'] = 'dataset_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)


                ##
                ## General Settings
                ##

                tol = 0.001

                # base expressions
                config['no_domain_bc_expression_value'] = f_bc
                config['no_domain_material_expression_value'] = k_low + i*(k_high-k_low)/(num_i-1)


                ##
                ## Root Domain and BC Settings
                ##

                config['root'] = {}

                # default temperature root domain
                config['root']['f'] = f_base

                config['root']['k'] = config['no_domain_material_expression_value']

                config['root']['width'] = (config['nx_grid']-2*config['padding'])*config['dx']
                config['root']['height'] = (config['ny_grid']-2*config['padding'])*config['dx']
                config['root']['center_x'] = (config['nx_grid']/2.0)*config['dx']
                config['root']['center_y'] = (config['ny_grid']/2.0)*config['dx']

                ll = np.array([config['root']['center_x']-config['root']['width']/2.0,config['root']['center_y']-config['root']['height']/2.0])
                ur = np.array([config['root']['center_x']+config['root']['width']/2.0,config['root']['center_y']+config['root']['height']/2.0])
                cl = np.array([config['root']['center_x']-config['root']['width']/2.0,config['root']['center_y']])
                cr = np.array([config['root']['center_x']+config['root']['width']/2.0,config['root']['center_y']])


                # 0=const | 1=linear| 2=dual linear
                # Waveequation -> Neumann BCs -> const -> 0
                config['root']['bc_type'] = 0
                
                # surface eps/ext
                config['root']['surface_eps'] = config['dx']/10.0
                config['root']['surface_ext'] = config['dx']/10.0

                if config['root']['bc_type'] == 0:
                    config['root']['f_0'] = f_bc
                else:
                    assert 0


                ##  
                ## Inclusion Domain and BC Settings
                ##

                hasBCInclusions = False

                config['inclusions'] = []

                inclusionConfig = {}

                # 0=no inclusion | 1=material inclusion | 2=bc inclusion
                if k == 0:
                    inclusionConfig['type'] = 0
                elif k == 1 or k == 3:
                    inclusionConfig['type'] = 1
                elif k == 2 or k == 4:
                    inclusionConfig['type'] = 2

                if inclusionConfig['type'] != 0:

                    # 0=rectangle | 1=ellipse
                    inclusionConfig['shape'] = 0
                    inclusionConfig['shape_name'] = 'RectangleDomain' if inclusionConfig['shape'] == 0 else 'EllipseDomain'

                    # inclusion rotation
                    if k == 3 or k == 4:
                        inclusionConfig['rot_deg'] = np.random.uniform(15, 45)
                    else:
                        inclusionConfig['rot_deg'] = 0

                    # surface eps/ext
                    if inclusionConfig['shape'] == 0:
                        inclusionConfig['surface_eps'] = config['dx']/5.0
                    else:
                        inclusionConfig['surface_eps'] = config['dx']/2.0

                    if inclusionConfig['shape'] == 0:
                        inclusionConfig['surface_ext'] = config['dx']/5.0
                    else:
                        inclusionConfig['surface_ext'] = config['dx']/2.0

                if inclusionConfig['type'] == 1: # material inclusion

                    # material inclusion material parameter
                    randBool = np.random.randint(0,2, dtype=np.uint8, size=1)

                    inclusionConfig['k'] = np.random.uniform(k_low, k_high)
                    
                    if k == 3 or k == 4:
                        inclusionConfig['safety_margin'] = config['root']['width']*0.15
                    else:
                        inclusionConfig['safety_margin'] = config['root']['width']*0.1

                    inclusionConfig['width'] = config['root']['width']*np.random.uniform(0.2, 0.4)
                    inclusionConfig['height'] = config['root']['height']*np.random.uniform(0.2, 0.4)
                    inclusionConfig['center_x'] = np.random.uniform(ll[0]+inclusionConfig['safety_margin']+inclusionConfig['width']/2, ur[0]-inclusionConfig['safety_margin']-inclusionConfig['width']/2)
                    inclusionConfig['center_y'] = np.random.uniform(ll[1]+inclusionConfig['safety_margin']+inclusionConfig['height']/2, ur[1]-inclusionConfig['safety_margin']-inclusionConfig['height']/2)

                    inclusionConfig['width'] = (inclusionConfig['width']//(2*config['dx']))*(2.0*config['dx'])
                    inclusionConfig['height'] = (inclusionConfig['height']//(2.0*config['dx']))*(2.0*config['dx'])
                    inclusionConfig['center_x'] = (inclusionConfig['center_x']//config['dx'])*config['dx'] 
                    inclusionConfig['center_x'] = (inclusionConfig['center_x']//config['dx'])*config['dx']

                elif inclusionConfig['type'] == 2: # bc inclusion

                    hasBCInclusions = True

                    inclusionConfig['f'] = f_bc

                    inclusionConfig['width'] = config['root']['width']*np.random.uniform(0.2, 0.4)
                    inclusionConfig['height'] = config['root']['height']*np.random.uniform(0.2, 0.4)

                    if k == 3 or k == 4:
                        inclusionConfig['safety_margin'] = config['root']['width']*0.15
                    else:
                        inclusionConfig['safety_margin'] = -config['root']['width']*0.1

                    inclusionConfig['center_x'] = np.random.uniform(ll[0]+inclusionConfig['safety_margin']+inclusionConfig['width']/2, ur[0]-inclusionConfig['safety_margin']-inclusionConfig['width']/2)
                    inclusionConfig['center_y'] = np.random.uniform(ll[1]+inclusionConfig['safety_margin']+inclusionConfig['height']/2, ur[1]-inclusionConfig['safety_margin']-inclusionConfig['height']/2)


                    inclusionConfig['width'] = (inclusionConfig['width']//(2*config['dx']))*(2.0*config['dx'])
                    inclusionConfig['height'] = (inclusionConfig['height']//(2.0*config['dx']))*(2.0*config['dx'])
                    inclusionConfig['center_x'] = (inclusionConfig['center_x']//config['dx'])*config['dx'] 
                    inclusionConfig['center_y'] = (inclusionConfig['center_y']//config['dx'])*config['dx']

                config['inclusions'].append(inclusionConfig)

                ##
                ## Initial Condition Settings
                ##

                config['initial_condition'] = {}

                # 0=none (no wave...) | 1=gauss | 2=shape(shouldn't be used for wave eqn.)
                config['initial_condition']['type'] = 1

                if config['initial_condition']['type'] != 0:

                    # dimension for initial condition
                    # gaussian uses "width" information
                    config['initial_condition']['safety_margin'] = config['root']['width']*0.1
                    config['initial_condition']['width'] = config['root']['width']*np.random.uniform(0.05, 0.15)
                    config['initial_condition']['height'] = config['initial_condition']['width']

                    if hasBCInclusions == True:
                        
                        isInAnyBCInclusion = True

                        # here we make sure that the center of the ic
                        # is not placed within an inner bc inclusion
                        while isInAnyBCInclusion:

                            config['initial_condition']['center_x'] = np.random.uniform(ll[0]+config['initial_condition']['safety_margin']+config['initial_condition']['width']/2, ur[0]-config['initial_condition']['safety_margin']-config['initial_condition']['width']/2)
                            config['initial_condition']['center_y'] = np.random.uniform(ll[1]+config['initial_condition']['safety_margin']+config['initial_condition']['height']/2, ur[1]-config['initial_condition']['safety_margin']-config['initial_condition']['height']/2)

                            isInAnyBCInclusion = False
                            for ii in range(len(config['inclusions'])):

                                if config['inclusions'][ii]['type'] == 2:
                                    isInAnyBCInclusion |= coord_is_in_rot_rect_domain(config['initial_condition']['center_x'],
                                                                            config['initial_condition']['center_y'],
                                                                            config['inclusions'][ii]['center_x'] ,
                                                                            config['inclusions'][ii]['center_y'], 
                                                                            config['inclusions'][ii]['width'],
                                                                            config['inclusions'][ii]['height'],
                                                                            math.radians(config['inclusions'][ii]['rot_deg']))

                    else:
                        config['initial_condition']['center_x'] = np.random.uniform(ll[0]+config['initial_condition']['safety_margin']+config['initial_condition']['width']/2, ur[0]-config['initial_condition']['safety_margin']-config['initial_condition']['width']/2)
                        config['initial_condition']['center_y'] = np.random.uniform(ll[1]+config['initial_condition']['safety_margin']+config['initial_condition']['height']/2, ur[1]-config['initial_condition']['safety_margin']-config['initial_condition']['height']/2)

                    config['initial_condition']['width'] = (config['initial_condition']['width']//(2*config['dx']))*(2.0*config['dx'])
                    config['initial_condition']['height'] = (config['initial_condition']['height']//(2.0*config['dx']))*(2.0*config['dx'])
                    config['initial_condition']['center_x'] = (config['initial_condition']['center_x']//config['dx'])*config['dx'] 
                    config['initial_condition']['center_y'] = (config['initial_condition']['center_y']//config['dx'])*config['dx']


                if config['initial_condition']['type'] == 1:

                    # temperatures for initial condition
                    config['initial_condition']['f'] = f_ic_low + j*(f_ic_high-f_ic_low)/(num_j-1)
                    

                if config['initial_condition']['type'] == 2:

                    assert 0

                tic = time.perf_counter()
                success = process_waveeqn_data(abs_path, config)
                toc = time.perf_counter()

                counter += 1

                print("Generated data {} of {} | {:.2f} %".format(counter, num_total, 100.0*(counter/num_total)))

                print(f"Generated data in {toc - tic:0.4f} seconds")

                sys.stdout.flush()



