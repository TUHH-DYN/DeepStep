import sys, os
import math
import time
from datetime import datetime
import numpy as np

from modules.process_heateqn_data import process_heateqn_data
from modules.helpers import coord_is_in_rot_rect_domain


##
## Settings for path
##

path_name = os.path.dirname(sys.argv[0])   
abs_path = os.path.abspath(path_name) 



##
## Settings for random field values
##

f_low = 0
f_high = 1
f_half = (f_high-f_low)/2

k_low = 0.5
k_high = 1.5
k_num = 5
k_delta = k_high-k_low
k_delta = k_delta/(k_num-1)
k_vals = np.arange(k_low,k_high+k_delta,k_delta)

##
## Loop Settings
##

num_k = 6 # type: range(0, num_k)
        # 0 = const base      + no incl       + root bc according to h (const, lin. horizontal or lin. diagonal)
        # 1 = gaussian base   + no incl       + root bc according to h (const, lin. horizontal or lin. diagonal)
        # 2 = const base      + inner bc      + root bc according to h (const, lin. horizontal or lin. diagonal)
        # 3 = gaussian base   + inner bc      + root bc according to h (const, lin. horizontal or lin. diagonal)
        # 4 = const base      + inner inhomog + root bc according to h (const, lin. horizontal or lin. diagonal)
        # 5 = gaussian base   + inner inhomog + root bc according to h (const, lin. horizontal or lin. diagonal)
num_k_reps = [2,2,2,2,2,2] # num simulations per type
assert len(num_k_reps) == num_k

def get_i_j_a_j_b_for_h(h):

    if h == 0:
        num_i = 5 # num base temperatures
        num_j_a = 5 # num const bc temperatures start
        num_j_b = 1 # num linear bc temperatures end
    else:
        num_i = 3 # num base temperatures
        num_j_a = 2 # num linear bc temperatures start
        num_j_b = 2 # num linear bc temperatures end

    return [num_i, num_j_a, num_j_b]

def get_num_total(h):
    
    [num_i, num_j_a, num_j_b] = get_i_j_a_j_b_for_h(h)

    range_i_min = 0
    range_i_max = num_i
    range_j_a_min = 0
    range_j_a_max = num_j_a
    range_j_b_min = 0
    range_j_b_max = num_j_b

    return np.sum(np.array(num_k_reps).dot((range_i_max-range_i_min)*(range_j_a_max-range_j_a_min)*(range_j_b_max-range_j_b_min)))

num_h = 3   # num base-cases (const, lin. horizontal, lin. diagonal)

counter = 0

num_total = 0
for h in range(0, num_h):
    num_total += get_num_total(h)

print("Total number of simulations for h={}/{}: {}".format(h,num_h,num_total))

for h in range(0, num_h):

    [num_i, num_j_a, num_j_b] = get_i_j_a_j_b_for_h(h)

    range_i_min = 0
    range_i_max = num_i
    range_j_a_min = 0
    range_j_a_max = num_j_a
    range_j_b_min = 0
    range_j_b_max = num_j_b
    range_k_min = 0
    range_k_max = num_k

    single = False
    if single:
        range_i_min = 0
        range_i_max = range_i_min+1
        range_j_a_min = 0
        range_j_a_max = range_j_a_min+1
        range_j_b_min = 0
        range_j_b_max = range_j_b_min+1
        range_k_min = 2
        range_k_max = range_k_min+1
        for i in range(num_k):
            num_k_reps[i] = 0
        num_k_reps[range_k_min] = 1

    for i in range(range_i_min, range_i_max):
        for j_a in range(range_j_a_min, range_j_a_max):
            for j_b in range(range_j_b_min, range_j_b_max):
                for k in range(range_k_min, range_k_max):
                    for l in range(num_k_reps[k]):

                        df_i = (f_high-f_low)/(num_i-1)
                        df_j_a = (f_half-f_low)/(num_j_a) # for h==1 and h==2
                        df_j_b = (f_half-f_high)/(num_j_b)# for h==1 and h==2
                        df_j = (f_high-f_low)/(num_j_a-1) # for h==0


                        config = {}

                        ##
                        ## Temporal and Spatial Settings
                        ##

                        # config['delta_time_fem'] = 1.0/100.0
                        config['delta_time_fem'] = 5.0/1000.0
                        # config['time_end'] = 0.4
                        config['time_end'] = 0.6
                        config['num_steps_fem'] = int(config['time_end']/config['delta_time_fem'])
                        config['skip_fem_steps_for_output'] = 1
                        config['nx_grid'] = 128
                        config['ny_grid'] = 128
                        # config['nx_grid'] = 256
                        # config['ny_grid'] = 256
                        config['padding'] = 22 #smallest padding is = config['padding']-1
                        config['dx'] = 2.0/128.0
                        # config['dx'] = 2.0/256.0
                        config['aabb'] = [[0, (config['nx_grid']-1)*config['dx']],[0, (config['ny_grid']-1)*config['dx']]] #[[x-limits] [y-limits]]
                        config['num_fem_cells_factor'] = 1.25
                        config['num_fem_cells'] = int((max(config['nx_grid'], config['ny_grid']) - 2*config['padding'] )/config['num_fem_cells_factor'])
                        

                        ##
                        ## File/Path Settings
                        ##

                        config['output_folder'] = 'datasets_train_heatequation'
                        config['filename_base'] = 'dataset_'+str(h)+'_'+str(i)+'_'+str(j_a)+'_'+str(j_b)+'_'+str(k)+'_'+str(l)


                        ##
                        ## General Settings
                        ##

                        # base expressions
                        # config['no_domain_material_expression_value'] = 1.0
                        config['no_domain_material_expression_value'] = k_vals[np.random.randint(len(k_vals))]
                        config['no_domain_bc_expression_value'] = 0.0


                        ##
                        ## Root Domain and BC Settings
                        ##

                        config['root'] = {}

                        # root material parameter
                        config['root']['k'] = config['no_domain_material_expression_value']

                        # default temperature root domain
                        config['root']['f'] = f_low + i*df_i

                        # root domain
                        config['root']['width'] = (config['nx_grid']-2*config['padding'])*config['dx']
                        config['root']['height'] = (config['ny_grid']-2*config['padding'])*config['dx']
                        config['root']['center_x'] = (config['nx_grid']/2.0)*config['dx']
                        config['root']['center_y'] = (config['ny_grid']/2.0)*config['dx']
                        
                        ll = np.array([config['root']['center_x']-config['root']['width']/2.0,config['root']['center_y']-config['root']['height']/2.0])
                        ur = np.array([config['root']['center_x']+config['root']['width']/2.0,config['root']['center_y']+config['root']['height']/2.0])
                        cl = np.array([config['root']['center_x']-config['root']['width']/2.0,config['root']['center_y']])
                        cr = np.array([config['root']['center_x']+config['root']['width']/2.0,config['root']['center_y']])


                        # 0=const | 1=linear| 2=dual linear
                        if h == 0:
                            config['root']['bc_type'] = 0
                        else:
                            config['root']['bc_type'] = 1

                        # surface eps/ext
                        config['root']['surface_eps'] = config['dx']/10.0
                        config['root']['surface_ext'] = config['dx']/10.0

                        if config['root']['bc_type'] != 0:
                            if h == 1:
                                config['root']['bc_dir'] = 0 # horizontal
                            elif h == 2:
                                config['root']['bc_dir'] = 1 # diagonal
                            else:
                                assert False

                        if config['root']['bc_type'] == 0:
                            config['root']['f_0'] = 0.5
                        elif config['root']['bc_type'] == 1:
                            config['root']['f_0'] = f_low + j_a*df_j_a
                            config['root']['f_1'] = f_high + j_b*df_j_b
                        elif config['root']['bc_type'] == 2:
                            config['root']['f_0'] = 0
                            config['root']['f_1'] = 1
                            config['root']['f_2'] = 0.5
                        else:
                            assert 0


                        ##
                        ## Inclusion Domain and BC Settings
                        ##

                        hasBCInclusions = False

                        config['inclusions'] = []
                        
                        inclusionConfig = {}

                        # 0=no inclusion | 1=material inclusion | 2=bc inclusion
                        if k==2 or k==3:
                            inclusionConfig['type'] = 2
                        elif k==4 or k==5:
                            inclusionConfig['type'] = 1
                        else:
                            inclusionConfig['type'] = 0

                        if inclusionConfig['type'] != 0:

                            # 0=rectangle | 1=ellipse
                            inclusionConfig['shape'] = 0
                            inclusionConfig['shape_name'] = 'RectangleDomain' if inclusionConfig['shape'] == 0 else 'EllipseDomain'

                            # inclusion rotation
                            if l%2 == 0:
                                inclusionConfig['rot_deg'] = np.random.uniform(15, 45)
                            else:
                                inclusionConfig['rot_deg'] = 0

                            # surface eps/ext
                            if inclusionConfig['shape'] == 0:
                                inclusionConfig['surface_eps'] = config['dx']/5.0
                                inclusionConfig['surface_ext'] = config['dx']/5.0
                            else:
                                inclusionConfig['surface_eps'] = config['dx']/2.0
                                inclusionConfig['surface_ext'] = config['dx']/2.0


                        if inclusionConfig['type'] == 1: # material inclusion

                            # material inclusion material parameter
                            # inclusionConfig['k'] = np.random.uniform(0.5, 1.5)

                            inc_k_vals_copy = k_vals.copy()
                            inc_k_vals_copy = inc_k_vals_copy[inc_k_vals_copy != config['root']['k']]
                            inclusionConfig['k'] = inc_k_vals_copy[np.random.randint(len(inc_k_vals_copy))]

                            if l%2 == 0:
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
                            inclusionConfig['center_y'] = (inclusionConfig['center_y']//config['dx'])*config['dx']


                        elif inclusionConfig['type'] == 2: # bc inclusion

                            hasBCInclusions = True

                            # temperatures for constant bc of inclusion domain
                            inclusionConfig['f'] = np.random.uniform(f_low, f_high)

                            if l%2 == 0:
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
                            inclusionConfig['center_y'] = (inclusionConfig['center_y']//config['dx'])*config['dx']

                        config['inclusions'].append(inclusionConfig)



                        ##
                        ## Initial Condition Settings
                        ##

                        config['initial_condition'] = {}

                        # 0=none | 1=gauss | 2=shape
                        if k==1 or k==3 or k==5:
                            config['initial_condition']['type'] = 1
                        else:
                            config['initial_condition']['type'] = 0


                        if config['initial_condition']['type'] != 0:

                            config['initial_condition']['rot_deg'] = 0                        

                            # dimension for initial condition
                            # gaussian uses "width" information
                            config['initial_condition']['safety_margin'] = config['root']['width']*0.1
                            config['initial_condition']['width'] = config['root']['width']*np.random.uniform(0.05, 0.4)
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
                            config['initial_condition']['f'] = np.random.uniform(f_low-config['root']['f'], f_high-config['root']['f'])

                        if config['initial_condition']['type'] == 2:

                            # temperatures for initial condition
                            config['initial_condition']['f'] = np.random.uniform(f_low, f_high)
                            config['initial_condition']['f'] = f_high


                            # 0=rectangle | 1=ellipse
                            config['initial_condition']['shape'] = np.random.randint(0,2, dtype=np.uint8, size=1)
                            config['initial_condition']['shape'] = config['initial_condition']['shape'][0]
                            config['initial_condition']['shape'] = 0

                            # surface eps
                            if config['inclusion_shape'] == 0:
                                config['initial_condition']['surface_eps'] = config['dx']/5.0
                            else:
                                config['initial_condition']['surface_eps'] = config['dx']/2.0

                            config['initial_condition']['type_name'] = 'RectangleDomain' if config['initial_condition']['shape'] == 0 else 'EllipseDomain'

                        tic = time.perf_counter()
                        success = process_heateqn_data(abs_path, config)
                        toc = time.perf_counter()

                        counter += 1

                        print("Generated data {} of {} | {:.2f} %".format(counter, num_total, 100.0*(counter/num_total)))

                        print(f"Generated data in {toc - tic:0.4f} seconds")

                        sys.stdout.flush()



