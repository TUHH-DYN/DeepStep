from dolfin import *
from fenics import *
from mshr import *

import numpy as np
import configparser
import json
from datetime import datetime
import time
import os
import math

from modules.domains import RectangleDomain
from modules.domains import EllipseDomain
from modules.domains import construct_domain

from modules.expressions import construct_linear_expression
from modules.expressions import construct_dual_linear_expression
from modules.expressions import construct_gaussian_expression
from modules.expressions import LinearExpression
from modules.expressions import DualLinearExpression


def process_config_data(abs_path, output_folder, config):

    ll = np.array([config['root']['center_x']-config['root']['width']/2.0,config['root']['center_y']-config['root']['height']/2.0])
    ur = np.array([config['root']['center_x']+config['root']['width']/2.0,config['root']['center_y']+config['root']['height']/2.0])
    cl = np.array([config['root']['center_x']-config['root']['width']/2.0,config['root']['center_y']])
    cr = np.array([config['root']['center_x']+config['root']['width']/2.0,config['root']['center_y']])

    no_domain_material_expression = Constant(config['no_domain_material_expression_value'])
    no_domain_bc_expression = Constant(config['no_domain_bc_expression_value'])

    # this is the root domain
    if config['root']['bc_type'] == 0:
        
        root_bc_expression = Constant(config['root']['f_0'])

    elif config['root']['bc_type'] == 1:

        if config['root']['bc_dir'] == 1:
            root_bc_expression = construct_linear_expression(config['root']['f_0'], config['root']['f_1'], ll, ur, degree = 1)
        else:
            root_bc_expression = construct_linear_expression(config['root']['f_0'], config['root']['f_1'], cl, cr, degree = 1)
    else:
        if config['root']['bc_dir'] == 1:
            root_bc_expression = construct_dual_linear_expression(config['root']['f_0'], config['root']['f_1'], config['root']['f_2'], ll, ur, degree = 1)
        else:
            root_bc_expression = construct_dual_linear_expression(config['root']['f_0'], config['root']['f_1'], config['root']['f_2'], cl, cr, degree = 1)
        
    # this list hold all domain info
    domain_list = []

    if config['initial_condition']['type'] == 1:
        root_ic_expression = construct_gaussian_expression(B = config['root']['f'],
                                                            A = config['initial_condition']['f'],
                                                            posX = config['initial_condition']['center_x'],
                                                            posY = config['initial_condition']['center_y'],
                                                            sX = config['initial_condition']['width']/2.0,
                                                            sY = config['initial_condition']['width']/2.0,
                                                            degree=0)
    else:
        root_ic_expression = Constant(config['root']['f'])

    domain_list.append(RectangleDomain(True,                            # isInner
                                    config['root']['surface_eps'],      # surfaceEps
                                    config['root']['surface_ext'],      # surface_ext
                                    root_bc_expression,                 # bcExpression
                                    Constant(config['root']['k']),      # materialExpression
                                    root_ic_expression,                 # icExpression
                                    config['root']['center_x'],
                                    config['root']['center_y'],
                                    config['root']['width'],
                                    config['root']['height']))  

    for i in range(len(config['inclusions'])):

        inclusionConfig = config['inclusions'][i]

        if inclusionConfig['type'] == 1: # material inclusion

            # this is an inclusion defining material parameters
            domain_list.append(construct_domain(inclusionConfig['shape_name'],
                                            True,
                                            inclusionConfig['surface_eps'],
                                            inclusionConfig['surface_ext'],
                                            None,
                                            Constant(inclusionConfig['k']),
                                            None,
                                            inclusionConfig['center_x'],
                                            inclusionConfig['center_y'],
                                            inclusionConfig['width'],
                                            inclusionConfig['height'],
                                            math.radians(inclusionConfig['rot_deg'])))      

        elif inclusionConfig['type'] == 2: # bc inclusion

            # this is an inclusion defining a boundary condition
            domain_list.append(construct_domain(inclusionConfig['shape_name'],
                                            False,
                                            inclusionConfig['surface_eps'],
                                            inclusionConfig['surface_ext'],
                                            Constant(inclusionConfig['f']),
                                            None,
                                            None,
                                            inclusionConfig['center_x'],
                                            inclusionConfig['center_y'],
                                            inclusionConfig['width'],
                                            inclusionConfig['height'],
                                            math.radians(inclusionConfig['rot_deg']))) 

    if config['initial_condition']['type'] == 2:

        # this is an inclusion defining a boundary condition
        domain_list.append(construct_domain(config['initial_condition']['type_name'],
                                        True,
                                        config['initial_condition']['surface_eps'],
                                        config['initial_condition']['surface_ext'],
                                        None,
                                        None,
                                        Constant(config['initial_condition']['f']),
                                        config['initial_condition']['center_x'],
                                        config['initial_condition']['center_y'],
                                        config['initial_condition']['width'],
                                        config['initial_condition']['height'],
                                        math.radians(config['initial_condition']['rot_deg']))) 

    config['processed'] = False

    output_path = abs_path+"/data/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, mode=0o755)
    output_path += output_folder+"/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, mode=0o755)

    with open(output_path+'config_'+config['filename_base']+'.json', 'w') as fp:
        json.dump(config, fp, sort_keys=True, indent=4)

    return [output_path, domain_list, no_domain_bc_expression, no_domain_material_expression]
