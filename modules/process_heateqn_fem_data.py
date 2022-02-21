from dolfin import *
from fenics import *
from mshr import *

import numpy as np
from datetime import datetime
import sys

from modules.domains import RectangleDomain
from modules.domains import EllipseDomain
from modules.expressions import DualLinearExpression
from modules.expressions import LinearExpression

def process_heateqn_fem_data(output_path,
                                   filename_base,
                                   time_end, 
                                   num_steps, 
                                   skip_fem_steps_for_output,
                                   num_fem_cells, 
                                   domain_list,
                                   no_domain_bc_expression,
                                   no_domain_material_expression):

    # Time stepping
    delta_time = time_end / num_steps # time step size

    # Create mesh and define function space
    width_root_domain = domain_list[0].width
    height_root_domain = domain_list[0].height
    center_x_root_domain = domain_list[0].center_x
    center_y_root_domain = domain_list[0].center_y

    print('width_root_domain: {}'.format(width_root_domain))
    print('height_root_domain: {}'.format(height_root_domain))
    print('center_x_root_domain: {}'.format(center_x_root_domain))
    print('center_y_root_domain: {}'.format(center_y_root_domain))

    domain = Rectangle(Point(center_x_root_domain-width_root_domain/2.0, center_y_root_domain-height_root_domain/2.0), Point(center_x_root_domain+width_root_domain/2.0, center_y_root_domain+height_root_domain/2.0))
    
    sub_domain = None
    
    for k in range(1, len(domain_list)):

        
        width_domain = domain_list[k].width
        height_domain = domain_list[k].height
        center_x_domain = domain_list[k].center_x
        center_y_domain = domain_list[k].center_y
        rot_rad_domain = -domain_list[k].rot_rad

        print('width_domain {}: {}'.format(k, width_domain))
        print('height_domain {}: {}'.format(k, height_domain))
        print('center_x_domain {}: {}'.format(k, center_x_domain))
        print('center_y_domain {}: {}'.format(k, center_y_domain))
        print('rot_rad_domain {}: {}'.format(k, rot_rad_domain))

        if not domain_list[k].bc_expression is None:

            if isinstance(domain_list[k], RectangleDomain):
                dom = Rectangle(Point(center_x_domain-width_domain/2.0, center_y_domain-height_domain/2.0), Point(center_x_domain+width_domain/2.0, center_y_domain+height_domain/2.0))
            if isinstance(domain_list[k], EllipseDomain):
                dom = Ellipse(Point(center_x_domain, center_y_domain), width_domain/2.0, height_domain/2.0)

            dom = CSGRotation(dom, Point(center_x_domain, center_y_domain), rot_rad_domain)   

            domain = domain-dom
        
        if not domain_list[k].material_expression is None:

            if isinstance(domain_list[k], RectangleDomain):
                sub_dom = Rectangle(Point(center_x_domain-width_domain/2.0, center_y_domain-height_domain/2.0), Point(center_x_domain+width_domain/2.0, center_y_domain+height_domain/2.0))
            if isinstance(domain_list[k], EllipseDomain):
                sub_dom = Ellipse(Point(center_x_domain, center_y_domain), width_domain/2.0, height_domain/2.0)

            sub_dom = CSGRotation(sub_dom, Point(center_x_domain, center_y_domain), rot_rad_domain)   

            if sub_domain is None:
                sub_domain = sub_dom
            else:
                sub_domain = sub_domain+sub_dom
        
        if not domain_list[k].ic_expression is None:

            if isinstance(domain_list[k], RectangleDomain):
                sub_dom = Rectangle(Point(center_x_domain-width_domain/2.0, center_y_domain-height_domain/2.0), Point(center_x_domain+width_domain/2.0, center_y_domain+height_domain/2.0))
            if isinstance(domain_list[k], EllipseDomain):
                sub_dom = Ellipse(Point(center_x_domain, center_y_domain), width_domain, height_domain)

            sub_dom = CSGRotation(sub_dom, Point(center_x_domain, center_y_domain), rot_rad_domain)   

            if sub_domain is None:
                sub_domain = sub_dom
            else:
                sub_domain = sub_domain+sub_dom


    if not sub_domain is None:   
        domain.set_subdomain(1, sub_domain)

    mesh = generate_mesh(domain, num_fem_cells) 

    V = FunctionSpace(mesh, 'P', 1)

    # Boundary Conditions
    bc_fcts = []
    bcs = []

    for k in range(len(domain_list)):

        if domain_list[k].bc_expression is None:
            continue

        bc = DirichletBC(V, domain_list[k].bc_expression, domain_list[k].compiled_subdomain)

        bcs.append(bc)

    # Define material parameters (with an inhomog. if len(inhomog) >)
    class K(UserExpression):
        def __init__(self, domain_list, no_domain_material_expression, **kwargs):
            super().__init__(**kwargs)
            self.domain_list = domain_list
            self.no_domain_material_expression = no_domain_material_expression


        def eval(self, value, x):
            domain_index = -1
            for i in range(len(self.domain_list)):
                if not (self.domain_list[i].material_expression is None):
                    tmp_domain_info = self.domain_list[i].coord_is_in_domain(x[0], x[1])
                    tmp_surface_info = self.domain_list[i].coord_is_on_domain_surface(x[0], x[1])
                    if tmp_domain_info or tmp_surface_info:
                        domain_index = i

            if domain_index >= 0 and domain_index < len(self.domain_list):
                value[0] = self.domain_list[domain_index].eval_material_expression(x[0], x[1])
            else:
                vals = np.array([0.0])
                coords = np.array([x[0], x[1]])
                self.no_domain_material_expression.eval(vals,coords)
                value[0] = vals[0]
        
        def value_shape(self):
            return ()

    # Initialize kappa
    kappa_fct = K(domain_list, no_domain_material_expression, degree=0)

    kappa = interpolate(kappa_fct, V)

    # Define initial condition
    class InitialCondition(UserExpression):
        def __init__(self, domain_list, **kwargs):
            super().__init__(**kwargs)
            self.domain_list = domain_list

        def eval(self, value, x):
            domain_index = 0
            for i in range(1, len(self.domain_list)):
                if not (domain_list[i].ic_expression is None):
                    tmp_domain_info = self.domain_list[i].coord_is_in_domain(x[0], x[1])
                    tmp_surface_info = self.domain_list[i].coord_is_on_domain_surface(x[0], x[1])
                    if tmp_domain_info or tmp_surface_info:
                        domain_index = i

            if domain_index >= 0 and domain_index < len(self.domain_list):
                value[0] = self.domain_list[domain_index].eval_ic_expression(x[0], x[1])
            else:
                assert 0
        
        def value_shape(self):
            return ()

    # Define initial value
    ic_expression = InitialCondition(domain_list, degree=1)
    u_0 = ic_expression
    u_n = interpolate(u_0, V)

    print('num_mesh_nodes: {}'.format(len(u_n.vector())))

    # Apply BCs to initial condition
    for i in range(len(bcs)):
        bcs[i].apply(u_n.vector())

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(0)     #No Source Term

    # Heat conductivity
    F = u*v*dx + delta_time*dot(kappa*grad(u), grad(v))*dx - (u_n + delta_time*f)*v*dx
    a, L = lhs(F), rhs(F)

    # set log level
    set_log_level(LogLevel.WARNING)

    # Save mesh
    xyz = V.tabulate_dof_coordinates()
    x = xyz[:, 0]
    y = xyz[:, 1]
    np_mesh = np.column_stack((x, y))
    np.save(output_path+'mesh_'+filename_base+'.npy', np_mesh)

    np_kappa = np.zeros_like(x)
    for i in range(x.shape[0]):
        values = np.array([0.0])
        coords = np.array([x[i], y[i]])
        kappa_fct.eval(values,coords)
        np_kappa[i] = values[0]

    np.save(output_path+'kappa_'+filename_base+'.npy', np_kappa)

    np_values = np.zeros((int(num_steps/skip_fem_steps_for_output), x.shape[0]))

    # Time-stepping
    u = Function(V)
    for n in range(num_steps):

        if n%skip_fem_steps_for_output == 0:
            np_values[int(n/skip_fem_steps_for_output), :] = u_n.vector()
            print('step: ' +str(n) + ' (out)')  
        else:
            print('step: ' +str(n))  

        sys.stdout.flush()

        # Compute solution
        solve(a == L, u, bcs, solver_parameters={'linear_solver':'gmres', 'preconditioner':'ilu'})

        # Update previous solution
        u_n.assign(u)

    np.save(output_path+'steps_'+filename_base+'.npy', np_values)

    return [np_mesh, np_values]

