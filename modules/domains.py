import numpy as np
from fenics import *
from dolfin import *
import math

from modules.helpers import rot_pnt_around_pnt

def construct_domain(type_name, is_inner, surface_eps, surface_ext, bc_expression, material_expression, ic_expression, center_x, center_y, width, height, rot_rad=0):

    if type_name == 'RectangleDomain':
        return RectangleDomain(is_inner, surface_eps, surface_ext, bc_expression, material_expression, ic_expression, center_x, center_y, width, height, rot_rad)
    elif type_name == 'EllipseDomain':
        return EllipseDomain(is_inner, surface_eps, surface_ext, bc_expression, material_expression, ic_expression, center_x, center_y, width, height, rot_rad)
    else:
        assert False

class Domain: 
    def __init__(self, is_inner, surface_eps, surface_ext, bc_expression, material_expression, ic_expression): 
        # for a root domain is_inner == True
        # for a bc inclusion is_inner == False
        # for an material inclusion is_inner == None
        # (bc inclusions are substracted from the root domain)
        self.is_inner = is_inner 
        self.surface_eps = surface_eps 
        self.surface_ext = surface_ext 
        self.bc_expression = bc_expression
        self.material_expression = material_expression
        self.ic_expression = ic_expression

    def eval_bc_expression(self, x, y):
        values = np.array([0.0])
        coords = np.array([x,y])
        self.bc_expression.eval(values,coords)
        return values[0]

    def eval_material_expression(self, x, y):
        values = np.array([0.0])
        coords = np.array([x,y])
        self.material_expression.eval(values,coords)
        return values[0]

    def eval_ic_expression(self, x, y):
        values = np.array([0.0])
        # values = 0.0
        coords = np.array([x,y])
        self.ic_expression.eval(values,coords)
        # return values
        return values[0]

class RectangleDomain(Domain):

    def __init__(self, is_inner, surface_eps, surface_ext, bc_expression, material_expression, ic_expression, center_x, center_y, width, height, rot_rad=0):
        super().__init__(is_inner, surface_eps, surface_ext, bc_expression, material_expression, ic_expression)
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.x1 = center_x - width/2
        self.y1 = center_y - height/2
        self.x2 = center_x + width/2
        self.y2 = center_y + height/2
        self.rot_rad = rot_rad
        
        self.compile_subomain()

    def coord_is_in_domain(self, x, y):
        if self.rot_rad == 0:
            p_rot = np.array([x, y])
        else:
            p = np.array([x, y])
            z = np.array([self.center_x, self.center_y])
            p_rot = rot_pnt_around_pnt(p, z, self.rot_rad)

        if (p_rot[0] > self.x1 and p_rot[0] < self.x2 and p_rot[1] > self.y1 and p_rot[1] < self.y2):
            return True if self.is_inner else False
        else:
            return False if self.is_inner else True
    
    def coord_is_in_extended_domain(self, x, y, ext_factor=1.0):
        ext = self.surface_ext*ext_factor
        if self.is_inner == False:
            ext = -ext
        if self.rot_rad == 0:
            p_rot = np.array([x, y])
        else:
            p = np.array([x, y])
            z = np.array([self.center_x, self.center_y])
            p_rot = rot_pnt_around_pnt(p, z, self.rot_rad)

        if (p_rot[0] > (self.x1-ext) and p_rot[0] < (self.x2+ext) and p_rot[1] > (self.y1-ext) and p_rot[1] < (self.y2+ext)):
            return True if self.is_inner else False
        else:
            return False if self.is_inner else True
    
    def coord_is_on_domain_surface(self, x, y):
        if self.rot_rad == 0:
            p_rot = np.array([x, y])
        else:
            p = np.array([x, y])
            z = np.array([self.center_x, self.center_y])
            p_rot = rot_pnt_around_pnt(p, z, self.rot_rad)

        if (p_rot[0] > self.x1-self.surface_eps and p_rot[0] < self.x2+self.surface_eps and (abs(p_rot[1]-self.y1) < self.surface_eps or abs(p_rot[1]-self.y2) < self.surface_eps)) or (p_rot[1] > self.y1-self.surface_eps and p_rot[1] < self.y2+self.surface_eps and (abs(p_rot[0]-self.x1) < self.surface_eps or abs(p_rot[0]-self.x2) < self.surface_eps)):
            return True
        else:
            return False
    
    def compile_subomain(self):

        cppcode = """
        #include <pybind11/pybind11.h>
        #include <pybind11/eigen.h>
        #include <math.h>
        #include <iostream>
        namespace py = pybind11;

        #include <dolfin/mesh/SubDomain.h>

        class RectangleSubDomain : public dolfin::SubDomain
        {
        public:

            RectangleSubDomain() : dolfin::SubDomain() {}

            bool inside(Eigen::Ref<const Eigen::VectorXd> x, bool on_boundary) const override
            {
                
                double px = x[0];
                double py = x[1];

                double qx = px-cx;
                double qy = py-cy;

                double cosr = cos(rot_rad);
                double sinr = sin(rot_rad);

                double qx_rot = cosr*qx+(-sinr)*qy;
                double qy_rot = sinr*qx+cosr*qy;

                px = cx + qx_rot;
                py = cy + qy_rot;
                
                // std::cout << "p: " << px << ", " << py << std::endl;

                if (on_boundary &&
                    ((px > x1-surface_eps && px < x2+surface_eps && (abs(py-y1) < surface_eps || abs(py-y2) < surface_eps)) || 
                     (py > y1-surface_eps && py < y2+surface_eps && (abs(px-x1) < surface_eps || abs(px-x2) < surface_eps))))
                    return true;
                else
                    return false;

                //if (on_boundary &&
                //    ((x[0] > x1-surface_eps && x[0] < x2+surface_eps && (abs(x[1]-y1) < surface_eps || abs(x[1]-y2) < surface_eps)) || 
                //     (x[1] > y1-surface_eps && x[1] < y2+surface_eps && (abs(x[0]-x1) < surface_eps || abs(x[0]-x2) < surface_eps))))
                //    return true;
                //else
                //    return false;
            }

            double x1;
            double x2;
            double y1;
            double y2;
            double cx;
            double cy;
            double rot_rad;
            double surface_eps;
        };

        PYBIND11_MODULE(SIGNATURE, m)
        {
            py::class_<RectangleSubDomain, std::shared_ptr<RectangleSubDomain>, dolfin::SubDomain>
                (m, "RectangleSubDomain")
                .def(py::init<>())
                .def("inside", &RectangleSubDomain::inside)
                .def_readwrite("x1", &RectangleSubDomain::x1)
                .def_readwrite("x2", &RectangleSubDomain::x2)
                .def_readwrite("y1", &RectangleSubDomain::y1)
                .def_readwrite("y2", &RectangleSubDomain::y2)
                .def_readwrite("cx", &RectangleSubDomain::cx)
                .def_readwrite("cy", &RectangleSubDomain::cy)
                .def_readwrite("rot_rad", &RectangleSubDomain::rot_rad)
                .def_readwrite("surface_eps", &RectangleSubDomain::surface_eps);

        }
        """

        self.compiled_subdomain = compile_cpp_code(cppcode).RectangleSubDomain()
        self.compiled_subdomain.x1 = self.x1
        self.compiled_subdomain.x2 = self.x2
        self.compiled_subdomain.y1 = self.y1
        self.compiled_subdomain.y2 = self.y2
        self.compiled_subdomain.cx = self.center_x
        self.compiled_subdomain.cy = self.center_y
        self.compiled_subdomain.rot_rad = self.rot_rad
        self.compiled_subdomain.surface_eps = self.surface_eps
        self.compiled_inside = self.compiled_subdomain.inside

class EllipseDomain(Domain):

    def __init__(self, is_inner, surface_eps, surface_ext, bc_expression, material_expression, ic_expression, center_x, center_y, width, height, rot_rad=0):
        super().__init__(is_inner, surface_eps, surface_ext, bc_expression, material_expression, ic_expression)
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.a = width/2
        self.b = height/2
        self.rot_rad = rot_rad

        self.compile_subomain()

    def coord_is_in_domain(self, x, y):
        if self.rot_rad == 0:
            p_rot = np.array([x, y])
        else:
            p = np.array([x, y])
            z = np.array([self.center_x, self.center_y])
            p_rot = rot_pnt_around_pnt(p, z, self.rot_rad)
        val = (p_rot[0]-self.center_x)**2/self.a**2 + (p_rot[1]-self.center_y)**2/self.b**2
        if val < 1:
            return True if self.is_inner else False
        else:
            return False if self.is_inner else True
    
    def coord_is_in_extended_domain(self, x, y, ext_factor=1.0):
        ext = self.surface_ext*ext_factor
        if self.is_inner == False:
            ext = -ext
        if self.rot_rad == 0:
            p_rot = np.array([x, y])
        else:
            p = np.array([x, y])
            z = np.array([self.center_x, self.center_y])
            p_rot = rot_pnt_around_pnt(p, z, self.rot_rad)
        val = (p_rot[0]-self.center_x)**2/(self.a+ext/2.0)**2 + (p_rot[1]-self.center_y)**2/(self.b+ext/2.0)**2
        if val < 1:
            return True if self.is_inner else False
        else:
            return False if self.is_inner else True

    def coord_is_on_domain_surface(self, x, y):
        if self.rot_rad == 0:
            p_rot = np.array([x, y])
        else:
            p = np.array([x, y])
            z = np.array([self.center_x, self.center_y])
            p_rot = rot_pnt_around_pnt(p, z, self.rot_rad)
        val = math.sqrt((p_rot[0]-self.center_x)**2/self.a**2 + (p_rot[1]-self.center_y)**2/self.b**2)
        if abs(val-1) < self.surface_eps:
            return True
        else:
            return False


    def compile_subomain(self):

        cppcode = """
        #include <pybind11/pybind11.h>
        #include <pybind11/eigen.h>
        #include <math.h>
        namespace py = pybind11;

        #include <dolfin/mesh/SubDomain.h>

        class EllipseSubDomain : public dolfin::SubDomain
        {
        public:

            EllipseSubDomain() : dolfin::SubDomain() {}

            bool inside(Eigen::Ref<const Eigen::VectorXd> x, bool on_boundary) const override
            {
                double val = sqrt(pow((x[0]-center_x),2)/pow(a,2) + pow((x[1]-center_y),2)/pow(b,2));
                if (on_boundary &&(abs(val-1.0) < surface_eps))
                    return true;
                else
                    return false;
            }

            double center_x;
            double center_y;
            double a;
            double b;
            double surface_eps;
        };

        PYBIND11_MODULE(SIGNATURE, m)
        {
            py::class_<EllipseSubDomain, std::shared_ptr<EllipseSubDomain>, dolfin::SubDomain>
                (m, "EllipseSubDomain")
                .def(py::init<>())
                .def_readwrite("center_x", &EllipseSubDomain::center_x)
                .def_readwrite("center_y", &EllipseSubDomain::center_y)
                .def_readwrite("a", &EllipseSubDomain::a)
                .def_readwrite("b", &EllipseSubDomain::b)
                .def_readwrite("surface_eps", &EllipseSubDomain::surface_eps);

        }
        """

        self.compiled_subdomain = compile_cpp_code(cppcode).EllipseSubDomain()
        self.compiled_subdomain.center_x = self.center_x
        self.compiled_subdomain.center_y = self.center_y
        self.compiled_subdomain.a = self.a
        self.compiled_subdomain.b = self.b
        self.compiled_subdomain.surface_eps = self.surface_eps
