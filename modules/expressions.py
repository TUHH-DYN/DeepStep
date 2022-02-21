import numpy as np
from fenics import *
from dolfin import *
# import math

from modules.helpers import rot_pnt_around_pnt

class LinearExpression(UserExpression):
    def __init__(self, start_value, end_value, start_pnt, end_pnt, degree = 2, **kwargs):
        try:    super(LinearExpression, self).__init__(**kwargs)
        except: pass
        self.degree = degree
        self.start_value = start_value
        self.end_value = end_value
        self.delta_value = end_value - start_value
        self.start_pnt = start_pnt
        self.end_pnt = end_pnt
        self.delta_pnts = end_pnt-start_pnt
        self.len_pnts = np.sqrt(np.sum(self.delta_pnts**2))
        assert self.len_pnts > 0.0 , "start and endpoint can't be equal"
        self.delta_pnts = self.delta_pnts / self.len_pnts

    def eval(self, value, x):
        delta_pos = x-self.start_pnt
        s = np.dot(delta_pos, self.delta_pnts)/self.len_pnts
        if s<0.0:
            value[0] = self.start_value
        elif s >1.0:
            value[0] = self.end_value
        else:
            value[0] = self.start_value + s*self.delta_value

    def eval_cell(self, value, x, cell):
        self.eval(self, value, x)

    def value_shape(self):
        return ()
        # return (1,)

class DualLinearExpression(UserExpression):
    def __init__(self, start_value, mid_value, end_value, start_pnt, end_pnt, degree = 2, **kwargs):
        try:    super(DualLinearExpression, self).__init__(**kwargs)
        except: pass
        self.degree = degree
        self.start_value = start_value
        self.mid_value = mid_value
        self.end_value = end_value
        self.delta_value_0 = mid_value - start_value
        self.delta_value_1 = end_value - mid_value
        self.start_pnt = start_pnt
        self.end_pnt = end_pnt
        self.delta_pnts = end_pnt-start_pnt
        self.len_pnts = np.sqrt(np.sum(self.delta_pnts**2))
        assert self.len_pnts > 0.0 , "start and endpoint can't be equal"
        self.delta_pnts = self.delta_pnts / self.len_pnts


    def eval(self, value, x):
        delta_pos = x-self.start_pnt
        s = np.dot(delta_pos, self.delta_pnts)/self.len_pnts
        if s<0.0:
            value[0] = self.start_value
        elif s > 1.0:
            value[0] = self.end_value
        elif s < 0.5:
            value[0] = self.start_value + s*2.0*self.delta_value_0
        else:
            # value[0] = self.end_value + 0.2 #+ (s-0.5)*2.0*self.delta_value_1
            value[0] = self.mid_value + (s-0.5)*2.0*self.delta_value_1

    def eval_cell(self, value, x, cell):
        self.eval(self, value, x)

    def value_shape(self):
        return ()
        # return (1,)


def construct_linear_expression(start_value, end_value, start_pnt, end_pnt, degree = 0):

    cppcode = """
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    #include <math.h>
    #include <dolfin/function/Expression.h>
    namespace py = pybind11;

    class LinearExpression : public dolfin::Expression
    {
    public:
        
        LinearExpression() : dolfin::Expression(){}

        void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
        {
            const double delta_pos_x = x[0]-start_pnt_x;
            const double delta_pos_y = x[1]-start_pnt_y;
            const double s = (delta_pos_x*delta_pnts_x + delta_pos_y*delta_pnts_y)/len_pnts;
            if (s<0.0) values[0] = start_value;
            else if(s >1.0) values[0] = end_value;
            else values[0] = start_value + s*delta_value;
        }

        double start_value;
        double end_value;
        double delta_value;
        double start_pnt_x;
        double start_pnt_y;
        double end_pnt_x;
        double end_pnt_y;
        double delta_pnts_x; //has to be normed
        double delta_pnts_y; //has to be normed
        double len_pnts; 
    };

    PYBIND11_MODULE(SIGNATURE, m)
    {
        py::class_<LinearExpression, std::shared_ptr<LinearExpression>, dolfin::Expression>
            (m, "LinearExpression")
            .def(py::init<>())
            .def("eval", &LinearExpression::eval)
            .def_readwrite("start_value", &LinearExpression::start_value)
            .def_readwrite("end_value", &LinearExpression::end_value)
            .def_readwrite("delta_value", &LinearExpression::delta_value)
            .def_readwrite("start_pnt_x", &LinearExpression::start_pnt_x)
            .def_readwrite("start_pnt_y", &LinearExpression::start_pnt_y)
            .def_readwrite("end_pnt_x", &LinearExpression::end_pnt_x)
            .def_readwrite("end_pnt_y", &LinearExpression::end_pnt_y)
            .def_readwrite("delta_pnts_x", &LinearExpression::delta_pnts_x)
            .def_readwrite("delta_pnts_y", &LinearExpression::delta_pnts_y)
            .def_readwrite("len_pnts", &LinearExpression::len_pnts);
    }
    """

    delta_value = end_value - start_value
    delta_pnts = end_pnt-start_pnt
    len_pnts = np.sqrt(np.sum(delta_pnts**2))
    assert len_pnts > 0.0 , "start and endpoint can't be equal"
    delta_pnts = delta_pnts / len_pnts
    exp = CompiledExpression(compile_cpp_code(cppcode).LinearExpression(),
                                                        start_value=start_value,
                                                        end_value=end_value,
                                                        delta_value=delta_value,
                                                        start_pnt_x=start_pnt[0],
                                                        start_pnt_y=start_pnt[1],
                                                        end_pnt_x=end_pnt[0],
                                                        end_pnt_y=end_pnt[1],
                                                        delta_pnts_x=delta_pnts[0],
                                                        delta_pnts_y=delta_pnts[1],
                                                        len_pnts=len_pnts,
                                                        degree=0)
    return exp

def construct_dual_linear_expression(start_value, mid_value, end_value, start_pnt, end_pnt, degree = 0):

    cppcode = """
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    #include <math.h>
    #include <dolfin/function/Expression.h>
    namespace py = pybind11;

    class DualLinearExpression : public dolfin::Expression
    {
    public:
        
        DualLinearExpression() : dolfin::Expression(){}

        void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
        {
            const double delta_pos_x = x[0]-start_pnt_x;
            const double delta_pos_y = x[1]-start_pnt_y;
            const double s = (delta_pos_x*delta_pnts_x + delta_pos_y*delta_pnts_y)/len_pnts;
            if (s<0.0) values[0] = start_value;
            else if(s >1.0) values[0] = end_value;
            else if(s <0.5) values[0] = start_value + s*2.0*delta_value_0;
            else values[0] = mid_value + (s-0.5)*2.0*delta_value_1;
        }

        double start_value;
        double mid_value;
        double end_value;
        double delta_value_0;
        double delta_value_1;
        double start_pnt_x;
        double start_pnt_y;
        double end_pnt_x;
        double end_pnt_y;
        double delta_pnts_x; //has to be normed
        double delta_pnts_y; //has to be normed
        double len_pnts; 
    };

    PYBIND11_MODULE(SIGNATURE, m)
    {
        py::class_<DualLinearExpression, std::shared_ptr<DualLinearExpression>, dolfin::Expression>
            (m, "DualLinearExpression")
            .def(py::init<>())
            .def("eval", &DualLinearExpression::eval)
            .def_readwrite("start_value", &DualLinearExpression::start_value)
            .def_readwrite("mid_value", &DualLinearExpression::mid_value)
            .def_readwrite("end_value", &DualLinearExpression::end_value)
            .def_readwrite("delta_value_0", &DualLinearExpression::delta_value_0)
            .def_readwrite("delta_value_1", &DualLinearExpression::delta_value_1)
            .def_readwrite("start_pnt_x", &DualLinearExpression::start_pnt_x)
            .def_readwrite("start_pnt_y", &DualLinearExpression::start_pnt_y)
            .def_readwrite("end_pnt_x", &DualLinearExpression::end_pnt_x)
            .def_readwrite("end_pnt_y", &DualLinearExpression::end_pnt_y)
            .def_readwrite("delta_pnts_x", &DualLinearExpression::delta_pnts_x)
            .def_readwrite("delta_pnts_y", &DualLinearExpression::delta_pnts_y)
            .def_readwrite("len_pnts", &DualLinearExpression::len_pnts);
    }
    """

    delta_value_0 = mid_value - start_value
    delta_value_1 = end_value - mid_value
    delta_pnts = end_pnt-start_pnt
    len_pnts = np.sqrt(np.sum(delta_pnts**2))
    assert len_pnts > 0.0 , "start and endpoint can't be equal"
    delta_pnts = delta_pnts / len_pnts
    exp = CompiledExpression(compile_cpp_code(cppcode).DualLinearExpression(),
                                                        start_value=start_value,
                                                        mid_value=mid_value,
                                                        end_value=end_value,
                                                        delta_value_0=delta_value_0,
                                                        delta_value_1=delta_value_1,
                                                        start_pnt_x=start_pnt[0],
                                                        start_pnt_y=start_pnt[1],
                                                        end_pnt_x=end_pnt[0],
                                                        end_pnt_y=end_pnt[1],
                                                        delta_pnts_x=delta_pnts[0],
                                                        delta_pnts_y=delta_pnts[1],
                                                        len_pnts=len_pnts,
                                                        degree=0)
    return exp


def construct_gaussian_expression(B, A, posX, posY, sX, sY, degree = 0):

    cppcode = """
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    #include <math.h>
    #include <dolfin/function/Expression.h>
    namespace py = pybind11;

    class GaussianExpression : public dolfin::Expression
    {
    public:
        
        GaussianExpression() : dolfin::Expression(){}

        void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
        {
            values[0] = B + A * exp(-((x[0]-posX)*(x[0]-posX)/(2.0*sX*sX) + (x[1]-posY)*(x[1]-posY)/(2.0*sY*sY)));
        }

        double B;
        double A;
        double posX;
        double posY;
        double sX;
        double sY;
    };

    PYBIND11_MODULE(SIGNATURE, m)
    {
        py::class_<GaussianExpression, std::shared_ptr<GaussianExpression>, dolfin::Expression>
            (m, "GaussianExpression")
            .def(py::init<>())
            .def("eval", &GaussianExpression::eval)
            .def_readwrite("B", &GaussianExpression::B)
            .def_readwrite("A", &GaussianExpression::A)
            .def_readwrite("posX", &GaussianExpression::posX)
            .def_readwrite("posY", &GaussianExpression::posY)
            .def_readwrite("sX", &GaussianExpression::sX)
            .def_readwrite("sY", &GaussianExpression::sY);
    }
    """

    exp = CompiledExpression(compile_cpp_code(cppcode).GaussianExpression(),
                                                        B=B,
                                                        A=A,
                                                        posX=posX,
                                                        posY=posY,
                                                        sX=sX,
                                                        sY=sY,
                                                        degree=0)
    return exp
