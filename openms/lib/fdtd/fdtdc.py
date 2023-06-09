# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _fdtdc
else:
    import _fdtdc

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


EMP = _fdtdc.EMP

def structure_size(x, y, z):
    return _fdtdc.structure_size(x, y, z)

def lattice_size(lx, ly, lz):
    return _fdtdc.lattice_size(lx, ly, lz)

def non_uniform_grid(component, z_i, z_f, nlz):
    return _fdtdc.non_uniform_grid(component, z_i, z_f, nlz)

def non_uniform_z_to_i(z):
    return _fdtdc.non_uniform_z_to_i(z)

def non_uniform_i_to_z(i):
    return _fdtdc.non_uniform_i_to_z(i)

def ngrid_lattice_nz_z(z):
    return _fdtdc.ngrid_lattice_nz_z(z)

def ngrid_lattice_nz_i(i):
    return _fdtdc.ngrid_lattice_nz_i(i)

def find_max_lattice_nz():
    return _fdtdc.find_max_lattice_nz()

def pml_size(il, ir, jl, jr, kl, kr):
    return _fdtdc.pml_size(il, ir, jl, jr, kl, kr)

def set_default_parameter(S):
    return _fdtdc.set_default_parameter(S)

def set_sigma_order(oxl, oxr, oyl, oyr, ozl, ozr):
    return _fdtdc.set_sigma_order(oxl, oxr, oyl, oyr, ozl, ozr)

def set_sigma_max(axl, axr, ayl, ayr, azl, azr):
    return _fdtdc.set_sigma_max(axl, axr, ayl, ayr, azl, azr)

def set_kappa(kappa_x, kappa_y, kappa_z):
    return _fdtdc.set_kappa(kappa_x, kappa_y, kappa_z)

def Hz_parity(x, y, z):
    return _fdtdc.Hz_parity(x, y, z)

def periodic_boundary(x_on, y_on, k_x, k_y):
    return _fdtdc.periodic_boundary(x_on, y_on, k_x, k_y)

def memory():
    return _fdtdc.memory()

def make_epsilon():
    return _fdtdc.make_epsilon()

def make_metal_structure():
    return _fdtdc.make_metal_structure()

def background(epsilon):
    return _fdtdc.background(epsilon)

def input_object(shape, matrix_file, centerx, centery, centerz, size1, size2, size3, epsilon):
    return _fdtdc.input_object(shape, matrix_file, centerx, centery, centerz, size1, size2, size3, epsilon)

def input_object_Euler_rotation(shape, matrix_file, centerx, centery, centerz, size1, size2, size3, alpha, beta, gamma, epsilon):
    return _fdtdc.input_object_Euler_rotation(shape, matrix_file, centerx, centery, centerz, size1, size2, size3, alpha, beta, gamma, epsilon)

def input_Drude_medium(shape, centerx, centery, centerz, size1, size2, size3, epsilon_b, omega_p, gamma_0, lattice_n):
    return _fdtdc.input_Drude_medium(shape, centerx, centery, centerz, size1, size2, size3, epsilon_b, omega_p, gamma_0, lattice_n)

def input_Drude_medium2(shape, matrix_file, centerx, centery, centerz, size1, size2, size3, epsilon_b, omega_p, gamma_0, lattice_n):
    return _fdtdc.input_Drude_medium2(shape, matrix_file, centerx, centery, centerz, size1, size2, size3, epsilon_b, omega_p, gamma_0, lattice_n)

def random_object(shape, radius, height, epsilon, x_min, x_max, y_min, y_max, z_min, z_max, gen_number, seed):
    return _fdtdc.random_object(shape, radius, height, epsilon, x_min, x_max, y_min, y_max, z_min, z_max, gen_number, seed)

def random_Gaussian_dipole(component, frequency, tdecay, x_min, x_max, y_min, y_max, z_min, z_max, gen_number, seed):
    return _fdtdc.random_Gaussian_dipole(component, frequency, tdecay, x_min, x_max, y_min, y_max, z_min, z_max, gen_number, seed)

def far_field_param(OMEGA, DETECT):
    return _fdtdc.far_field_param(OMEGA, DETECT)

def make_2n_size(NROW, mm):
    return _fdtdc.make_2n_size(NROW, mm)

def far_field_FFT(NROW, NA, Nfree, OMEGA, mm):
    return _fdtdc.far_field_FFT(NROW, NA, Nfree, OMEGA, mm)

def coefficient():
    return _fdtdc.coefficient()

def sigmax(a):
    return _fdtdc.sigmax(a)

def sigmay(a):
    return _fdtdc.sigmay(a)

def sigmaz(a):
    return _fdtdc.sigmaz(a)

def propagate():
    return _fdtdc.propagate()

def propagate_tri():
    return _fdtdc.propagate_tri()

def Gaussian_dipole_source(component, x, y, z, frequency, phaes, to, tdecay):
    return _fdtdc.Gaussian_dipole_source(component, x, y, z, frequency, phaes, to, tdecay)

def Gaussian_planewave(Ecomp, Hcomp, position, frequency, to, tdecay):
    return _fdtdc.Gaussian_planewave(Ecomp, Hcomp, position, frequency, to, tdecay)

def Gaussian_beam_prop_Gauss(Ecomp, Hcomp, x, y, z, z_c, wo, n, frequency, to, tdecay):
    return _fdtdc.Gaussian_beam_prop_Gauss(Ecomp, Hcomp, x, y, z, z_c, wo, n, frequency, to, tdecay)

def Lorentzian_planewave(Ecomp, Hcomp, position, frequency, to, tdecay):
    return _fdtdc.Lorentzian_planewave(Ecomp, Hcomp, position, frequency, to, tdecay)

def Gaussian_beam_prop_Lorentz(Ecomp, Hcomp, x, y, z, z_c, wo, n, frequency, to, tdecay):
    return _fdtdc.Gaussian_beam_prop_Lorentz(Ecomp, Hcomp, x, y, z, z_c, wo, n, frequency, to, tdecay)

def Gaussian_line_source(component, position_x, position_z, frequency, phase, to, tdecay):
    return _fdtdc.Gaussian_line_source(component, position_x, position_z, frequency, phase, to, tdecay)

def Lorentzian_line_source(component, position_x, position_z, frequency, phase, to, tdecay):
    return _fdtdc.Lorentzian_line_source(component, position_x, position_z, frequency, phase, to, tdecay)

def Lorentzian_dipole_source(component, x, y, z, frequency, phaes, to, tdecay):
    return _fdtdc.Lorentzian_dipole_source(component, x, y, z, frequency, phaes, to, tdecay)

def Lorentzian_phase(Wn, tdecay):
    return _fdtdc.Lorentzian_phase(Wn, tdecay)

def Gaussian_phase(Wn, t_peak):
    return _fdtdc.Gaussian_phase(Wn, t_peak)

def incoherent_point_dipole(function, x, y, z, frequency, to, tdecay, t_mu, sd):
    return _fdtdc.incoherent_point_dipole(function, x, y, z, frequency, to, tdecay, t_mu, sd)

def eps(i, j, k):
    return _fdtdc.eps(i, j, k)

def eps_m(i, j, k):
    return _fdtdc.eps_m(i, j, k)

def eps_m2(i, j, k):
    return _fdtdc.eps_m2(i, j, k)

def eps_m3(i, j, k):
    return _fdtdc.eps_m3(i, j, k)

def meps(i, j, k):
    return _fdtdc.meps(i, j, k)

def out_epsilon(plane, value, name):
    return _fdtdc.out_epsilon(plane, value, name)

def out_epsilon_periodic(plane, value, name, m_h, m_v):
    return _fdtdc.out_epsilon_periodic(plane, value, name, m_h, m_v)

def out_epsilon_projection(dirc, name):
    return _fdtdc.out_epsilon_projection(dirc, name)

def out_plane(component, plane, value, lastname):
    return _fdtdc.out_plane(component, plane, value, lastname)

def out_plane_projection(component, dirc, lastname, k_shift):
    return _fdtdc.out_plane_projection(component, dirc, lastname, k_shift)

def out_plane_periodic(component, plane, value, lastname, m_h, m_v):
    return _fdtdc.out_plane_periodic(component, plane, value, lastname, m_h, m_v)

def out_plane_time_average(component, plane, value, start, end, field_avg, lastname):
    return _fdtdc.out_plane_time_average(component, plane, value, start, end, field_avg, lastname)

def out_plane_time_average_projection(component, dirc, start, end, field_avg, lastname, k_shift):
    return _fdtdc.out_plane_time_average_projection(component, dirc, start, end, field_avg, lastname, k_shift)

def out_several_points(component, zposition, xside, yside, pNx, pNy, ti, tf, name):
    return _fdtdc.out_several_points(component, zposition, xside, yside, pNx, pNy, ti, tf, name)

def out_point(component, x, y, z, ti, tf, name):
    return _fdtdc.out_point(component, x, y, z, ti, tf, name)

def grid_value(component, i, j, k):
    return _fdtdc.grid_value(component, i, j, k)

def total_E_energy():
    return _fdtdc.total_E_energy()

def total_EM_energy():
    return _fdtdc.total_EM_energy()

def total_E_energy_block(centerx, centery, centerz, size1, size2, size3):
    return _fdtdc.total_E_energy_block(centerx, centery, centerz, size1, size2, size3)

def total_E_energy2_block(centerx, centery, centerz, size1, size2, size3):
    return _fdtdc.total_E_energy2_block(centerx, centery, centerz, size1, size2, size3)

def total_E_energy3_block(centerx, centery, centerz, size1, size2, size3):
    return _fdtdc.total_E_energy3_block(centerx, centery, centerz, size1, size2, size3)

def total_E_energy_thin_block_z(centerx, centery, centerz, size1, size2, size3, eps_L, eps_H, name):
    return _fdtdc.total_E_energy_thin_block_z(centerx, centery, centerz, size1, size2, size3, eps_L, eps_H, name)

def max_E_Energy_detector(centerx, centery, centerz, size1, size2, size3):
    return _fdtdc.max_E_Energy_detector(centerx, centery, centerz, size1, size2, size3)

def total_EM_energy_block(centerx, centery, centerz, size1, size2, size3):
    return _fdtdc.total_EM_energy_block(centerx, centery, centerz, size1, size2, size3)

def total_E2():
    return _fdtdc.total_E2()

def Drude_energy_loss_in_block(centerx, centery, centerz, size1, size2, size3):
    return _fdtdc.Drude_energy_loss_in_block(centerx, centery, centerz, size1, size2, size3)

def Drude_energy_loss_in_block2(centerx, centery, centerz, size1, size2, size3, WC, lattice_n):
    return _fdtdc.Drude_energy_loss_in_block2(centerx, centery, centerz, size1, size2, size3, WC, lattice_n)

def Poynting_total():
    return _fdtdc.Poynting_total()

def Poynting_block(centerx, centery, centerz, size1, size2, size3):
    return _fdtdc.Poynting_block(centerx, centery, centerz, size1, size2, size3)

def Poynting_half_sphere_point(component, z0, R, theta, phi):
    return _fdtdc.Poynting_half_sphere_point(component, z0, R, theta, phi)

def Poynting_side(value, zposition):
    return _fdtdc.Poynting_side(value, zposition)

def print_energy():
    return _fdtdc.print_energy()

def Poynting_UpDown(value, zposition):
    return _fdtdc.Poynting_UpDown(value, zposition)

def transform_farfield(NROW, tnum, name, mm):
    return _fdtdc.transform_farfield(NROW, tnum, name, mm)

def add_farfield(tnum, name):
    return _fdtdc.add_farfield(tnum, name)

def print_amp_and_phase(mm):
    return _fdtdc.print_amp_and_phase(mm)

def print_real_and_imag(mm):
    return _fdtdc.print_real_and_imag(mm)

def print_real_and_imag_2n_size(NROW, mm):
    return _fdtdc.print_real_and_imag_2n_size(NROW, mm)

def field_initialization():
    return _fdtdc.field_initialization()

def real_space_param(a_nm, w_n):
    return _fdtdc.real_space_param(a_nm, w_n)

def get_period_in_update(w_n):
    return _fdtdc.get_period_in_update(w_n)

def Gauss_amp(frequency, phase, to, tdecay):
    return _fdtdc.Gauss_amp(frequency, phase, to, tdecay)

def Lorentz_amp(frequency, phase, to, tdecay):
    return _fdtdc.Lorentz_amp(frequency, phase, to, tdecay)

def iGauss_amp(frequency, phase, to, tdecay):
    return _fdtdc.iGauss_amp(frequency, phase, to, tdecay)

def iLorentz_amp(frequency, phase, to, tdecay):
    return _fdtdc.iLorentz_amp(frequency, phase, to, tdecay)

cvar = _fdtdc.cvar

