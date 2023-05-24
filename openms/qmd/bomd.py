
import numpy
import openms

class BaseMD(object):
    r""" Basic Class for nuclear/electronic propagator used in MQC dynamics

        :param object molecule: Molecule object
        :param object thermostat: Thermostat object
        :param double dt: Time interval
        :param integer nsteps: Total step of nuclear propagation
        :param integer init_state: Initial state
        :param integer out_freq: Frequency of printing output
        :param integer verbosity: Verbosity of output
    """
    def __init__(self, molecule, thermostat, **kwargs):
        self.dt = 1.0
        self.nsteps = 10
        self.out_freq = 1
        self.verbosity = 0
        self.init_state = 0
        self.__dict__.update(kwargs)
 
        # Initialize Molecule object
        self.mol = molecule

        # Initialize Thermostat object
        self.thermo = thermostat

        # Initialize time step
        self.cstep = -1  # classical step
        self.qstep = -1  # quantum step

        self.force = numpy.zeros((self.mol.natm, self.mol.ndim))
        #self.accel = numpy.zeros((self.mol.natm, self.mol.ndim))
        self.accel = None


    def initialize(self, qm):
        r""" Initialize MD dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
        """
        pass

    #def compute_accel(self):
    #    self.accel = self.force / self.mol.mass.reshape(-1,1)
        
    def update_position(self):
        """ Routine to update nuclear positions

        .. math::
            r(t_i+1) = r(t_i) + \Delta t * v(t_i) + 0.5(/delta t)^2 a(t_i)
        """
        #self.mol.veloc += 0.5 * self.dt * self.force / self.mol.mass.reshape(-1, 1)
        self.update_velocity()
        self.mol.pos += self.dt * self.mol.veloc

    def update_velocity(self): 
        """
        Compute the next velocity using the Velocity Verlet algorithm. The
        necessary equations of motion for the velocity is
        
        .. math::

           v(t_i+1) = v(t_i) + 0.5(a(t_i+1) + a(t_i))

        Hence, this function should be called twice, with forces at t_i + 1 and t_i respectively
        (the first is called with update_position).
        """
        self.mol.veloc += 0.5 * self.dt * self.force /self.mol.mass.reshape(-1, 1)
        #self.mol.update_kinetic()

    def calculate_force(self):
        """ Routine to calculate the forces
        """
        pass

    def update_potential(self):
        """ Routine to update the potential of molecules
        """
        pass

    def temperature(self):
        return self.mol.ekin * 2. / float(self.mol.ndof) * au2K

###
class BOMD(BaseMD):
    r""" Class for born-oppenheimer molecular dynamics (BOMD)

    object molecule: Molecule object
    object thermostat: Thermostat object
    integer init_state: Electronic state
    double dt: Time interval
    integer nsteps: Total step of nuclear propagation
    integer out_freq: Frequency of printing output
    integer verbosity: Verbosity of output
    """
    def __init__(self, molecule, thermostat=None, **kwargs):
        # Initialize input values
        super().__init__(molecule, thermostat, **kwargs)
        self.md_type = self.__class__.__name__

    def kernel(self, qm, **kwargs):
        r""" Run MQC dynamics according to BOMD
        """
        pass

