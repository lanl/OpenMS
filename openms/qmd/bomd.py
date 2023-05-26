
import numpy
import openms

from __future__ import division
from openms.lib.misc import fs2au, au2A, au2K, call_name, typewriter
import textwrap, datetime
import os, shutil
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


    def initialize(self, qm, output_dir="./", save_qm_log=False, save_scratch=False, restart=None, scratch_dir=None):
        r""" Initialize MD dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
            :param string output_dir: path of output files, this can be a scratch space for storing large intermediate files
            :param string scratch_dir: scratch path of qm calculators. default [None]. use the output_dir
            :param boolean save_qm_log: Logical for saving QM calculation log
            :param boolean save_scr: Logical for saving scratch directory
            :param string restart: Option for controlling dynamics restarting
        """
        
        if (restart != None): restart = restart.lower()
        if not (restart in [None, "write", "append"]):
            error_message = "Invalid restart option!"
            error_vars = f"restart = {restart}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Check if NACVs are calculated for Ehrenfest dynamics
        if (self.md_type == "Eh" and self.mol.lnacme):
            error_message = "Ehrenfest dynamics needs evaluation of NACVs, check your QM object!"
            error_vars = f"(QM) qm_prog.qm_method = {qm.qm_prog}.{qm.qm_method}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Set directory information
        output_dir = os.path.expanduser(output_dir)
        base_dir = []
        md_dir = []
        qm_log_dir = []

        dir_tmp = os.path.join(os.getcwd(), output_dir)
        if (self.md_type != "CT"):
            base_dir.append(dir_tmp)
        else:
            for itraj in range(self.ntrajs):
                itraj_dir = os.path.join(dir_tmp, f"TRAJ_{itraj + 1:0{self.digit}d}")
                base_dir.append(itraj_dir)

        for idir in base_dir:
            md_dir.append(os.path.join(idir, "md"))
            qm_log_dir.append(os.path.join(idir, "qm_log"))

        # Check and make directories
        if (restart == "append"):
            # For MD output directory
            for md_idir in md_dir:
                if (not os.path.exists(md_idir)):
                    error_message = f"Directory {md_idir} to be appended for restart not found!"
                    error_vars = f"restart = {restart}, output_dir = {output_dir}"
                    raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

            # For QM output directory
            if (save_qm_log):
                for qm_idir in qm_log_dir:
                    if (not os.path.exists(qm_idir)):
                        os.makedirs(qm_idir)

        else:
            # save old output
            # For MD output directory
            for md_idir in md_dir:
                if (os.path.exists(md_idir)):
                    shutil.move(md_idir, md_idir + "_old_" + str(os.getpid()))
                os.makedirs(md_idir)

                self.touch_file(md_idir)

            # For QM output directory
            for qm_idir in qm_log_dir:
                if (os.path.exists(qm_idir)):
                    shutil.move(qm_idir, qm_idir + "_old_" + str(os.getpid()))
                if (save_qm_log):
                    os.makedirs(qm_idir)

        os.chdir(base_dir[0])

        if (self.md_type != "CT"):
            return base_dir[0], md_dir[0], qm_log_dir[0]
        else:
            return base_dir, md_dir, qm_log_dir

    #def compute_accel(self):
    #    self.accel = self.force / self.mol.mass.reshape(-1,1)
        
    def next_position(self):
        """ Routine to update nuclear positions

        .. math::
            r(t_i+1) = r(t_i) + \Delta t * v(t_i) + 0.5(/delta t)^2 a(t_i)
        """
        #self.mol.veloc += 0.5 * self.dt * self.force / self.mol.mass.reshape(-1, 1)
        self.next_velocity()
        self.mol.pos += self.dt * self.mol.veloc

    def next_velocity(self): 
        """
        Compute the next velocity using the Velocity Verlet algorithm. The
        necessary equations of motion for the velocity is
        
        .. math::

           v(t_i+1) = v(t_i) + 0.5(a(t_i+1) + a(t_i))

        Hence, this function should be called twice, with forces at t_i + 1 and t_i respectively
        (the first is called with next_position).
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
    
    def print_init(self, qm, restart):
        """ Routine to print the initial information of dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
            :param string restart: Option for controlling dynamics restarting
        """
        cur_time = datetime.datetime.now()
        cur_time = cur_time.strftime("%Y-%m-%d %H:%M:%S")
        print(openms.__logo__)

        citation_info = textwrap.dedent(f"""\
        {"-" * 68}
        {" " * 10} Citation information
        {"-" * 68}

        {" " * 4}Please cite OpenMS as follows:
        """)
        citation_info += openms.__citation__
        citation_info += textwrap.dedent(f"""\
        {" " * 4}
        {" " * 4} openms md modules are modified from PyUNIxMD version 20.1

        {"< Developers >":>40s}
        {" " * 4}Seung Kyu Min,  In Seong Lee,  Jong-Kwon Ha,  Daeho Han,
        {" " * 4} PyUnixMD citation:
        {" " * 4}
        {" " * 4}I. S. Lee, J.-K. Ha, D. Han, T. I. Kim, S. W. Moon, & S. K. Min.
        {" " * 4}PyUNIxMD: A Python-based excited state molecular dynamics package.
        {" " * 4}Journal of Computational Chemistry, 42:1755-1766. 2021

        {" " * 4} QMD begins on {cur_time}
        """)
        print (citation_info, flush=True)


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

        output_dir = kwargs["output_dir"] if "output_dir" in kwargs else "./"
        save_qm_log = kwargs["save_qm_log"] if "save_qm_log" in kwargs else False
        save_scr = kwargs["save_scr"] if "save_scr" in kwargs else True
        restart = kwargs["restart"] if "restart" in kwargs else None

        # Initialize 
        base_dir, md_dir, qm_log_dir =\
             self.initialize(qm, output_dir, save_qm_log, save_scr, restart)

        bo_list = [self.init_state]
        qm.calc_coupling = False
        self.print_init(qm, restart)

        # move to initialize

