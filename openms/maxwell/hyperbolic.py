import numpy
import scipy

e_mass = 9.1093837139e-31 # free electron mass
e_SI = 1.602176634e-19 # C
charge = 1.0   # in au
au2ev = 4.3597447222060e-18/e_SI
epsilon0 = 8.8541878188e-12 # C^2 s^2 Kg^{-1} m^{-3}
hbar = 6.582119569e-16      # ev*s


r"""
Drude model

.. math::
    \sigma(\omega) = \frac{ne^2\tau}{m(1-i\omega\tau)}
                   = \frac{i ne^2}{m(\omega + i 1/\tau)}
                   \equiv \frac{i ne^2}{m(\omega + i\eta)}
                   = \frac{i\epsilon_0\omega^2_p}{\omega+i\eta}

where :math:`\eta \equiv 1/\tau` is the damping of the free electrons.

"""

def format_array_line(sigma):
    sigma_array = numpy.atleast_1d(sigma)
    formatted = [f"{s.real:12.8f}  {s.imag:12.8f} ;" for s in sigma_array]
    return " ".join(formatted)


class hyperbolicBase(object):
    r"""
    density: electronic density that controls the plasmon frequency. Unit: nm^{-3}
    mass: effective mass (in the unit of :math:`m_e`)
    omegas: array of frequncies, corresponding to the jth component of the conductivity,
    strengths: j-th element accounts for the strength of the j-th interband component


    Ref: PRL 116, 066804 (2016)

    .. math::
        \sigma_{jj} = \frac{ie^2 n}{(\omega + i\eta) m_j}
        + s_j \left[\Theta(\omega - \omega_j)  + \frac{i}{\pi}
        \text{ln}\left|\frac{\omega - \omega_j}{\omega + \omega_j}\right| \right]

    Plasmon frequency can be computed from the electron density and effective mass

    .. math::
        \omega_p = \sqrt{\frac{e^2 n}{\epsilon_0 m}}

    or

    .. math::
        \omega_p = \sqrt{\frac{e^2 nq}{2\epsilon_0 m}}
    for 2D materials


    units:
      - :math:`e^2 n` --> :math:`C^2 * m^{-3}`
      - :math:`\epsilon_0` --> :math:`C^2 s^2 Kg^{-1} m^{-3}`
      - :math:`e^2 n/(\epsilon_0 m_e)` --> :math:`s^{-2}`
      - :math:`e^2 n / m (\omega + i\eta)` --> :math:`C^2 m^{-3} * Kg^{-1} s`
      - :math:`e^2/\hbar` --> :math:`C^2/ (kg*m^2/s^2 s) = C^2 kg s/m^2`
    """
    def __init__(self,
        plasmon_freqs=None,       # plasmonic frequencies
        interband_freqs=None,     # interband transition frequencies
        interband_strengths=None, # interband transiiton strengths
        density=None,        # unit:
        mass=None,           # electronic effective mass, unit:
        **kwargs
    ):
        self.interband_freqs = interband_freqs
        self.interband_strengths = interband_strengths
        self.density = density
        if density is not None:
            # compute the plasmon_freqs from density
            tmp = density * 1.e27 / (epsilon0 * mass * e_mass)
            plasmon_freqs = hbar * e_SI * numpy.sqrt(tmp)
            print("sqrt{n^2/(\epsilon_0 * m) =", numpy.sqrt(tmp))
            print("plasmon_frequencies are ", plasmon_freqs)
        self.damping = kwargs.get("damping", 0.1)
        print("damping is", self.damping)

        self.plasmon_freqs = plasmon_freqs


    def get_conductivity(self, wlist):
        r"""compute the real and imaginary part of the conductivities

        """

        sigmas = numpy.zeros((len(wlist), len(self.plasmon_freqs)), dtype=numpy.complex128)
        for iw, w in enumerate(wlist):
            sigma = 1j * self.plasmon_freqs / (w + 1j * self.damping)
            tmp = abs((w - self.interband_freqs) / (w + self.interband_freqs))
            tmp = 1j / numpy.pi * numpy.log(tmp)
            for i in range(self.plasmon_freqs.size):
                if w > self.plasmon_freqs[i]:
                    tmp += 1.0
            sigma += tmp * self.interband_strengths
            outstring = format_array_line(sigma)
            print(f"w, sigma = {w:.3f} {outstring}")
            sigmas[iw] = sigma
        return sigmas


    def get_dielectric(self, wlist):
        r"""compute the real and imaginary part of the dielectric function

        Dielectric function can be obtained from the conductivity:

        .. math::
            \epsilon(\omega) = 1 + i\frac{\sigma(\omega)}{\epsilon_0\omega}
            = 1 - \frac{\omega^2_p}{\omega(\omega + i\gamma)}
        """

        for w in wlist:
            epsilon_r = 1.0
            epsilon_i = 1.0


def plot_sigma(omegas, sigmas, figname="None"):
    r"""plot the sigmas (real and imaginary parts) vs frequencies"""
    import matplotlib.pyplot as plt

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharex=True)

    # --- Panel (a): Real part ---
    for i in range(sigmas.shape[1]):
        ax1.plot(omegas, sigmas[:, i].real, label=f"Mode {i+1}")
    ax1.set_ylabel("Re(σ)", fontsize=12)
    # ax1.set_title("(a) Real Part of σ", loc="left", fontsize=14)
    ax1.legend()
    #ax1.grid(True)

    # --- Panel (b): Imaginary part ---
    for i in range(sigmas.shape[1]):
        ax2.plot(omegas, sigmas[:, i].imag, label=f"Mode {i+1}")
    ax2.set_xlabel("ω", fontsize=12)
    ax2.set_ylabel("Im(σ)", fontsize=12)
    # ax2.set_title("(b) Imaginary Part of σ", loc="left", fontsize=14)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    # ax2.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    Au = hyperbolicBase(density=59.0, mass=1.1)
    # compute the real/imaginary part of the conductivity

    n = 0.1 # nm^{-2} for 2D
    q = 1.0 # 1/nm
    eta = 0.01 # ev
    # s0 = e^2/4\hbar
    s0 = e_SI / (4. * hbar) # C^2 / (J*s)
    sx = 1.7 * s0
    sy = 3.7 * s0
    mass_x = 0.2
    mass_y = 1.0
    omega_x = 1.0
    omega_y = 0.35

    HPhP2D = hyperbolicBase(density=n * numpy.sqrt(q/2.0),
                            interband_freqs=numpy.array([omega_x, omega_y]),
                            interband_strengths=numpy.array([sx, sy]),
                            #mass=1.0,
                            mass=numpy.array([mass_x, mass_y]),
                            damping=eta
                            )

    wlist = numpy.arange(0.05, 1.0, 0.02)
    # wlist[0] += 1.e-2
    print(f"len of wlist is {len(wlist)}\nwlist = {wlist}")
    sigmas = HPhP2D.get_conductivity(wlist)
    plot_sigma(wlist, sigmas)

