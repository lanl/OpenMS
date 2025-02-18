#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Authors:   Yu Zhang    <zhy@lanl.gov>
#          Ilia Mazin <imazin@lanl.gov>
#

import os
import sys
import textwrap

__version__ = "0.2.0"
__author__ = "Yu Zhang (zhy@lanl.gov)"
__copyright__ = f"""
{" " * 3}
{" " * 3} @ 2023. Triad National Security, LLC. All rights reserved.
{" " * 3}
{" " * 3}This program was produced under U.S. Government contract 89233218CNA000001
{" " * 3}for Los Alamos National Laboratory (LANL), which is operated by Triad
{" " * 3}National Security, LLC for the U.S. Department of Energy/National Nuclear
{" " * 3}Security Administration. All rights in the program are reserved by Triad
{" " * 3}National Security, LLC, and the U.S. Department of Energy/National Nuclear
{" " * 3}Security Administration. The Government is granted for itself and others acting
{" " * 3}on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
{" " * 3}material to reproduce, prepare derivative works, distribute copies to the
{" " * 3}public, perform publicly and display publicly, and to permit others to do so.
{" " * 3}
{" " * 3}Authors:   Yu Zhang    <zhy@lanl.gov>
"""

#__logo__ = f"""
#==========================================================
#*    ____    _____    _____   _   _   __  __    _____    *
#*   / __ \  |  __ \  |  ___| | \ | | |  \/  |  / ____|   *
#*  | |  | | | |__) | | |___  |  \| | | \  / | | (___     *
#*  | |  | | |  ___/  |  ___| | \   | | |\/| |  \___ \    *
#*  | |__| | | |      | |___  | |\  | | |  | |  ____) |   *
#*   \____/  |_|      |_____| |_| \_| \_|  |_| |_____/    *
#*                                                        *
#==========================================================
#{__copyright__}
#Version: {__version__}
#"""


__logo__ = r"""
    ===============================================================
    *  ______   _____    ______   _     _   __      __   _______  *
    * /  __  \ |  __ \  | _____| | \   | | |  \    /  | /  _____| *
    * | |  | | | |  | | | |      |  \  | | | _ \  / _ | | /       *
    * | |  | | | |__| | | |____  | \ \ | | | |\ \/ /| | | \_____  *
    * | |  | | |  ___/  |  ____| | |\  \ | | | \  / | | \_____  \ *
    * | |  | | | |      | |      | | \   | | |  \/  | |       \ | *
    * | |__| | | |      | |____  | |  \  | | |      | |  _____/ | *
    * \______/ |_|      |______| |_|   \_| |_|      |_| |_______/ *
    *                                                             *
    ===============================================================
""" + f"""
   {__copyright__}
   Version: {__version__}
"""

# Dictionary of Zhang Group publications
_citations = {}
_citations["openms"] = textwrap.dedent(f"""\
{" " * 4} Y. Zhang and I.M. Mazin. \"OpenMS: A multi-scale ecosystem for
{" " * 4} Maxwell-Schroedinger equations in open quantum environments.\"
{" " * 6} {r'https://github.com/lanl/OpenMS'}
""")

_citations["scqedhf"] = textwrap.dedent(f"""\
{" " * 4} X. Li and Y. Zhang, \"First-principles molecular quantum electrodynamics
{" " * 4} theory at all coupling strengths\",
{" " * 6} {r'https://arxiv.org/abs/2310.18228'}
""")

_citations["vsq_qedhf"] = textwrap.dedent(f"""\
{" " * 4} Y. Zhang, "QEDHF with Squeezed operator. TBA.
""")

_citations["vtqedhf"] = textwrap.dedent(f"""\
{" " * 4} X. Li and Y. Zhang, \"First-principles molecular quantum electrodynamics
{" " * 4} theory at all coupling strengths\",
{" " * 6} {r'https://arxiv.org/abs/2310.18228'}
""")

_citations["pccp2023"] = textwrap.dedent(f"""\
{" " * 4} B.M. Weight, X. Li, Y. Zhang. \"Theory and modeling of light-matter
{" " * 4} interactions in chemistry: current and future\",
{" " * 6} Phys. Chem. Chem. Phys., 25, 3154 (2023).
""")

_citations["pra2024"] = textwrap.dedent(f"""\
{" " * 4} BM Weight, S Tretiak, Y Zhang, Diffusion quantum Monte Carlo approach to the
{" " * 4}     polaritonic ground state. Phys. Rev. A 109, 032804 (2024).
""")

_citations["afqmc2024"] = textwrap.dedent(f"""\
{" " * 4} BM Weight, Y Zhang, TBA.
""")

# Empty list store the citation info for each job
runtime_refs = ["openms", "pccp2023"]

#
OPENMS_PATH = os.getenv("OPENMS_PATH")
if OPENMS_PATH:
    for p in OPENMS_PATH.split(":"):
        if os.path.isdir(p):
            submodules = os.listdir(p)
            if "openms" in submodules:
                # OPENMS_PATH points to the root directory of a submodule
                __path__.append(os.path.join(p, "openms"))
            else:
                # Load all modules in OPENMS_PATH if it's a folder that
                # contains all extended modules
                for f in submodules:
                    __path__.append(os.path.join(p, f, "openms"))
                del f
        elif os.path.exists(p):
            # Load all moduels defined inside the file OPENMS_PATH
            with open(p, "r") as f:
                __path__.extend(f.read().splitlines())
            del f
    del p
elif "/site-packages/" in __file__ or "/dist-packages/" in __file__:
    # If openms has been installed in the standard runtime path (typically
    # under site-packages), and plugins are installed with the pip editable
    # mode, load namespace plugins. In this case, it is likely all modules are
    # managed by pip/conda or their virtual environments. It is safe to search
    # namespace plugins according to the old style of PEP 420.
    __path__ = __import__("pkgutil").extend_path(__path__, __name__)
else:
    # We need a better way to load plugins if openms is imported by the
    # environment variable PYTHONPATH. Current treatment may mix installed
    # plugins (e.g.  through pip install) with the developing plugins which
    # were accidentally placed under PYTHONPATH. When PYTHONPATH mechanism is
    # taken, an explicit list of extended paths (using environment
    # OPENMS_PATH) is recommended.
    __path__ = __import__("pkgutil").extend_path(__path__, __name__)
    if not all("/site-packages/" in p for p in __path__[1:]):
        sys.stderr.write(
            "openms plugins found in \n%s\n"
            "When PYTHONPATH is set, it is recommended to load "
            "these plugins through the environment variable "
            "OPENMS_PATH\n" % "\n".join(__path__[1:])
        )

# from distutils.version import LooseVersion

from openms import __config__
from openms import lib
from openms import maxwell

# from . import __config__
# from . import cc
# from . import lib
# from . import maxwell
# from . import models
# from . import mqed
# from . import oqs
# from . import solvers

# Whether to enable debug mode. When this flag is set, some modules may run
# extra debug code.
DEBUG = __config__.DEBUG

from openms import spindy
from openms import qmd
