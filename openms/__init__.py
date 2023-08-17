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
# Author: Yu Zhang <zhy@lanl.gov>
#

"""
OpenMS - Open-Source code for coupled Maxwell-Schrodinger equations in Open Quantum systems.

    Open = Open quantum systems, Open-Source, Open-Science, you name it

:noindex:
"""

import os
import sys
import textwrap

__version__ = "0.1_beta"
__author__ = "Yu Zhang (zhy@lanl.gov)"
__copyright__ = f"""
{" " * 3}
{" " * 3} @ 2023. Triad National Security, LLC. All rights reserved.
{" " * 3}
{" " * 3}This program was produced under U.S. Government contract 89233218CNA000001
{" " * 3} for Los Alamos National Laboratory (LANL), which is operated by Triad
{" " * 3}National Security, LLC for the U.S. Department of Energy/National Nuclear
{" " * 3}Security Administration. All rights in the program are reserved by Triad
{" " * 3}National Security, LLC, and the U.S. Department of Energy/National Nuclear
{" " * 3}Security Administration. The Government is granted for itself and others acting
{" " * 3}on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
{" " * 3}material to reproduce, prepare derivative works, distribute copies to the
{" " * 3}
{" " * 3}Authors: Yu Zhang <zhy@lanl.gov>
"""

__logo__ = f"""
==========================================================
*    ____    _____    _____   _   _   __  __    _____    *
*   / __ \  |  __ \  |  ___| | \ | | |  \/  |  / ____|   *
*  | |  | | | |__) | | |___  |  \| | | \  / | | (___     *
*  | |  | | |  ___/  |  ___| | \   | | |\/| |  \___ \    *
*  | |__| | | |      | |___  | |\  | | |  | |  ____) |   *
*   \____/  |_|      |_____| |_| \_| \_|  |_| |_____/    *
*                                                        *
==========================================================
{__copyright__}
Version: {__version__}
"""

__logo__ = f"""
{" " * 3}===============================================================
{" " * 3}*  ______   _____    ______   _     _   __      __   _______  *
{" " * 3}* /  __  \ |  __ \  | _____| | \   | | |  \    /  | /  _____| *
{" " * 3}* | |  | | | |  | | | |      |  \  | | | _ \  / _ | | /       *
{" " * 3}* | |  | | | |__| | | |____  | \ \ | | | |\ \/ /| | | \_____  *
{" " * 3}* | |  | | |  ___/  |  ____| | |\  \ | | | \  / | | \_____  \ *
{" " * 3}* | |  | | | |      | |      | | \   | | |  \/  | |       \ | *
{" " * 3}* | |__| | | |      | |____  | |  \  | | |      | |  _____/ | *
{" " * 3}* \______/ |_|      |______| |_|   \_| |_|      |_| |_______/ *
{" " * 3}*                                                             *
{" " * 3}===============================================================
{" " * 3}{__copyright__}
{" " * 3}Version: {__version__}
"""


__citation__ = textwrap.dedent(
    f"""\
{" " * 4} Yu, Zhang. \"Openms: A Multiscale ecosystem for solving coupled Maxwell-Schrodinger
{" " * 4} equations in Open quantum environments", https://github.com/lanl/OpenMS,
"""
)

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

from distutils.version import LooseVersion
import numpy

from openms import __config__
from openms import lib
from openms import maxwell

# from openms import qed

# Whether to enable debug mode. When this flag is set, some modules may run
# extra debug code.
DEBUG = __config__.DEBUG

from openms import spindy
from openms import qmd
