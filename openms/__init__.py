# Copyright 2023. Triad National Security, LLC. All rights reserved. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

__version__ = '0.1_beta'
__author__ = "Yu Zhang (zhy@lanl.gov)"
__copyright__ = f"""
{" " * 3}Copyright 2023. Triad National Security, LLC. All rights reserved. 
{" " * 3}
{" " * 3}Licensed under the Apache License, Version 2.0 (the "License");
{" " * 3}you may not use this file except in compliance with the License.
{" " * 3}You may obtain a copy of the License at
{" " * 3}
{" " * 3}    http://www.apache.org/licenses/LICENSE-2.0
{" " * 3}
{" " * 3}Unless required by applicable law or agreed to in writing, software
{" " * 3}distributed under the License is distributed on an "AS IS" BASIS,
{" " * 3}WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
{" " * 3}See the License for the specific language governing permissions and
{" " * 3}limitations under the License.
{" " * 3}
{" " * 3}Authors: xx, Yu Zhang <zhy@lanl.gov>
"""

__logo__=f"""
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

__logo__=f"""
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


__citations__=f"""
TBA
"""

#
OPENMS_PATH = os.getenv('OPENMS_PATH')
if OPENMS_PATH:
    for p in OPENMS_PATH.split(':'):
        if os.path.isdir(p):
            submodules = os.listdir(p)
            if 'openms' in submodules:
                # OPENMS_PATH points to the root directory of a submodule
                __path__.append(os.path.join(p, 'openms'))
            else:
                # Load all modules in OPENMS_PATH if it's a folder that
                # contains all extended modules
                for f in submodules:
                    __path__.append(os.path.join(p, f, 'openms'))
                del f
        elif os.path.exists(p):
            # Load all moduels defined inside the file OPENMS_PATH
            with open(p, 'r') as f:
                __path__.extend(f.read().splitlines())
            del f
    del p
elif '/site-packages/' in __file__ or '/dist-packages/' in __file__:
    # If openms has been installed in the standard runtime path (typically
    # under site-packages), and plugins are installed with the pip editable
    # mode, load namespace plugins. In this case, it is likely all modules are
    # managed by pip/conda or their virtual environments. It is safe to search
    # namespace plugins according to the old style of PEP 420.
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)
else:
    # We need a better way to load plugins if openms is imported by the
    # environment variable PYTHONPATH. Current treatment may mix installed
    # plugins (e.g.  through pip install) with the developing plugins which
    # were accidentally placed under PYTHONPATH. When PYTHONPATH mechanism is
    # taken, an explicit list of extended paths (using environment
    # OPENMS_PATH) is recommended.
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)
    if not all('/site-packages/' in p for p in __path__[1:]):
        sys.stderr.write('openms plugins found in \n%s\n'
                         'When PYTHONPATH is set, it is recommended to load '
                         'these plugins through the environment variable '
                         'OPENMS_PATH\n' % '\n'.join(__path__[1:]))

from distutils.version import LooseVersion
import numpy

from openms import __config__
from openms import lib
from openms import maxwell
#from openms import qed

# Whether to enable debug mode. When this flag is set, some modules may run
# extra debug code.
DEBUG = __config__.DEBUG

from openms import qmd

