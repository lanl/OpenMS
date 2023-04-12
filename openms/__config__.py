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

import os, sys

#
# All parameters initialized before loading openms_conf.py will be overwritten
# by the dynamic importing procedure.
#

DEBUG = False

MAX_MEMORY = int(os.environ.get('OpenMS_MAX_MEMORY', 4000)) # MB
TMPDIR = os.environ.get('TMPDIR', '.')
TMPDIR = os.environ.get('OpenMS_TMPDIR', TMPDIR)

VERBOSE = 3  # default logger level (logger.NOTE)
UNIT = 'angstrom'

#
# Loading openms_conf.py and overwriting above parameters
#
for conf_file in (os.environ.get('OpenMS_CONFIG_FILE', None),
                  os.path.join(os.path.abspath('.'), '.openms_conf.py'),
                  os.path.join(os.environ.get('HOME', '.'), '.openms_conf.py')):
    if conf_file is not None and os.path.isfile(conf_file):
        break
else:
    conf_file = None

if conf_file is not None:
    if sys.version_info < (3,0):
        import imp
        imp.load_source('openms.__config__', conf_file)
        del (imp)
    else:
        from importlib import machinery
        machinery.SourceFileLoader('openms.__config__', conf_file).load_module()
        del (machinery)
del (os, sys)

#
# All parameters initialized after loading openms_conf.py will be kept in the
# program.
#
