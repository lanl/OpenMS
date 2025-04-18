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
Logging system

more details TBA
"""

import sys
import os
import time
import datetime

if sys.version_info < (3, 0):
    process_clock = time.clock
    perf_counter = time.time
else:
    process_clock = time.process_time
    perf_counter = time.perf_counter

import openms.__config__


VERBOSE_DEBUG  = 5
VERBOSE_INFO   = 4
VERBOSE_NOTICE = 3
VERBOSE_WARN   = 2
VERBOSE_ERR    = 1
VERBOSE_QUIET  = 0

DEBUG4 = VERBOSE_DEBUG + 4
DEBUG3 = VERBOSE_DEBUG + 3
DEBUG2 = VERBOSE_DEBUG + 2
DEBUG1 = VERBOSE_DEBUG + 1

from openms.__mpi__ import MPI
rank = MPI.COMM_WORLD.Get_rank()

def flush(rec, msg, *args):
    if rank == 0:
        rec.stdout.write(msg%args)
        rec.stdout.write('\n')
        rec.stdout.flush()

def log(rec, msg, *args):
    if rec.verbose > VERBOSE_QUIET:
        flush(rec, msg, *args)

def error(rec, msg, *args):
    if rec.verbose >= VERBOSE_ERROR:
        flush(rec, '\nERROR: '+msg+'\n', *args)
    sys.stderr.write('ERROR: ' + (msg%args) + '\n')

def warn(rec, msg, *args):
    if rec.verbose >= VERBOSE_WARN:
        flush(rec, '\nWARN: '+msg+'\n', *args)
        if rec.stdout is not sys.stdout:
            sys.stderr.write('WARN: ' + (msg%args) + '\n')

def info(rec, msg, *args):
    if rec.verbose >= VERBOSE_INFO:
        flush(rec, msg, *args)

def note(rec, msg, *args):
    if rec.verbose >= VERBOSE_NOTICE:
        flush(rec, msg, *args)

def debug(rec, msg, *args):
    if rec.verbose >= VERBOSE_DEBUG:
        flush(rec, msg, *args)

def debug1(rec, msg, *args):
    if rec.verbose >= VERBOSE_DEBUG1:
        flush(rec, msg, *args)

def debug2(rec, msg, *args):
    if rec.verbose >= VERBOSE_DEBUG2:
        flush(rec, msg, *args)

def debug3(rec, msg, *args):
    if rec.verbose >= VERBOSE_DEBUG3:
        flush(rec, msg, *args)

def debug4(rec, msg, *args):
    if rec.verbose >= VERBOSE_DEBUG4:
        flush(rec, msg, *args)

def stdout(rec, msg, *args):
    if rec.verbose >= VERBOSE_DEBUG:
        flush(rec, msg, *args)
    if rank == 0:
        sys.stdout.write('>>> %s\n' % msg)



def task_title(msg, level=1):
    if level == 0:
        length = 80
        len1 = (length - len(msg) ) // 2
        len2 = length - len(msg) - len1
        info = f"\n{'=' * length}"
        info += f"\n{' ' * len1} {msg}\n{'=' * length}"
    elif level == 2:
        length = 60
        len1 = (length - len(msg) ) // 2
        len2 = length - len(msg) - len1
        info = f"\n{'*' * len1} {msg} {'*' * len2}"
    else:
        length = 80
        len1 = (length - len(msg) ) // 2
        len2 = length - len(msg) - len1
        info = f"\n{'-' * len1} {msg} {'-' * len2}"
    return info


class Logger(object):

    #def __init__(self, log_file="log.txt", verbose="INFO"):
    def __init__(self, log_file=sys.stdout, verbose="INFO"):
        self.verbose = verbose
        self.log_file = log_file
        self.log_levels = {
                "QUIET": 0,
                "CRITICAL": 5,
                "INFO": 10,
                "ERROR": 15,
                "WARNING": 20,
                "DEBUG": 25,
                "DEBUGALL": 30
                }

    def log_basic(self, level, message):
        if level not in self.log_levels and level !="":
            raise ValueError(f"Invalid log level: {level}")

        #timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #log_entry = f"{timestamp} [{level}] {message}\n"
        log_entry = f"{level} {message}\n"

        if self.log_file == sys.stdout:
            print(log_entry)
        else:
            with open(self.log_file, "a") as f:
                f.write(log_entry)

    def debugall(self, message):
        if self.log_levels[self.verbose] >= self.log_levels["DEBUGALL"]:
            self.log_basic("DEBUGALL", message)

    def debug(self, message):
        if self.log_levels[self.verbose] >= self.log_levels["DEBUG"]:
            self.log_basic("DEBUG", message)

    def info(self, message):
        if self.log_levels[self.verbose] >= self.log_levels["INFO"]:
            self.log_basic("INFO", message)

    def warning(self, message):
        if self.log_levels[self.verbose] >= self.log_levels["WARNING"]:
            self.log_basic("WARNING", message)

    def error(self, message):
        if self.log_levels[self.verbose] >= self.log_levels["ERROR"]:
            self.log_basic("ERROR", message)

    def critical(self, message):
        if self.log_levels[self.verbose] >= self.log_levels["CRITICAL"]:
            self.log_basic("CRITICAL", message)

    def log(self, message):
        if self.log_levels[self.verbose] >= self.log_levels["QUIET"]:
            self.log_basic("", message)

if __name__ == "__main__":
    logger = Logger(log_file="test.dat")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
