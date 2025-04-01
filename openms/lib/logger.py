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


def task_title(msg, level=1):
    if level == 0:
        length = 80
        len1 = (length - len(msg) ) // 2
        len2 = length - len(msg) - len1
        info = f"\n{'=' * length}"
        info += f"\n{' ' * len1} {msg}\n{'=' * length} "
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
