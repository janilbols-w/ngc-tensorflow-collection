#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys

MPI_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def get_custom_ld_libpath_for_openmpi():
    ld_libpath = os.getenv('LD_LIBRARY_PATH', None)
    ld_libpath = "{old_val}{mpi_lib_dir}".format(
        old_val=ld_libpath + ":" if ld_libpath is not None else "",
        mpi_lib_dir="{}/lib".format(MPI_DIR)
    )
    return ld_libpath


def get_custom_path_for_openmpi():
    return "{}:{}".format(os.getenv('PATH', ""), "{}/bin".format(MPI_DIR))

CUSTOM_PATH = get_custom_path_for_openmpi()
CUSTOM_LD_LIBRARY_PATH = get_custom_ld_libpath_for_openmpi()

def launch_command(binary, args, env):
    os.execve(binary, args, env)


def launch_mpi_command(binary_name, argv):
    custom_env = os.environ.copy()
    custom_env["LD_LIBRARY_PATH"] = CUSTOM_LD_LIBRARY_PATH
    custom_env["PATH"] = CUSTOM_PATH
    custom_env["MPI_DIR"] = MPI_DIR
    custom_env["OPAL_PREFIX"] = MPI_DIR

    command_bin = os.path.join(MPI_DIR, "bin/{}".format(binary_name))

    launch_command(command_bin, argv, env=custom_env)
