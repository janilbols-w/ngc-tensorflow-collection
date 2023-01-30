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
import shutil
import sys

MPI_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

def main():
    if os.path.exists(os.path.join(MPI_DIR, "bin/mpiexec")):
        from horovod.openmpi_dist.mpi_bin_utils import launch_mpi_command
        launch_mpi_command("horovodrun", sys.argv)
    else:
        # Use system installed MPI instead.
        from horovod.runner.launch import run_commandline
        run_commandline()

if __name__ == "__main__":
    sys.exit(main())
