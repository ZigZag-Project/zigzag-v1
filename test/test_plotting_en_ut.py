#!/usr/bin/env python3

#################################### HEADER ####################################

# Copyright 2020 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License_Identifier: BSD-3-Clause

# File creation : Thu Oct  8 12:42:00 2020
# Language : Python3

################################### IMPORTS ####################################

# Standard library
from os.path import join  # Used for path manipulations

# Internal imports
from test.utils import zigzag  # Used to run ZigZag with the given files.

##################################### CODE #####################################

# NOTE
# This file is not only meant as a simple test for ZigZag, but also as a
# template that should be copy pasted to create other tests.


def test_base():
    """
    BUG
    ===
    This test should always run correctly.
    """
    # First, the name of the directory in which ZigZag's input files are
    # located.
    directory = "./test/files/plotting_en_ut"
    # Then, the base name of all the input files we will be working with.
    settings = "settings.yaml"
    mapping = "mapping.yaml"
    memory_pool = "memory_pool.yaml"
    architecture = "architecture.yaml"
    # Now we run the zigzag framework and assert that the run is successfull.
    assert zigzag(
        settings=join(directory, settings),
        mapping=join(directory, mapping),
        memory_pool=join(directory, memory_pool),
        architecture=join(directory, architecture),
    )


##################################### MAIN #####################################

if __name__ == "__main__":
    # The actions to perform when this file if called as a script go here
    pass

##################################### EOF ######################################
