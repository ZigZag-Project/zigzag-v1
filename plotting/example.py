#!/usr/bin/env python3

#################################### HEADER ####################################

# Copyright 2020 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause

# File creation : Wed Sep 30 16:50:08 2020
# Language : Python3

################################### IMPORTS ####################################

# Internal imports
from reader import Reader  # Used to read a bunch of ZigZag outputs at once.

##################################### CODE #####################################


def main():
    """
    An example of use of the Reader class. It requires the output to already
    exist though, you can get it by running:
    
    python3 top_module.py
        --arch ./inputs/architecture.yaml
        --map ./inputs/mapping.yaml
        --set ./inputs/settings.yaml
        --mempool ./inputs/memory_pool.yaml
    """
    # The directory where the output is stored, starting from the current
    # directory (hence the ".." to get back to the root of the repository).
    output_directory = "../results/best_su_best_tm/"
    # We load all the layers with a Reader.
    reader = Reader(output_directory)
    # And then we print all the total energy values.
    for layer_name, layer_energy in reader.energy().items():
        print(layer_name, layer_energy["total_energy"])

    # Pretty line for the style.
    print("*" * 80)

    # Example of working with the memory view.
    memory_view = reader.memory()
    print(
        *[
            memory
            for layer_view in memory_view.values()
            for data_type_view in layer_view.values()
            for memory in data_type_view.values()
        ],
        sep="\n\n"
    )

    # Pretty line for the style.
    print("*" * 80)

    # Example of working with the flattened DataFrame.
    data_frame = reader.flatten()
    print(data_frame)

    # Pretty line for the style.
    print("*" * 80)

    # Example of working with the memory DataFrame.
    data_frame = reader.memory_flatten("Eyeriss")
    print(data_frame)


##################################### MAIN #####################################

if __name__ == "__main__":
    main()

##################################### EOF ######################################
