#!/usr/bin/env python3

#################################### HEADER ####################################

# Copyright 2020 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause

# File creation : Tue Nov 24 16:50:51 2020
# Language : Python3

################################### IMPORTS ####################################

# Standard library
import sys  # For argv.

# External imports
import matplotlib.pyplot as plt  # For plotting.
import pandas as pd  # For DataFrame manipulations.

# Internal imports
from reader import Reader  # Used to read a bunch of ZigZag outputs at once.

##################################### CODE #####################################


def main():
    """
    Simple preset pandas based plotting functions for ZigZag.

    The ARGV when calling this function should be the path of the directories
    and YAML files where the output of ZigZag can be found.
    """
    # First, we load all the provided locations into the Reader.
    reader = Reader(*sys.argv[1:])
    # Plotting global energy information.
    data_frame = reader.flatten().sort_values(by="full_name")
    # Computin the memory share of energy consumption.
    data_frame["memory_energy"] = (
        data_frame["total_energy"] - data_frame["mac_energy"]
    )
    data_frame.plot(
        y=["memory_energy", "mac_energy"],
        kind="bar",
        rot=45,
        stacked=True,
        title="Energy consumption across runs",
    )
    plt.tight_layout()

    # Plotting detail of energy consumption for the different memories.
    data_frame = reader.memory_flatten().reset_index()
    # Splitting the DatFrame across the different runs.
    for name in set(data_frame["full_name"].to_list()):
        sub_data_frame = data_frame[data_frame["full_name"] == name].set_index(
            "memory"
        )
        # Plotting the subset of the memory DataFrame.
        sub_data_frame.plot(
            x="memory",
            y="energy",
            kind="pie",
            title=f"Energy share for {name}",
            autopct="%1.0f%%",
        )
        plt.tight_layout()
    plt.show()


##################################### MAIN #####################################

if __name__ == "__main__":
    # Running the main.
    main()

##################################### EOF ######################################
