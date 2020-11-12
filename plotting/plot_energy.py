#!/usr/bin/env python3

#################################### HEADER ####################################

# File creation : Fri Oct  2 18:05:42 2020
# Language : Python3

################################### IMPORTS ####################################

# Internal imports
from plotter import Plotter

##################################### CODE #####################################


def main():
    """
    Main function which automatically draws some plots. This can be used as an
    entry point for a python package later on.
    """
    # Path where the output files are located.
    # For now, stick to a single memory hierarchy
    # and fixed spatial unrolling results
    path = "test_outputs/best_su_best_tm/"

    # Instantiate a plotter object
    plotter = Plotter(path)

    # Call the three energy plots of varying detail
    plotter.plot_energy()


##################################### MAIN #####################################

if __name__ == "__main__":
    main()

##################################### EOF ######################################
