#!/usr/bin/env python3

#################################### HEADER ####################################

# File creation : Sat Oct  2 20:05:12 2020
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
    # Paths where the output files are located.
    # test_outputs2 is fictive, I only changed the total energy by one order
    # with respect to test_outputs to quickly test the functionality.
    paths = ["test_outputs/best_su_best_tm/","test_outputs2/best_su_best_tm/"]

    # Instantiate a plotter object
    plotter = Plotter(paths[0])

    # Call the three energy plots of varying detail
    plotter.plot_design_space(paths)


##################################### MAIN #####################################

if __name__ == "__main__":
    main()

##################################### EOF ######################################
