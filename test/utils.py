#!/usr/bin/env python3

#################################### HEADER ####################################

# File creation : Thu Oct  8 12:30:43 2020
# Language : Python3

################################### IMPORTS ####################################

# Standard library
from typing import Optional  # Used for type hints.
from subprocess import run  # Used to run zigzag from the command line.

##################################### CODE #####################################


def zigzag(
    settings: Optional[str] = None,
    mapping: Optional[str] = None,
    memory_pool: Optional[str] = None,
    architecture: Optional[str] = None,
) -> bool:
    """
    Runs the ZigZag framework with the provided input files.

    Arguments
    =========
     - settings: The settings file (--set) expected by ZigZag,
     - mapping: The mapping file (--map) expected by ZigZag,
     - memory_pool: The memory_pool file (--mempool) expected by ZigZag,
     - architecture: The architecture file (--arch) expected by ZigZag,
    
    Returns
    =======
    True if the ZigZag exited with code 0, False otherwise.
    """
    # We start by building the command string.
    command: List[str] = ["python3", "top_module.py"]
    # Then we add each parameter, if relevant.
    if settings is not None:
        command += ["--set", settings]
    if mapping is not None:
        command += ["--map", mapping]
    if memory_pool is not None:
        command += ["--mempool", memory_pool]
    if architecture is not None:
        command += ["--arch", architecture]

    # We run the expected commad and wait for its completion.
    completed_command = run(command)
    # Finally we return True if the exit code was 0, False otherwise.
    return completed_command.returncode == 0


##################################### MAIN #####################################

if __name__ == "__main__":
    # The actions to perform when this file if called as a script go here
    pass

##################################### EOF ######################################
