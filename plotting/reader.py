#!/usr/bin/env python3

#################################### HEADER ####################################

# File creation : Wed Sep 30 15:57:38 2020
# Language : Python3

################################### IMPORTS ####################################

# Standard library
from typing import Dict, Any  # Used for type hints.
import os  # Used for directory exploration.

# Internal imports
from layer_output import (
    LayerOutput,
)  # Used to load the content of a layer file.

# NOTE
# I am assuming that python is being run from the current directory.

##################################### CODE #####################################


class Reader:
    """
    Class dedicated to readin the output of ZigZag and providing the requested
    values from it.
    """

    def __init__(self, path: str):
        """
        Constructor of the Reader class.

        Arguments
        =========
         - path: The path of the directory where the output of ZigZag was built.
        """
        # Sanity check, we verify that the provded path exists.
        assert os.path.exists(path)

        # We simply try to read a Layer from all the files within the given
        # directory.
        self.layers: Dict[str, LayerOutput] = dict()

        # SPECIAL CASE
        # If the provided path is a file, we try to load it directly.
        if os.path.isfile(path):
            # We try to infer the name of the layer from the path of the file.
            layer_name = os.path.basename(path)
            self.layers[layer_name] = LayerOutput(path)
        else:
            # We try to load any file in the directory.
            for element in os.listdir(path):
                # We get the full path of the element.
                element_path = os.path.join(path, element)
                if os.path.isfile(element_path):
                    # We have found a file, we try to load it.
                    try:
                        layer_name, _ = os.path.splitext(
                            os.path.basename(element_path)
                        )
                        self.layers[layer_name] = LayerOutput(element_path)
                    except ValueError:
                        # This wasn't an expected file, we just skip it.
                        continue

    def energy(self) -> Any:
        """
        Returns the energy section of each loaded layer.
        
        Returns
        =======
        In concise output, the returned dictionary is structured like as:

        energy (the returned dictionary)
        |
        |--total_energy: a float
        |
        |--mac_energy: a float
        |
        |--energy_breakdown
           |
           |--W: A list of floats
           |
           |--I: A list of floats
           |
           |--O: A list of floats

        Note
        ====
        Sorry for the loose type hints, But I can't correctly provide more than
        that.
        """
        # This can be done with a one-liner and dictionary comprehension.
        return {
            layer_name: layer_output["simulation"]["results"]["energy"]
            for layer_name, layer_output in self.layers.items()
        }


##################################### MAIN #####################################

if __name__ == "__main__":
    # The actions to perform when this file if called as a script go here
    pass

##################################### EOF ######################################
