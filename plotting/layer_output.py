#!/usr/bin/env python3

#################################### HEADER ####################################

# File creation : Wed Sep 30 16:03:20 2020
# Language : Python3

################################### IMPORTS ####################################

# Standard library
from typing import Dict, Any  # Used for type hints.

# External imports
import yaml  # Used to deserialize the yaml output layer file.

##################################### CODE #####################################


class LayerOutput:
    """
    A class used to represent the output of ZigZag for a single layer.
    """

    def __init__(self, path: str):
        """
        Constructor of the LayerOutput class.

        Arguments
        =========
         - path: The path where the out[ut layer file is stored.
         
        Note
        ====
        The deserialization method will be picked from the file extension. As of
        now, the supported extensions are:
         - yaml

        Exceptions
        ==========
        If the file has an unrecognized extension, a ValueError will be raised.
        """
        # We build a dictionary holding the values of the output of ZigZag. We
        # pick the deserialization format based on the file extension.
        if path.endswith(".yaml"):
            with open(path, "r") as yaml_file:
                # We only need basic YAML loading here.
                self.dictionary: Dict[str, Any] = yaml.safe_load(yaml_file)
        else:
            # The extension is not recognized.
            raise ValueError("No known way to read the file {}".format(path))

    def __getitem__(self, key: str) -> Any:
        """
        Proxy to the LayerOutput.dictionary getitem method.

        Arguments
        =========
         - key: The key for which we want to get the value.

        Returns
        =======
        The value associated with the key.
        """
        return self.dictionary[key]


##################################### MAIN #####################################

if __name__ == "__main__":
    # The actions to perform when this file if called as a script go here
    pass

##################################### EOF ######################################
