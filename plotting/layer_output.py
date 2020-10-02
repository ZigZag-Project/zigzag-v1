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
         - path: The path where the output layer file is stored.
         
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

                # Set some of the layer properties based on path
                self.layer_number = int(path[path.find('_L')+2])
                self.layer_name = 'Layer %d' % self.layer_number
                self.memory_number = int(path[path.find('_M')+2])
                self.su_number = int(path[path.find('_SU')+3])
                self.opt_type = 'min_en' if 'min_en' in path else 'max_ut'

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

    def find(self, key, d=None):
        """
        Recursively finds the value of a key in LayerOutput.dictionary
        Assumes the key you provide is unique in the dictionary.

        Arguments
        =========
         - key: The key for which we want to get the value.

        Returns
        =======
        The value associated with the key.
        None if the key doesn't exist in the dictionary.
        """
        if d is None:
            d = self.dictionary
        if key in d:
            return d[key]
        for k, v in d.items():
            if isinstance(v, dict):
                found = self.find(key, v)
                if found is not None:
                    return found

##################################### MAIN #####################################

if __name__ == "__main__":
    # The actions to perform when this file if called as a script go here
    pass

##################################### EOF ######################################
