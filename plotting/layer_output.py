#!/usr/bin/env python3

#################################### HEADER ####################################

# Copyright 2020 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause

# File creation : Wed Sep 30 16:03:20 2020
# Language : Python3

################################### IMPORTS ####################################

# Standard library
from typing import Dict, Optional, Any  # Used for type hints.
import os  # Used for filesystem path manipulations.
import re  # Used for regular expression matching.

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
         - path: The path where the output layer file is stored. The basename
            of this path should follow the template:
            
            "<arch>_<NN>_L<layer no>_M<memory no>_SU<su>_<type>.<ext>"

            Where:
             - "arch" is the name of the hardware architecture,
             - "NN" is the name of the neural network used,
             - "layer no" is the number of the layer used within the neural
                network,
             - "memory no" is ??
             - "su" is the index where the spatial unrolling has been performed,
             - "type" is the type of solution found, it should be one of
                "min_en" (minimum energy) or "max_ut" (maximum utilization or
                minimal latency).
             - "ext" is the extension of the file.
         
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

        # If we could load the content of the file, we parse the filename to get
        # some values which will be usefull for plotting.

        # First, we extract the base name of the file.
        filename, _ = os.path.splitext(os.path.basename(path))
        # We try to deduce the various fields we are looking for by
        # using a regular expression.
        template = "(.*)_(.*)_L(\d*)_M(\d*)_SU(\d*)_(min_en|max_ut)"
        matched = re.match(template, filename)
        # Then we assign our different groups to usefull field values. We
        # discard the name of the architecture and of the neural network. Though
        # it is not visually obvious, we are just unpacking the resulting tuple.
        (
            self.architecture,
            self.neural_network,
            number,
            memory,
            spatial_unrolling,
            self.optimum_type,
        ) = matched.groups()
        # We convert the string values to int before assigning them.
        self.number = int(number)
        self.memory = int(memory)
        self.spatial_unrolling = int(spatial_unrolling)
        # We also set a name for the layer.
        self.name = "Layer {}".format(self.number)

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

    def find(self, key: str) -> Optional[Any]:
        """
        Proxy to the find_function applied to the dictionary of values for this
        layer.

        Arguments
        =========
         - key: the key whose value we are searching.

        Returns
        =======
        The value associated with the provided key.

        Exceptions
        ==========
        If no value is found for the provided key, a KeyError will be raised.
        """
        # We recursively search for the value.
        recursive_found = find_function(key, self.dictionary)

        if recursive_found is not None:
            # We return the value we have found.
            return recursive_found
        else:
            raise KeyError(
                "No value associated with the key {} could be ".format(key)
                + "found for the layer {}.".format(self.name)
            )


def find_function(key: str, dictionary: Dict[str, Any]) -> Optional[Any]:
    """
    Recursively finds the value associated with the provided key in the given
    dictionary or any of its sub-dictionaries. The implementation is put inside
    of a function and not a method because it has no side effects.

    Arguments
    =========
     - key: The key that we are looking for,
     - dictionary: The dictionary in which we are looking for the key,
     - recursive: A boolean used within the recursion to know if the exception
        should be thrown. Default value is False and the end user should not
        need to touch it.

    Returns
    =======
    The value associated with the key if it found, None otherwise.

    Note
    ====
    The behavior of the function is undefined when the key can be found several
    time in the dictionary or its sub dictionaries. The first match will be
    returned, but it is not always obvious which value will be matched first.

    Side-effects
    ============
    This function has no side effects.
    """
    # Base case.
    if key in dictionary:
        return dictionary[key]
    else:
        # Recursive call.
        # We try all the elements in the dictionary to see if they we find
        # another nested dictionary which holds our key.
        for dictionary_element in dictionary.values():
            if isinstance(dictionary_element, dict):
                # We look fo the key in the nested dictionary.
                found = find_function(key, dictionary_element)
                if found is not None:
                    # We have found a value associated with the provided key,
                    # we return it as described in the docstring.
                    return found
                # else we kepp looking for our key.
    # If we reach this line, it means that the key can neither be found in this
    # dictionary nor any nested one. We thus return None.
    return None


##################################### MAIN #####################################

if __name__ == "__main__":
    # The actions to perform when this file if called as a script go here
    pass

##################################### EOF ######################################
