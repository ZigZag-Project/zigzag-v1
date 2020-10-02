#!/usr/bin/env python3

#################################### HEADER ####################################

# File creation : Wed Sep 30 15:57:38 2020
# Language : Python3

################################### IMPORTS ####################################

# Standard library
from typing import Dict, List, Tuple, Any  # Used for type hints.
import os  # Used for directory exploration.
from collections import (
    defaultdict,
)  # Used to create nested dictionaries easily.

# Internal imports
from layer_output import (
    LayerOutput,
)  # Used to load the content of a layer file.

# NOTE
# I am assuming that python is being run from the current directory.

##################################### CODE #####################################


class Reader:
    """
    Class dedicated to reading the output of ZigZag and providing the requested
    values from it.
    """

    def __init__(self, path: str):
        """
        Constructor of the Reader class.

        Arguments
        =========
         - path: The path of the directory where the output of ZigZag was built.

        Exceptions
        ==========
        If the path of a single file is provided, and that file is not a
        recognized ZigZag output, a ValueError will be raised.
        If some output files use a different hardware architecture than others,
        an AssertionError will be raised.
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

        # We store the sorted list of all our layer numbers, without redundancy.
        self.layer_numbers = sorted(
            {layer.number for layer in self.layers.values()}
        )

        # We try to save the labels of the memory elements in the hierarchy. For
        # this to work, all the provided layers must use the same hardware.
        #
        # We grab the labels for one of the layers.
        self.memory_labels = next(iter(self.layers.values())).find(
            "memory_name_in_the_hierarchy"
        )
        # # We assert that all the layers use the same hardware.
        # # NOTE
        # # This is not very efficient, because we use find in a loop which will
        # # always use the same access path. But it works.
        # assert all(
        #     self.memory_labels == layer.find("memory_name_in_the_hierarchy")
        #     for layer in self.layers
        # )

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

    def nested_energy(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Returns the energy section of each loaded layer in a nested dictionary.
        
        Returns
        =======
        In concise output, the returned dictionary is structured like as:

        nested_energy (the returned dictionary)
        |
        |--'min_en','max_ut'
            |
            |--'Lx','Ly',...  (the layer number, an integer)
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
        """
        # The dictionaries that we are going to return, we use a defaultdict to
        # make the construction easier.
        nested_energy: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)

        # Add energies to nested dict.
        for layer in self.layers.values():
            nested_energy[layer.optimum_type][layer.number] = layer.find(
                "energy"
            )

        return nested_energy

    def coarse_energy(self) -> Tuple[Dict[str, List[float]]]:
        """
        Returns the total and coarse grain energy of all loaded ZigZag outputs.
        
        Returns
        =======
        A tuple of two dictionaries:

        total_energy = {
                            'min_en':[list of total energy], 
                            'max_ut':[list of total energy]
                       }

        coarse_energy = {
                            'min_en': [
                                        [list of mem energy],
                                        [list of mac energy]
                                      ]
                            'max_ut': [
                                        [list of mem energy],
                                        [list of mac energy]
                                      ]
                        }
        """
        # Get the nested energies
        nested_energy = self.nested_energy()

        # Get energy divisions
        total_energy: Dict[str, List[float]] = dict()
        mac_energy: Dict[str, List[float]] = dict()
        memory_energy: Dict[str, List[float]] = dict()

        # We use a list comprehension to build the lists of total energy used,
        # energy used by macs and by memory for the minimum energy and maximum
        # utilization. We start by grabbing the values in a list of tuples.
        #
        # Total energy
        total_energy["min_en"] = [
            layer_energy["total_energy"]
            for layer_energy in nested_energy["min_en"].values()
        ]
        total_energy["max_ut"] = [
            layer_energy["total_energy"]
            for layer_energy in nested_energy["max_ut"].values()
        ]

        # MAC energy
        mac_energy["min_en"] = [
            layer_energy["mac_energy"]
            for layer_energy in nested_energy["min_en"].values()
        ]
        mac_energy["max_ut"] = [
            layer_energy["mac_energy"]
            for layer_energy in nested_energy["max_ut"].values()
        ]

        # Memory energy
        memory_energy["min_en"] = [
            layer_energy["total_energy"] - layer_energy["mac_energy"]
            for layer_energy in nested_energy["min_en"].values()
        ]
        memory_energy["max_ut"] = [
            layer_energy["total_energy"] - layer_energy["mac_energy"]
            for layer_energy in nested_energy["max_ut"].values()
        ]

        # We also reorganize our values to make the coarse energy dictionary.
        coarse_energy = {
            "min_en": [memory_energy["min_en"], mac_energy["min_en"]],
            "max_ut": [memory_energy["max_ut"], mac_energy["max_ut"]],
        }

        return total_energy, coarse_energy

    def fine_energy(self) -> Dict[str, List[List[float]]]:
        """
        Returns the fine grain energy distribution of all loaded ZigZag outputs.
        
        Returns
        =======
        One dictionary:

        fine_energy = {
                        'min_en': [
                                    [list of inputs level 0 energy], 
                                    [list of inputs level 1 energy],
                                    ...
                                    [list of weights level 0 energy],
                                    [list of weights level 1 energy], 
                                    ...
                                    [list of outputs level 0 energy], 
                                    [list of outputs level 1 energy],
                                    ...
                                    [list of mac energy]
                                  ],
                        'max_ut': [
                                    [list of inputs level 0 energy], 
                                    [list of inputs level 1 energy],
                                    ...
                                    [list of weigths level 0 energy],
                                    [list of weigths level 1 energy], 
                                    ...
                                    [list of outputs level 0 energy], 
                                    [list of outputs level 1 energy],
                                    ...
                                    [list of mac energy]
                                 ]
                        }

        Note
        ====
        If there are less memory levels for an operand, the corresponding list
        is omitted.
        Also, the level 0 of the memory is the DRAM. The higher the memory
        level, the closer to the processing element it gets.
        """
        # Get the nested energies.
        nested_energy = self.nested_energy()

        # We build our fine energy dictionary.
        fine_energy: Dict[str, List[List[float]]] = dict()

        ######### MINIMUM ENERGY

        # We build the list of input level energy.
        inputs_level_energy_for_minimum_energy_transposed = [
            layer_energy["energy_breakdown"]["I"]
            for layer_energy in nested_energy["min_en"].values()
        ]
        # This looks like:
        # [[list of level energies for input 0], [list of level energies for
        # input 1], ...].
        # We are now going to transpose this list of list, this can be done
        # using the zip function and the star operator. We end up with a list
        # of tuples, but it should be enough for our purposes.
        inputs_level_energy_for_minimum_energy = list(
            zip(*inputs_level_energy_for_minimum_energy_transposed)
        )
        # We can do the same for the weights, except that we fuse the two steps
        # into one.
        weights_level_energy_for_minimum_energy = list(
            zip(
                *[
                    layer_energy["energy_breakdown"]["W"]
                    for layer_energy in nested_energy["min_en"].values()
                ]
            )
        )

        outputs_level_energy_for_minimum_energy = list(
            zip(
                *[
                    layer_energy["energy_breakdown"]["O"]
                    for layer_energy in nested_energy["min_en"].values()
                ]
            )
        )
        # We also need our mac energy to put it at the end of the list.
        mac_energy_for_minimum_energy = tuple(
            layer_energy["mac_energy"]
            for layer_energy in nested_energy["min_en"].values()
        )
        # We can store the concatenation of the four list in out dictionary. We
        # have to turn our mac energy into a tuple to get an homogeneous data
        # structure.
        fine_energy["min_en"] = (
            inputs_level_energy_for_minimum_energy
            + weights_level_energy_for_minimum_energy
            + outputs_level_energy_for_minimum_energy
            + [tuple(mac_energy_for_minimum_energy)]
        )

        ######### MAXIMUM UTILIZATION

        # We repeat the process for the maximum utilization.
        inputs_level_energy_for_maximum_utilization = list(
            zip(
                *[
                    layer_energy["energy_breakdown"]["I"]
                    for layer_energy in nested_energy["max_ut"].values()
                ]
            )
        )
        # We still use zip to transpose our lists of lists.
        weights_level_energy_for_maximum_utilization = list(
            zip(
                *[
                    layer_energy["energy_breakdown"]["W"]
                    for layer_energy in nested_energy["max_ut"].values()
                ]
            )
        )
        outputs_level_energy_for_maximum_utilization = list(
            zip(
                *[
                    layer_energy["energy_breakdown"]["O"]
                    for layer_energy in nested_energy["max_ut"].values()
                ]
            )
        )
        # We also need our mac energy to put it at the end of the list.
        mac_energy_for_maximum_utilization = tuple(
            layer_energy["mac_energy"]
            for layer_energy in nested_energy["max_ut"].values()
        )
        # We can store the concatenation of the four list in out dictionary. We
        # have to turn our mac energy into a tuple to get an homogeneous data
        # structure.
        fine_energy["max_ut"] = (
            inputs_level_energy_for_maximum_utilization
            + weights_level_energy_for_maximum_utilization
            + outputs_level_energy_for_maximum_utilization
            + [tuple(mac_energy_for_maximum_utilization)]
        )

        return fine_energy


##################################### MAIN #####################################

if __name__ == "__main__":
    # The actions to perform when this file if called as a script go here
    pass

##################################### EOF ######################################
