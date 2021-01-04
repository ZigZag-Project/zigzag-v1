#!/usr/bin/env python3

#################################### HEADER ####################################

# Copyright 2020 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause

# File creation : Wed Sep 30 15:57:38 2020
# Language : Python3

################################### IMPORTS ####################################

# Standard library
from typing import Dict, List, Tuple, Any, Optional  # Used for type hints.
import os  # Used for directory exploration.
from collections import (
    defaultdict,
)  # Used to create nested dictionaries easily.

# External imports
import pandas as pd  # Used for analysis.

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

    def __init__(self, *paths: str):
        """
        Constructor of the Reader class.

        Arguments
        =========
         - paths: The paths of the directories or files where the output of
            ZigZag is located. This is a variadic argument, i.e. several paths
            can be given at once.

        Exceptions
        ==========
        If the path of a single file is provided, and that file is not a
        recognized ZigZag output, a ValueError will be raised.
        """
        # We simply try to read a Layer from all the files within the given
        # directory.
        self.layers: Dict[str, LayerOutput] = dict()

        for path in paths:
            # SPECIAL CASE
            # If the provided path is a file, we try to load it directly.
            if os.path.isfile(path):
                # We try to infer the name of the layer from the path of the
                # file.
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

    def nested_latency(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Returns the latency section of each loaded layer in a nested dictionary.
        
        Returns
        =======
        In concise output, the returned dictionary is structured like as:

        nested_latency (the returned dictionary)
        |
        |--'min_en','max_ut'
            |
            |--'Lx','Ly',...  (the layer number, an integer)
                |
                |--latency_cycle_with_data_loading: a float
                |
                |--latency_cycle_without_data_loading: a float
                |
                |--ideal_computing_cycle: a float
                |
                |-- ...
        """
        # The dictionaries that we are going to return, we use a defaultdict to
        # make the construction easier.
        nested_latency: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)

        # Add latencies to nested dict.
        for layer in self.layers.values():
            nested_latency[layer.optimum_type][layer.number] = layer.find(
                "latency"
            )

        return nested_latency

    def total_latency(self):
        """
        Returns the total latency of all loaded layers in a nested dictionary.
        Total latency is the total number of latency cyles with data loading.

        Returns
        =======
        In concise output, the returned dictionary is structured like as:

        total_latency (the returned dictionary)
        |
        |--'min_en','max_ut'
            |
            |--total_latency_with_data_loading: a float
        """

        # Get the nested latency
        nested_latency = self.nested_latency()

        # Create the dictionary
        total_latency: Dict[str, float] = dict()

        total_latency["min_en"] = [
            layer_latency["latency_cycle_with_data_loading"]
            for layer_latency in nested_latency["min_en"].values()
        ]

        total_latency["max_ut"] = [
            layer_latency["latency_cycle_with_data_loading"]
            for layer_latency in nested_latency["max_ut"].values()
        ]

        return total_latency

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
        # NOTE from Arne: Do we get values in correct order as dict is orderless?
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
            layer_energy["mac_energy"]["active"]
            + layer_energy["mac_energy"]["idle"]
            for layer_energy in nested_energy["min_en"].values()
        ]
        mac_energy["max_ut"] = [
            layer_energy["mac_energy"]["active"]
            + layer_energy["mac_energy"]["idle"]
            for layer_energy in nested_energy["max_ut"].values()
        ]

        # Memory energy
        memory_energy["min_en"] = [
            layer_energy["total_energy"]
            - layer_energy["mac_energy"]["active"]
            - layer_energy["mac_energy"]["idle"]
            for layer_energy in nested_energy["min_en"].values()
        ]
        memory_energy["max_ut"] = [
            layer_energy["total_energy"]
            - layer_energy["mac_energy"]["active"]
            - layer_energy["mac_energy"]["idle"]
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
            layer_energy["mac_energy"]["active"]
            + layer_energy["mac_energy"]["idle"]
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
            layer_energy["mac_energy"]["active"]
            + layer_energy["mac_energy"]["idle"]
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

    def memory(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Returns a view of all the layers in this object centered on their memory
        hierarchy.

        Returns
        =======
        <root>  # The dictionary itself
        |
        |--(layer_name) # The name associated with the layer
           |
           |--(data_type)   # One of "W", "I", "O"
              |
              |--(memory_name)  # The name of the memory element in this context
                 |
                 |--(key) # Where key is the name of the information you want to
                                    retrieve for the given memory element.

        Possible keys
        =============
         - name: The name of the memory element,
         - size: *self-explanatory*,
         - word_length: *self-explanatory*, this yields a list,
         - energy_per_access: *self-explanatory*, this yields a list,
         - type: *self-explanatory*
         - area_module: The area of a single module of this memory,
         - unrolling: ??, original name is "memory_urolling",
         - temporal_mapping: The temporal mapping associated with this element,
         - energy_breakdown: Energy consumed in this memory for the given
            mapping,
        """
        # We build our returned dict iteratively for more readability.
        returned_view: Dict[str, Dict[str, Dict[str, Any]]] = dict()

        for layer_name, layer_output in self.layers.items():
            view_layer: Dict[str, Dict[str, Any]] = dict()

            # We have to work on each data type (I for inputs, W for weights, O
            # for outputs.
            for data_type in ["W", "I", "O"]:
                # We create a dictionary for the data type.
                view_data_type: Dict[str, Any] = dict()

                # We gather the data for all the relevant memory elements.
                for index, memory_name in enumerate(
                    layer_output["simulation"]["hardware_specification"][
                        "memory_hierarchy"
                    ]["memory_name_in_the_hierarchy"][data_type]
                ):
                    # We build a dict of fields for of our memory element.
                    view_memory: Dict[str, Any] = dict()
                    # We add the memory name to the list.
                    view_memory["name"] = memory_name
                    # We get the size of the memory.
                    memory_size = layer_output["simulation"][
                        "hardware_specification"
                    ]["memory_hierarchy"]["memory_size_bit"][data_type][index]
                    view_memory["size"] = memory_size

                    # We get the word length of the memory.
                    memory_word_length = layer_output["simulation"][
                        "hardware_specification"
                    ]["memory_hierarchy"]["memory_word_length"][data_type][
                        index
                    ]
                    view_memory["word_length"] = memory_word_length

                    # We get the energy access per word.
                    memory_energy_per_access = layer_output["simulation"][
                        "hardware_specification"
                    ]["memory_hierarchy"]["memory_access_energy_per_word"][
                        data_type
                    ][
                        index
                    ]
                    view_memory["energy_per_access"] = memory_energy_per_access

                    # We get the memory type.
                    memory_type = layer_output["simulation"][
                        "hardware_specification"
                    ]["memory_hierarchy"]["memory_type"][data_type][index]
                    view_memory["type"] = memory_type

                    # We get the area of a module.
                    memory_area_module = layer_output["simulation"][
                        "hardware_specification"
                    ]["memory_hierarchy"]["memory_area_single_module"][
                        data_type
                    ][
                        index
                    ]
                    view_memory["area_module"] = memory_area_module

                    # We get the unrolling of the memory.
                    memory_unrolling = layer_output["simulation"][
                        "hardware_specification"
                    ]["memory_hierarchy"]["memory_unrolling"][data_type][index]
                    view_memory["memory_unrolling"] = memory_area_module

                    # We get the temporal mapping of the memory. This is a big
                    # list.
                    memory_temporal_mapping = layer_output["simulation"][
                        "results"
                    ]["basic_information"]["temporal_mapping"][data_type][index]
                    view_memory["temporal_mapping"] = memory_temporal_mapping

                    # We get the energy breakdown forthis memory.
                    memory_energy_breakdown = layer_output["simulation"][
                        "results"
                    ]["energy"]["energy_breakdown"][data_type][index]
                    view_memory["energy_breakdown"] = memory_energy_breakdown

                    # We add the values we gathered to the view of the data type.
                    view_data_type[memory_name] = view_memory

                # We add the built dictionary for this data type to the layer
                # dictionary.
                view_layer[data_type] = view_data_type

            # Finally, we add the view of the layer to the global view.
            returned_view[layer_name] = view_layer

        # We return the built view.
        return returned_view

    def flatten(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame of all the ZigZag outputs.

        IMPORTANT
        =========
        This call will fail on the concise input.

        Returns
        =======
        A pandas dataframe with high-level information about the different runs.

        Exceptions
        ==========
        When used on a concise output, some values cannot be found, which yields
        a KeyError (I believe).
        """
        # We use list comprehension to gather all the data for pandas.
        pandas_data = [
            [
                layer_full_name,
                layer.architecture,
                layer.neural_network,
                layer.number,
                layer.memory,
                layer.spatial_unrolling,
                layer.optimum_type,
                layer.find("total_MAC_operation"),
                layer.find("total_energy"),
                # Accounting for MAC idle energy.
                layer.find("mac_energy")["active"]
                + layer.find("mac_energy")["idle"],
                layer.find("utilization_with_data_loading"),
                layer.find("utilization_without_data_loading"),
                layer.find("utilization_spatial"),
                layer.find("utilization_temporal_with_data_loading"),
                layer.find("utilization_temporal_without_data_loading"),
                layer.find("latency_cycle_with_data_loading"),
                layer.find("latency_cycle_without_data_loading"),
                layer.find("ideal_computing_cycle"),
                layer.find("load_cycle_total"),
                layer.find("memory_stalling_cycle_count"),
                layer.find("area"),
            ]
            for layer_full_name, layer in self.layers.items()
        ]

        # The labels for the DataFrame.
        pandas_labels = [
            "full_name",
            "architecture",
            "neural_network",
            "layer_number",
            "memory_number",
            "spatial_unrolling",
            "optimum_type",
            "total_MAC_operation",
            "total_energy",
            "mac_energy",
            "utilization_with_data_loading",
            "utilization_without_data_loading",
            "utilization_spatial",
            "utilization_temporal_with_data_loading",
            "utilization_temporal_without_data_loading",
            "latency_cycle_with_data_loading",
            "latency_cycle_without_data_loading",
            "ideal_computing_cycle",
            "load_cycle_total",
            "memory_stalling_cycle_count",
            "area",
        ]

        # We build and return the DataFrame.
        return pd.DataFrame(pandas_data, columns=pandas_labels).set_index(
            "full_name"
        )

    def memory_flatten(
        self, architecture: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Returns a view of all the layers in this object centered on their memory
        hierarchy.

        Arguments
        =========
         - architecture: The architecture for which the analysis should be
            performed. The different Zigzag run must have the same architecture
            for the pandas DataFrame to be built, and this parameter enables
            filtering in case more runs have been loaded. When unspecified, all
            the layers will be used.

        Returns
        =======
        A doubly indexed DatFrame. The first index is the full name of the run,
        the second name is the name of the memory considered. The columns are:
         - size
         - read_length
         - write_length
         - read_cost
         - write_cost
         - type
         - unrolling
         - utilization
         - effective_size
         - energy
         - load_cycle
         - read_stall
         - write_stall
         - required_read_bandwidth
         - required_write_bandwidth
        """
        if architecture is not None:
            filtered = {
                layer_name: layer
                for layer_name, layer in self.layers.items()
                if layer.architecture == architecture
            }
        else:
            filtered = self.layers
        return memory_flat(filtered)


def memory_flat(layers: Dict[str, LayerOutput]) -> pd.DataFrame:
    """
    Returns a view of all the layers in this object centered on their memory
    hierarchy.

    Returns
    =======
    A doubly indexed DatFrame. The first index is the full name of the run, the
    second name is the name of the memory considered. The columns are:
     - size
     - read_length
     - write_length
     - read_cost
     - write_cost
     - type
     - unrolling
     - utilization
     - effective_size
     - energy
     - load_cycle
     - read_stall
     - write_stall
     - required_read_bandwidth
     - required_write_bandwidth
    """
    # We build our returned dict iteratively for more readability.
    returned_view: Dict[str, Dict[str, Dict[str, Any]]] = dict()

    for layer_name, layer_output in layers.items():
        view_layer: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # We have to work on each data type (I for inputs, W for weights, O
        # for outputs.
        for data_type in ["W", "I", "O"]:
            # We gather the data for all the relevant memory elements.
            for index, memory_name in enumerate(
                layer_output["simulation"]["hardware_specification"][
                    "memory_hierarchy"
                ]["memory_name_in_the_hierarchy"][data_type]
            ):
                # We build a dict of fields for of our memory element.
                view_memory = view_layer[memory_name]

                ### SIZE
                memory_size = layer_output["simulation"][
                    "hardware_specification"
                ]["memory_hierarchy"]["memory_size_bit"][data_type][index]
                if "size" not in view_memory:
                    view_memory["size"] = memory_size
                else:
                    # Cumulating the memory size amoung the operands.
                    view_memory["size"] += memory_size

                ### WORD LENGTH
                memory_word_length = layer_output["simulation"][
                    "hardware_specification"
                ]["memory_hierarchy"]["memory_word_length"][data_type][index]
                # There are two values for the word length, one for read and
                # one for write.
                if "read_length" not in view_memory:
                    view_memory["read_length"] = memory_word_length[0]
                if "write_length" not in view_memory:
                    view_memory["write_length"] = memory_word_length[1]

                ### ACCESS COST
                memory_energy_per_access = layer_output["simulation"][
                    "hardware_specification"
                ]["memory_hierarchy"]["memory_access_energy_per_word"][
                    data_type
                ][
                    index
                ]
                if "read_cost" not in view_memory:
                    view_memory["read_cost"] = memory_energy_per_access[0]
                if "write_cost" not in view_memory:
                    view_memory["write_cost"] = memory_energy_per_access[1]

                ### MEMORY TYPE
                memory_type = layer_output["simulation"][
                    "hardware_specification"
                ]["memory_hierarchy"]["memory_type"][data_type][index]
                if "type" not in view_memory:
                    view_memory["type"] = memory_type

                ### UNROLLING
                memory_unrolling = layer_output["simulation"][
                    "hardware_specification"
                ]["memory_hierarchy"]["memory_unrolling"][data_type][index]
                if "unrolling" not in view_memory:
                    view_memory["unrolling"] = memory_unrolling

                ### UTILIZATION
                # NOTE
                # Only the shared value is interesting for us since we are
                # grouping the memories by name without taking care of the
                # operations.
                memory_utilization = layer_output["simulation"]["results"][
                    "basic_information"
                ]["actual_memory_utilization_shared"][data_type][index]
                if "utilization" not in view_memory:
                    view_memory["utilization"] = memory_utilization

                ### EFFECTIVE SIZE
                memory_effective_size = layer_output["simulation"]["results"][
                    "basic_information"
                ]["effective_memory_size"][data_type][index]
                if "effective_size" not in view_memory:
                    view_memory["effective_size"] = memory_effective_size

                ### ENERGY
                memory_energy_breakdown = layer_output["simulation"]["results"][
                    "energy"
                ]["energy_breakdown"][data_type][index]
                if "energy" not in view_memory:
                    view_memory["energy_breakdown"] = memory_energy_breakdown
                else:
                    # We accumulate the energy spent for all the operands.
                    view_memory["energy_breakdown"] += memory_energy_breakdown

                ### LOAD CYCLE
                if data_type != "O":
                    # The output is not loaded from memory but computed.
                    memory_load_cycle = layer_output["simulation"]["results"][
                        "performance"
                    ]["latency"]["data_loading"]["load_cycle_individual"][
                        data_type
                    ][
                        index
                    ]
                    if "load_cycle" not in view_memory:
                        view_memory["load_cycle"] = memory_load_cycle
                    else:
                        view_memory["load_cycle"] += memory_load_cycle
                else:
                    if "load_cycle" not in view_memory:
                        # I don't really know which value would make more sense
                        # here, so I am putting a NaN.
                        view_memory["load_cycle"] = float("NaN")

                ### STALLING
                memory_stall = layer_output["simulation"]["results"][
                    "performance"
                ]["latency"]["memory_stalling"]["memory_stalling_cycle_shared"][
                    data_type
                ][
                    index
                ]
                # Index 0 is treated differently, for some reason it only has a
                # single value in the YAML output instead of two.
                if len(memory_stall) == 1:
                    if "read_stall" not in view_memory:
                        view_memory["read_stall"] = memory_stall[0]
                    else:
                        # We accumulate the stalling cycles, even if doesn't
                        # completely makes sense.
                        view_memory["read_stall"] += memory_stall[0]
                    # Because I don't really know which value to put here I put
                    # a NaN.
                    view_memory["write_stall"] = float("NaN")
                else:
                    if "read_stall" not in view_memory:
                        view_memory["read_stall"] = memory_stall[0]
                    else:
                        # We accumulate the stalling cycles, even if doesn't
                        # completely makes sense.
                        view_memory["read_stall"] += memory_stall[0]
                    if "write_stall" not in view_memory:
                        view_memory["write_stall"] = memory_stall[1]
                    else:
                        # We accumulate the stalling cycles, even if doesn't
                        # completely makes sense.
                        view_memory["write_stall"] += memory_stall[1]

                ### BANDWIDTH
                memory_required_bandwidth = layer_output["simulation"][
                    "results"
                ]["performance"]["latency"]["memory_stalling"][
                    "required_memory_bandwidth_per_cycle_shared"
                ][
                    data_type
                ][
                    index
                ]
                if "required_read_bandwidth" not in view_memory:
                    view_memory[
                        "required_read_bandwidth"
                    ] = memory_required_bandwidth[0]
                if "required_write_bandwidth" not in view_memory:
                    view_memory[
                        "required_write_bandwidth"
                    ] = memory_required_bandwidth[1]

        # Finally, we add the view of the layer to the global view.
        returned_view[layer_name] = view_layer

    # Used to build the multi indexed labels of the DataFrame.
    labels: List[List[str]] = [
        [
            layer_name
            for layer_name, layer in returned_view.items()
            for _ in layer
        ],
        [
            memory_name
            for layer in returned_view.values()
            for memory_name in layer
        ],
    ]

    # The names to use for all the columns of the DataFrame.
    columns = [
        "size",
        "read_length",
        "write_length",
        "read_cost",
        "write_cost",
        "type",
        "unrolling",
        "utilization",
        "effective_size",
        "energy",
        "load_cycle",
        "read_stall",
        "write_stall",
        "required_read_bandwidth",
        "required_write_bandwidth",
    ]

    # We gather the data for the panda array in a list of lists.
    pandas_data = [
        memory.values()
        for layer in returned_view.values()
        for memory in layer.values()
    ]

    # We return the built view.
    data_frame = pd.DataFrame(pandas_data, index=labels, columns=columns)
    data_frame.index = data_frame.index.set_names(["full_name", "memory"])
    return data_frame


##################################### MAIN #####################################

if __name__ == "__main__":
    # The actions to perform when this file if called as a script go here
    pass

##################################### EOF ######################################
