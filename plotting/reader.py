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
    Class dedicated to reading the output of ZigZag and providing the requested
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
        self.layer_numbers = []

        # SPECIAL CASE
        # If the provided path is a file, we try to load it directly.
        if os.path.isfile(path):
            # We try to infer the name of the layer from the path of the file.
            layer_name = os.path.basename(path)
            self.layers[layer_name] = LayerOutput(path)
            # Add layer number to list of seen layer numbers
            layer_number = self.layers[layer_name].layer_number
            if layer_number not in self.layer_numbers:
                self.layer_numbers.append(layer_number)
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
                        # Add layer number to list of seen layer numbers
                        layer_number = self.layers[layer_name].layer_number
                        if layer_number not in self.layer_numbers:
                            self.layer_numbers.append(layer_number)
                    except ValueError:
                        # This wasn't an expected file, we just skip it.
                        continue

        # Sort list of layer numbers
        self.layer_numbers.sort()

        # Save the memory hierarchy names
        # NOTE This only holds if all layers have the same hierarchy
        self.mem_hierarchy_labels = self.layers[layer_name].find('memory_name_in_the_hierarchy')

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

    def nested_energy(self) -> Any:
        """
        Returns the energy section of each loaded layer.
        Returned in a nested dictionary.
        
        Returns
        =======
        In concise output, the returned dictionary is structured like as:

        nested_energy (the returned dictionary)
          |
          |--'min_en','max_ut'
              |
              |--'Lx','Ly',...
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
        
        # Construct a nested dict
        nested_energy = {
        'min_en': {nb:None for nb in self.layer_numbers},
        'max_ut': {nb:None for nb in self.layer_numbers}
        }

        # Add energies to nested dict
        for layer_name, layer_output in self.layers.items():
            opt_type = layer_output.opt_type
            layer_number = layer_output.layer_number
            nested_energy[opt_type][layer_number] = layer_output.find('energy')

        return nested_energy
        
    def get_coarse_energy(self):
        """
        Returns the total and coarse grain energy of all loaded layer_outputs.
        
        Returns
        =======
        Two dictionaries:

        total_energy = {'min_en':[list of total energy], 
                        'max_ut':[list of total energy]
                        }
        coarse_energy = {
                        'min_en':[[list of mem energy], [list of mac energy]],
                        'max_ut':[[list of mem energy], [list of mac energy]]
                        }

        """
        # Get all layer numbers present in reader
        layer_numbers = self.layer_numbers

        # Get the nested energies
        nested_energy = self.nested_energy()

         # Get energy divisions 
        total_energy = {'min_en': [], 'max_ut': []}
        mac_energy = {'min_en': [], 'max_ut': []}
        mem_energy = {'min_en': [], 'max_ut': []}
        for layer_number in layer_numbers:
            total_energy_min_en = nested_energy['min_en'][layer_number]['total_energy']
            total_energy_max_ut = nested_energy['max_ut'][layer_number]['total_energy']
            mac_energy_min_en = nested_energy['min_en'][layer_number]['mac_energy']
            mac_energy_max_ut = nested_energy['max_ut'][layer_number]['mac_energy']
            mem_energy_min_en = total_energy_min_en - mac_energy_min_en
            mem_energy_max_ut = total_energy_max_ut - mac_energy_max_ut

            total_energy['min_en'].append(total_energy_min_en)
            total_energy['max_ut'].append(total_energy_max_ut)
            mac_energy['min_en'].append(mac_energy_min_en)
            mac_energy['max_ut'].append(mac_energy_max_ut)
            mem_energy['min_en'].append(mem_energy_min_en)
            mem_energy['max_ut'].append(mem_energy_max_ut)

        coarse_energy = {
            'min_en':[mem_energy['min_en'], mac_energy['min_en']],
            'max_ut':[mem_energy['max_ut'], mac_energy['max_ut']]
            }

        return total_energy, coarse_energy

    def get_fine_energy(self):
        """
        Returns the fine grain energy distr of all loaded layer_outputs.
        
        Returns
        =======
        One dictionary:

        fine_energy = {
                        'min_en':[
                            [list of i level 0 energy], 
                            [list of i level 1 energy],
                            ...
                            [list of w level 0 energy],
                            [list of w level 1 energy], 
                            ...
                            [list of o level 0 energy], 
                            [list of o level 1 energy],
                            ...
                            [list of mac energy]
                            ],
                        'max_ut':[
                            [list of i level 0 energy], 
                            [list of i level 1 energy],
                            ...
                            [list of w level 0 energy],
                            [list of w level 1 energy], 
                            ...
                            [list of o level 0 energy], 
                            [list of o level 1 energy],
                            ...
                            [list of mac energy]
                            ]
                        }
        If there are less memory levels for an operand, that list is omitted

        """
        # Get all layer numbers present in reader
        layer_numbers = self.layer_numbers

        # Get the nested energies
        nested_energy = self.nested_energy()

        # Get the number of memory levels for each operand
        # This is required to correctly initialize the nested lists inside of fine_energy
        # First layer present in layer_numbers used, but number of mem levels should stay constant
        en_breakdown_i = nested_energy['min_en'][layer_numbers[0]]['energy_breakdown']['I']
        en_breakdown_w = nested_energy['min_en'][layer_numbers[0]]['energy_breakdown']['W']
        en_breakdown_o = nested_energy['min_en'][layer_numbers[0]]['energy_breakdown']['O']
        n_mem_lvl_i = len(en_breakdown_i) 
        n_mem_lvl_w = len(en_breakdown_w)
        n_mem_lvl_o = len(en_breakdown_o)

        # Initialize fine_energy with the correct amount of nested lists
        # +1 for mac energy
        fine_energy = {
            'min_en': [[] for _ in range(n_mem_lvl_i + n_mem_lvl_w + n_mem_lvl_o + 1)], 
            'max_ut': [[] for _ in range(n_mem_lvl_i + n_mem_lvl_w + n_mem_lvl_o + 1)]
            }

        for layer_number in layer_numbers:

            # Proxy to the energy breakdown for this layer
            min_en_en_breakdown = nested_energy['min_en'][layer_number]['energy_breakdown']
            max_ut_en_breakdown = nested_energy['max_ut'][layer_number]['energy_breakdown']

            for i_idx in range(n_mem_lvl_i):
                min_en_en_i = min_en_en_breakdown['I'][i_idx]
                max_ut_en_i = max_ut_en_breakdown['I'][i_idx]
                fine_energy['min_en'][i_idx].append(min_en_en_i)
                fine_energy['max_ut'][i_idx].append(max_ut_en_i)
            i_idx += 1
            for w_idx in range(n_mem_lvl_w):
                min_en_en_i = min_en_en_breakdown['W'][w_idx]
                max_ut_en_i = max_ut_en_breakdown['W'][w_idx]
                fine_energy['min_en'][w_idx + i_idx].append(min_en_en_i)
                fine_energy['max_ut'][w_idx + i_idx].append(max_ut_en_i)
            w_idx += 1
            for o_idx in range(n_mem_lvl_o):
                min_en_en_i = min_en_en_breakdown['O'][o_idx]
                max_ut_en_i = max_ut_en_breakdown['O'][o_idx]
                fine_energy['min_en'][o_idx + w_idx + i_idx].append(min_en_en_i)
                fine_energy['max_ut'][o_idx + w_idx + i_idx].append(max_ut_en_i)


            mac_energy_min_en = nested_energy['min_en'][layer_number]['mac_energy']
            mac_energy_max_ut = nested_energy['max_ut'][layer_number]['mac_energy']

            fine_energy['min_en'][o_idx + w_idx + i_idx + 1].append(mac_energy_min_en)
            fine_energy['max_ut'][o_idx + w_idx + i_idx + 1].append(mac_energy_max_ut)


        return fine_energy

##################################### MAIN #####################################

if __name__ == "__main__":
    # The actions to perform when this file if called as a script go here
    pass

##################################### EOF ######################################
