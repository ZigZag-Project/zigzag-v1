# Input settings setup
All the 4 files enlisted are required to run the framework
```
python3 top_module.py \
--set <path_to_settings_file> \
--map <path_to_mapping_file> \
--mempool <path_to_mempool_file> \
--arch <path_to_arch_file> \
```
## Setting file
The setting file (to be defined with the  ``--set`` flag) describes:
- Paths to the output file (``result_path``, ``result_filename``)
- The output file printing options: ```concise``` and ```complete```. Under ```concise``` mode, the tool prints the basic information for each optimal design point found, i.e. energy and performance; under ```complete``` mode, the tool prints all detailed information for each optimal design point found for in-depth analysis.
- General settings for the framework
  
  
  | | Fixed architecture | Fixed spatial unrolling | Fixed temporal mapping |
  |:----:|:----:|:----:|:----:|
  | Single cost estimation | True | True | True |
  | Temporal mapping exploration | True | True | False |
  | Temporal and spatial mapping exploration | True | False | False |
  | Complete DSE | False | False | False |
  
  The fixed temporal mapping and spatial unrolling can be defined in the mapping file. See [**Mapping**](#mapping-file)<!-- @IGNORE PREVIOUS: anchor -->
  
  If the temporal mapping exploration is enabled, the workload and the precision of the operands has to be specified, beside the fixed architecture and fixed spatial unrolling.
  
  If spatial unrolling exploration is enabled, different schemes may/may not be defined in the mapping file. See [**Mapping**](#mapping-file)<!-- @IGNORE PREVIOUS: anchor -->
  
  If Full DSE is enabled, the settings for the architecture genrator (max area, area utilization, mem_ratio ...) must be defined, beside the PE array size (in the architecture file) and the memory pool.
 
- Workload definition: The path to the python file (under the [**NN_layers**](../NN_layers) folder) that contains the description of each layer (```layer_filename```) and the list of layers (```layer_indices```) to be analyzed if exploration is carried out.

- Temporal mapping exploration settings: ```search_method``` can be either defined ```exhaustive```, ```heuristic v1```, ```heuristic v2```, or ```iterative```.
- Spatial unrolling exploration settings: ```search_method``` can be either defined ```exhaustive```, ```heuristic v1```, ```heuristic v2```, or ```hint_driven```. For first three search methods, 'spatial_utilization_threshold' need to be defined. For the last search method, 'spatial_mapping_list' in the mapping file is used, and 'spatial_utilization_threshold' is ignored.

- Multiprocessing settings: If the workstation/server has multiple cores, multiple design points can be evaluated in parallel to speed up the search procedure.
  - ```layer_multiprocessing``` defines how many elements of the ```layer_indices``` list will be evaluated in parallel.
  - ```architecture_multiprocessing``` defines how many hardware architectures are evaluated in parallel.
  - ```spatial_unrolling_multiprocessing``` defines how many spatial unrolling schemes are evaluated in parallel.

## Architecture file
The architecture settings file (to be defined with the  ``--arch`` flag) describes:
- The characteristics of the PE array (```PE_array/nrows``` and ```PE_array/ncols```, the cost of a single mac operation, the precision ...);
- A list of memory instances given as a hint (memory scheme hint);
- A list of memory instances that define the fixed architecture (in memory_hierarchy). They will be considered only if the ```fixed_architecture``` flag in the settings file is set to True;
- The settings for the architecture generator.

A single memory instance is defined by its name, and a set of enlisted characteristics:
  - ```memory_instance```: refers to the element in the memory pool which is taken;
  - ```bank_instances```: refers to the number of *separate* instances present in the architecture;
  - ```operand_stored```: contains a list of the operands that the instance contains. The operands are defined as I (inputs), W (weights), O (outputs).

 
The PE array characteristics describe:
 - The size of the PE array (``nrows`` and ``ncols``) ;
 - The precision of each operand in bits. The precision of the partial sums is described by ```O_partial```, while the precision of the final outputs as ```O_final```;
 - The cost of a single MAC (```single_mac_energy```);
 - The characteristics of the systolic flow of data in the array: 0 is horizontal, 1 is vertical, 2 is diagonal. It affects the utilization and eventual stallings within the array. Leave it empty if the array is not working under the systolic mode. 
 

The settings for the architecture generator are used *only* when the ``fixed_architecture`` parameter in the settings file is set to False.
The architecture generator requires:
 - ``max_area`` defines the total core area;
 - ``area_utilization`` defines the percentage (0-1) of the max area that has to be utilized;
 - ``mem_ratio`` defines the minimum ratio between two adjacent levels in the memory hierarchy;
 - ``PE_memory_depth`` defines the max number of memory levels in the hierarchy inside the PEs.

and a list of sizes for the memory levels inside each PE, and for L1 and L2: 
 - ``PE_threshold`` indicates the maximum memory size that can be stored inside each PE.
 - ``L1_size`` and ``L2_size`` define the list of L1 and L2 sizes to be explored. Each size listed is expressed as a banked combination of elements drawn from the memory pool. L1 and L2 are not necessity. The tool can explore/represent memory hierarchies with any number of memory level.
 ## Mapping file
The Mapping description settings file (to be defined with the  ``--mac`` flag) describes the the fixed temporal mapping and the fixed spatial unrolling (if their respective flag is set in the settings file)

The mappings are defined for each operand *separately*

The nested loops are seprated by the levels that they belong to. The levels are indexed from the innermost in the hierarchy (level 0) to the outermost (most cases the largest on-chip SRAM or the off-chip DRAM).

Within each level, the nested loops are enlisted from the innermost to the outermost. Each nested loop is defined as ```[loop_type, loop_size]``` where ```loop_type``` can FX (filter width), FY (filter height), OX (output width), OY (output height), C (input channel), K (output channel), B (batch size).

A level in the hierarchy which does not hold any loop blocking is left empty (```[]```).

If the fixed spatial mapping is not set, the ```spatial_unrolling_list``` may/may not be defined depending on the search mode selected.

The spatial unrolling list contain a list of spatial mappings to be evaluated separately. Each spatial mapping is defined as two lists: one for the loop types (FX,FY,OX,...) unrolled along array column dimension and another for the loop types unrolled across array column dimension.





 
