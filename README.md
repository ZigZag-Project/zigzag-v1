# ZigZag: A Joint Architecture-Mapping Design Space Exploration Framework for DNN Accelerators

## Paper
Old version: https://arxiv.org/abs/2007.11360

New version: https://ieeexplore.ieee.org/document/9360462


## Functions of ZigZag
**Supported Mode**|**Function**|**Memory Scheme Search**|**Spatial Unrolling Search**|**Temporal Mapping Search**|**Cost Estimation**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
1|Hardware Cost Evaluation|No|No|No|Yes|Publicly released
2|Find the best Temporal Mapping (minimize energy or maximize throughput) for a single NN layer|No|No|Yes|Yes
3|Find the best Spatial Unrolling and the best Temporal Mapping  for a single NN layer|No|Yes|Yes|Yes
4|Find the best Memory Scheme with fixed Spatial Unrolling for a single NN layer|Yes|No|Yes|Yes
5|Find the best Memory Scheme with the best Spatial Unrolling for a single NN layer|Yes|Yes|Yes|Yes
6|Find the best Spatial Unrolling for multiple NN layers (could be a complete NN or multiple NNs)|No|Yes|Yes|Yes
7|Find the best Memory Scheme with fixed Spatial Unrolling for multiple NN layers (could be a complete NN or multiple NNs)|Yes|No|Yes|Yes
8|Find the best Memory Scheme with the best Spatial Unrolling for multiple NN layers (could be a complete NN or multiple NNs)|Yes|Yes|Yes|Yes
## Quickstart
To run the framework
```
python3 top_module.py \
--set <path_to_settings_file> \
--map <path_to_mapping_file> \
--mempool <path_to_mempool_file> \
--arch <path_to_arch_file> \
```

## Examples

A few input setting files are contained in the ``inputs`` folder.
### Single cost estimation
In the example provided a single cost estimation is carried out for the inference of CONV4 of AlexNet on Eyeriss.

In the settings file (``inputs/settings.yaml``) the architecture and the mapping of the dataflow are fixed. (``fixed_architecture``, ``fixed_spatial_mapping`` and ``fixed_temporal_mapping`` are all set to ``True``).

The architecture specs are defined in ``inputs/architecture.yaml`` while the mapping specs are define in ``inputs/mapping.yaml``. For more info on how these specifications are set, refer to [Input setting parameters](https://github.com/ZigZag-Project/zigzag/blob/master/inputs/README.md).

The cost estimation can be run with:
```python3 top_module.py --arch ./inputs/architecture.yaml --map ./inputs/mapping.yaml --set ./inputs/settings.yaml --mempool ./inputs/memory_pool.yaml```
### Temporal mapping exploration
A temporal mapping exploration can be carried out on the same architecture with the same workload by setting ``fixed_temporal_mapping`` to ``False`` in the settings file.

If the temporal mapping exploration is enabled, a search method must be specified. The search method (``exhaustive``, ``heuristic_v1``, ``heuristic_v2``, ``iterative``, or ``loma``) can be set in the settings file.

The temporal mapping exploration can be then run with:
```python3 top_module.py --arch ./inputs/architecture.yaml --map ./inputs/mapping.yaml --set ./inputs/settings.yaml --mempool ./inputs/memory_pool.yaml```
### Spatial unrolling exploration

Beside the temporal mapping exploration, spatial unrolling exploration can be carried out by setting the ``fixed_spatial_mapping`` to ``False`` as well.

If the spatial unrolling exploration is enabled, a search method must be
specified. The search method (``exhaustive``, ``heuristic_v1``,
``heuristic_v2``, or ``hint_driven``) can be set in the settings file.

A MAC array utilization threshold of >0.75 is suggested for reducing the exploration space. It can be specified in the settings file (``spatial_utilization_threshold``)

The spatial unrolling exploration can be then run with:
```python3 top_module.py --arch ./inputs/architecture.yaml --map ./inputs/mapping.yaml --set ./inputs/settings.yaml --mempool ./inputs/memory_pool.yaml```
### Architecture exploration

A basic architecture exploration run can be started by setting the ``fixed_architecture`` parameter to ``False`` in the settings file.

Ther architecture exploration can be then run with:
```python3 top_module.py --arch ./inputs/architecture.yaml --map ./inputs/mapping.yaml --set ./inputs/settings.yaml --mempool ./inputs/memory_pool_exploration.yaml```

### Input settings parameters

Please refer to [Input setting parameters](https://github.com/ZigZag-Project/zigzag/blob/master/inputs/README.md)

### Output file data format
Please refer to [Example result files](https://github.com/ZigZag-Project/zigzag/blob/master/example_result_file)

### Console information
While the tool is running, it prints some useful information on the console. Understand this information helps user to to better understand and control the DSE flow.

Please refer to [Console information](https://github.com/ZigZag-Project/zigzag/blob/master/example_result_file/console_info/README.md)


***
The research team is continuing building and polishing ZigZag. We welcome any comment, discussion, and contribution from the community.
