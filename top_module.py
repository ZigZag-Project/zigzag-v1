import output_funcs as of
import classes as cls

import sys
import msg
from msg import mem_scheme_fit_check
import argparse
import input_funcs
import time
from multiprocessing import Process, Value, Manager
from datetime import datetime
import evaluate
from classes.multi_manager import MultiManager
from im2col_funcs import im2col_layer_transform

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", help="Path to the mapping setting file")
    parser.add_argument("--mempool", help="Path to the memory pool file")
    parser.add_argument("--arch", help="Path to the architecture setting file")
    parser.add_argument("--set", help="Path to the main setting file")
    args = parser.parse_args()

    input_settings = input_funcs.get_input_settings(args.set, args.map, args.mempool, args.arch)
    layer_spec = importlib.machinery.SourceFileLoader('%s' % (input_settings.layer_filename),
                                                      '%s.py' % (input_settings.layer_filename)).load_module()

    if input_settings.im2col_enable:
        layer_spec_origin = importlib.machinery.SourceFileLoader('%s' % (input_settings.layer_filename),
                                                          '%s.py' % (input_settings.layer_filename)).load_module()
        layer_spec.layer_info = im2col_layer_transform(layer_spec.layer_info)

    # Extract the layer information from the layer_spec
    layers = [cls.Layer.extract_layer_info(layer_spec.layer_info[layer_number])
              for layer_number in input_settings.layer_number]

    # If there are duplicate layers, set flag for the latter ones.
    # This flag will prevent the layer from being evaluated later on to speed up run.
    for idx, layer in enumerate(layers):
        layers_seen = layers[:idx]
        for idx_other, other in enumerate(layers_seen):
            if layer == other:
                layer.set_duplicate(input_settings.layer_number[idx_other])

    # Setup a layer dictionary of following format for easy access
    # key: layer number
    # value: Layer class

    # layers_dict = {input_settings.layer_number[i]: layers[i]
    #             for i in range(len(layers))}

    results_path = input_settings.results_path

    if input_settings.mem_hierarchy_single_simulation:
        ma = [1]
    else:
        ma = input_settings.max_area
    tmp_mem_node_list = []

    print('ZigZag started running.')
    t1 = time.time()
    if not input_settings.mem_hierarchy_single_simulation:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, ' MSG started')
        mem_scheme_sim, mem_scheme_node_sim = msg.msg(input_settings.mem_pool,
                                                      input_settings.mac_array_info['array_size'],
                                                      ma, input_settings.utilization_rate_area,
                                                      input_settings.memory_hierarchy_ratio,
                                                      input_settings.prune_PE_RF,
                                                      input_settings.PE_RF_size_threshold,
                                                      input_settings.PE_RF_depth, input_settings.CHIP_depth,
                                                      input_settings.memory_scheme_hint, tmp_mem_node_list,
                                                      input_settings.mem_hierarchy_single_simulation,
                                                      input_settings.banking, input_settings.L1_size,
                                                      input_settings.L2_size)
        current_time = now.strftime("%H:%M:%S")
        print(current_time, ' MSG finished')
    if input_settings.mem_hierarchy_single_simulation:
        mem_scheme_sim, mem_scheme_node_sim = msg.msg(input_settings.mem_pool,
                                                      input_settings.mac_array_info['array_size'],
                                                      ma, input_settings.utilization_rate_area,
                                                      input_settings.memory_hierarchy_ratio,
                                                      input_settings.prune_PE_RF,
                                                      input_settings.PE_RF_size_threshold,
                                                      input_settings.PE_RF_depth, input_settings.CHIP_depth,
                                                      input_settings.memory_scheme_hint, tmp_mem_node_list,
                                                      input_settings.mem_hierarchy_single_simulation,
                                                      input_settings.banking, input_settings.L1_size,
                                                      input_settings.L2_size)

    # CHECKS IF THE LAST LEVEL IN MEMORY HIERARCHY MEETS STORAGE REQUIREMENTS FOR OPERANDS
    if not mem_scheme_sim:
        raise ValueError('No memory hierarchy found. Consider changing constraints in the architecture file.')
    for idx, each_mem_scheme in enumerate(mem_scheme_sim):
        mem_scheme_fit = mem_scheme_fit_check(each_mem_scheme, input_settings.precision, layer_spec.layer_info,
                                              input_settings.layer_number)
        if not mem_scheme_fit:
            del mem_scheme_sim[idx]
    if not mem_scheme_sim:
        raise ValueError('The largest memory in the hierarchy is still too small for holding the required workload.')

    # Manages the variables passed to the multiple parallel processes
    multi_manager = MultiManager(input_settings, mem_scheme_sim, layer_spec, layers)

    # A list containing the chunks that will be processed sequentially
    # Each element within a chunk will be processed in parallel
    # inter-chunk = serial
    # intra-chunk = parallel
    mem_scheme_sim_chunk_list = [mem_scheme_sim[i:i + input_settings.mem_scheme_parallel_processing] for i in
                                 range(0, len(mem_scheme_sim), input_settings.mem_scheme_parallel_processing)]

    for ii_mem_scheme_chunk, mem_scheme_sim_chunk in enumerate(mem_scheme_sim_chunk_list): # serial processing of chunks

        procs = []
        for mem_scheme_index, mem_scheme in enumerate(mem_scheme_sim_chunk):  # parallel processing of one chunk
            current_mem_scheme_index = mem_scheme_index + input_settings.mem_scheme_parallel_processing * ii_mem_scheme_chunk
            procs.append(Process(target=evaluate.mem_scheme_list_evaluate,
                                args=(input_settings, mem_scheme, current_mem_scheme_index, layers, multi_manager)))
        for p in procs: p.start()
        for p in procs: p.join()

    ''' Collect the optimum spatial unrolling results for all memory schemes if doing architecture exploration'''
    if not input_settings.mem_hierarchy_single_simulation:
        evaluate.optimal_su_evaluate(input_settings, layers, multi_manager)

    of.print_helper(input_settings, layers, multi_manager)

    total_time = int(time.time() - t1)
    print('ZigZag finished running. Total elapsed time: %d seconds.' % total_time)
    print('Results are saved to %s.' % results_path)
    print()
