import output_funcs as of
import cost_model_funcs as cmf
import bsg_ite
import classes as cls
import importlib.machinery
import sys
import numpy as np
from copy import deepcopy
from bsgutils import utilization_rate_optimizer
import msg
from msg import mem_scheme_fit_check
import pickle
import argparse
import copy
import input_funcs
import bsg_exh
import time
from multiprocessing import Process, Value, Manager
from datetime import datetime
from pathlib import Path

def mem_scheme_su_evaluate(input_settings, layer, layer_index, ii_layer_index, mem_scheme, mem_scheme_index, 
                            ii_su, spatial_unrolling, spatial_unrolling_count, multi_manager):

    mem_scheme_count = multi_manager.mem_scheme_count
    list_min_energy = multi_manager.list_min_energy
    list_min_en_output = multi_manager.list_min_en_output
    list_max_utilization = multi_manager.list_max_utilization
    list_max_ut_output = multi_manager.list_max_ut_output
    list_tm_count_en = multi_manager.list_tm_count_en
    list_tm_count_ut = multi_manager.list_tm_count_ut
    layer_spec = multi_manager.layer_spec
    list_sim_time = multi_manager.list_sim_time
    list_su_count = multi_manager.list_su_count

    tl_list = []
    t1 = time.time()
    ''' for pickle file'''
    pickle_enable = input_settings.tm_search_result_saving
    if pickle_enable:
        energy_collect = []
        utilization_collect = []
    min_energy = float('inf')
    max_utilization = 0
    max_utilization_energy = float('inf')
    best_output_energy = None
    best_output_utilization = None
    if input_settings.fixed_spatial_unrolling is True and input_settings.mem_hierarchy_single_simulation is False:
        mem_scheme = cmf.su_correction(mem_scheme)

    if input_settings.spatial_unrolling_mode != 4:
        layer_post = layer_spec.layer_info[layer_index]
        spatial_loop = cls.SpatialLoop.extract_loop_info(mem_scheme.spatial_unrolling[ii_su], layer_post)
        spatial_loop_comb = [spatial_loop, spatial_loop]
    else:
        layer_post = layer_spec.layer_info[layer_index][ii_su]
        spatial_loop = cls.SpatialLoop.extract_loop_info(mem_scheme.spatial_unrolling[ii_su], layer_post)
        spatial_loop_fractional = cls.SpatialLoop.extract_loop_info(mem_scheme.fraction_spatial_unrolling[ii_su],layer_post)
        spatial_loop_comb = [spatial_loop, spatial_loop_fractional]

    active_mac_cost = cmf.get_active_mac_cost(layer, input_settings.mac_array_info['single_mac_energy'])
    layer_rounded = cls.Layer.extract_layer_info(layer_post)
    idle_mac_cost = cmf.get_idle_mac_cost(layer, layer_rounded, input_settings.mac_array_info['array_size'],
                                          input_settings.mac_array_info['idle_mac_energy'],
                                          mem_scheme.spatial_unrolling)

    mem_scheme.mem_utilization_rate, good_scheme = utilization_rate_optimizer(mem_scheme.mem_size,
                                                                              mem_scheme.spatial_unrolling[ii_su],
                                                                              layer_post,
                                                                              input_settings.precision,
                                                                              mem_scheme.mem_utilization_rate)
    if not input_settings.utilization_optimizer_pruning:
        good_scheme = True
    tl_list = []
    if not good_scheme:
        print('Utilization pruning active. Mem scheme sub-optimal')
        discard_mem_scheme = True
    if good_scheme:
        print('SU', ii_su + 1, '/', len(mem_scheme.spatial_unrolling), mem_scheme.spatial_unrolling[ii_su])
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
              mem_scheme_index + 1, '/', mem_scheme_count, ', SU', ii_su + 1,
              '/', len(mem_scheme.spatial_unrolling), ' TMG started')

        if not input_settings.fixed_temporal_mapping:
            if input_settings.tmg_search_method == 0:
                tl_list, tl_combinations = bsg_ite.bsg(mem_scheme.mem_size, mem_scheme.mem_share,
                                                       input_settings.precision,
                                                       mem_scheme.mem_utilization_rate,
                                                       layer_post,
                                                       mem_scheme.spatial_unrolling[ii_su], layer, mem_scheme,
                                                       input_settings)
            if input_settings.tmg_search_method == 1:
                tl_list = bsg_exh.bsg(mem_scheme.mem_size, mem_scheme.mem_share, input_settings.precision,
                                      mem_scheme.mem_utilization_rate, layer_post,
                                      layer_index,
                                      mem_scheme.spatial_unrolling[ii_su], input_settings.drc_enabled,
                                      input_settings.stationary_optimization_enable)
                tl_combinations = len(tl_list)
        if input_settings.fixed_temporal_mapping:
            tl_list.append(input_settings.temporal_mapping_single)
            tl_combinations = 1
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if tl_list:
            if (not input_settings.fixed_temporal_mapping) and (input_settings.tmg_search_method == 0):
                print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
                      mem_scheme_index + 1, '/', mem_scheme_count, ', SU',
                      ii_su + 1, '/', len(mem_scheme.spatial_unrolling), ' TMG finished', '| Valid TM found ( partial:',
                      tl_combinations, ', final:', len(tl_list), ')')
            else:
                print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
                      mem_scheme_index + 1, '/', mem_scheme_count, ', SU',
                      ii_su + 1, '/', len(mem_scheme.spatial_unrolling), ' TMG Finished', '| Valid TM found',
                      len(tl_list))
        else:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, ' M', mem_scheme_index + 1, '/', mem_scheme_count, ', L', layer_index, ', SU',
                  ii_su + 1, '/',
                  len(mem_scheme.spatial_unrolling), ' No TM found')
            return

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
              mem_scheme_index + 1, '/', mem_scheme_count, ', SU', ii_su + 1,
              '/', len(mem_scheme.spatial_unrolling), ' CM  started')

        for tl_index, tl in enumerate(tl_list):
            temporal_loop = cls.TemporalLoop.extract_loop_info(layer, tl, spatial_loop)
            loop = cls.Loop.extract_loop_info(layer, temporal_loop, spatial_loop, input_settings.precision,
                                              input_settings.fixed_temporal_mapping)

            if input_settings.spatial_unrolling_mode == 4:
                temporal_loop_fractional = cls.TemporalLoop.extract_loop_info(layer, tl, spatial_loop_fractional)
                loop_fractional = cls.Loop.extract_loop_info(layer, temporal_loop_fractional, spatial_loop_fractional,
                                                             input_settings.precision, input_settings.fixed_temporal_mapping)
            else:
                loop_fractional = loop
            mem_scheme_cost = copy.deepcopy(mem_scheme)
            msc_list = [mem_scheme_cost]
            for ii, msc in enumerate(msc_list):
                utilization = cls.Utilization.get_utilization(layer, temporal_loop, spatial_loop_comb, loop,
                                                              input_settings.mac_array_info, msc.mem_size,
                                                              msc.mem_share, msc.mem_type,
                                                              input_settings.mac_array_stall,
                                                              input_settings.precision, msc.mem_bw)

                occupied_area = msg.get_mem_scheme_area(msc)

                total_cost_layer = 0
                # loop.array_wire_distance = {'W': [], 'I': [], 'O': []}
                operand_cost = {'W': [], 'I': [], 'O': []}
                schedule_info = {
                    'temporal': tl,
                    'spatial': mem_scheme.spatial_unrolling[ii_su],
                    'flooring': mem_scheme.flooring[ii_su]
                }

                for operand in ['W', 'I', 'O']:
                    for level in range(0, len(tl[operand])):
                        operand_cost[operand].append(
                            cmf.get_operand_level_energy_cost(operand, level, msc.mem_cost,
                                                              input_settings.mac_array_info,
                                                              schedule_info, loop_fractional,
                                                              msc.mem_fifo, msc, input_settings.precision,
                                                              utilization, ii))
                        # loop.array_wire_distance[operand].append(
                        #     cmf.get_operand_level_wire_distance(operand, level,
                        #                                         schedule_info,
                        #                                         input_settings.mac_array_info, loop,
                        #                                         msc.mem_fifo))
                    total_cost_layer += np.sum(operand_cost[operand])
                total_cost_layer += active_mac_cost + idle_mac_cost[ii_su]
                ''' for pickle file (collecting all temporal mappings' energy and array utilization)'''
                if pickle_enable:
                    energy_collect.append(int(total_cost_layer))
                    utilization_collect.append(utilization.mac_utilize_no_load)

                if (total_cost_layer < min_energy) or (
                        total_cost_layer == min_energy and utilization.mac_utilize_no_load > min_energy_utilization):
                    min_energy_utilization = utilization.mac_utilize_no_load
                    min_energy = total_cost_layer
                    output_result = of.CostModelOutput(total_cost_layer, deepcopy(operand_cost),
                                                       (active_mac_cost, idle_mac_cost[ii_su]),
                                                       deepcopy(temporal_loop.temporal_loop),
                                                       deepcopy(mem_scheme.spatial_unrolling[ii_su]),
                                                       deepcopy(mem_scheme.flooring[ii_su]),
                                                       deepcopy(loop_fractional), deepcopy(spatial_loop),
                                                       deepcopy(temporal_loop), occupied_area,
                                                       utilization, ii)
                    best_output_energy = output_result

                if (utilization.mac_utilize_no_load > max_utilization) or (
                        utilization.mac_utilize_no_load == max_utilization and total_cost_layer < max_utilization_energy):
                    max_utilization = utilization.mac_utilize_no_load
                    max_utilization_energy = total_cost_layer
                    output_result = of.CostModelOutput(total_cost_layer, deepcopy(operand_cost),
                                                       (active_mac_cost, idle_mac_cost[ii_su]),
                                                       deepcopy(temporal_loop.temporal_loop),
                                                       deepcopy(mem_scheme.spatial_unrolling[ii_su]),
                                                       deepcopy(mem_scheme.flooring[ii_su]),
                                                       deepcopy(loop_fractional), deepcopy(spatial_loop),
                                                       deepcopy(temporal_loop), occupied_area,
                                                       utilization, ii)
                    best_output_utilization = output_result

    if pickle_enable:
        rf = input_settings.results_path + '/all_tm_results/' + input_settings.results_filename + '_L_' + str(
            layer_index) + '_SU_' + str(ii_su + 1)
        ''' Create result folder if it does not exist. '''
        Path(input_settings.results_path + '/all_tm_results/').mkdir(parents=True, exist_ok=True)
        with open(rf + '_energy.pickle', 'wb') as f:
            pickle.dump(energy_collect, f)
            f.close()
        with open(rf + '_utilization.pickle', 'wb') as f:
            pickle.dump(utilization_collect, f)
            f.close()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
          mem_scheme_index + 1, '/', mem_scheme_count, ', SU', ii_su + 1, '/',
          spatial_unrolling_count, ' CM  finished', end='')
    discard_mem_scheme = False
    if tl_list and not discard_mem_scheme:
        t2 = time.time() - t1
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if input_settings.fixed_temporal_mapping:
            print(
                ' | Elapsed time: {0:d} sec | (energy, mac_utilization, area): ({1:d}, {2:.2f}, {3:d})'.format(
                    int(t2), int(round(min_energy)), min_energy_utilization, int(best_output_energy.area)))
        else:
            print(
                ' | Elapsed time: {0:d} sec | [min en: ({1:d}, {2:.2f}, {3:d}) max ut: ({4:d}, {5:.2f}, {6:d})] in all TMs'.format(
                    int(t2), int(round(min_energy)), min_energy_utilization, int(best_output_energy.area),
                    int(round(max_utilization_energy)), max_utilization, int(best_output_utilization.area)))

        '''
        Append common terms to the result list.
        '''
        spatial_unrolling_index = str(ii_su + 1) + '/' + str(spatial_unrolling_count)
        mem_scheme_count = str(mem_scheme_index + 1) + '/' + str(mem_scheme_count)
        common_settings = of.CommonSetting(input_settings,
                                           ii_layer_index,
                                           mem_scheme_count,
                                           spatial_unrolling_index,
                                           msc_list[0])

        if input_settings.fixed_temporal_mapping or input_settings.tmg_search_method != 0:
            tm_count = len(tl_list)
        else:
            tm_count = {'partial': tl_combinations, 'final': len(tl_list)}

        if input_settings.fixed_spatial_unrolling and input_settings.fixed_temporal_mapping:
            sub_path = '/fixed_tm_for_fixed_su/'
        elif input_settings.fixed_spatial_unrolling:
            sub_path = '/best_tm_for_fixed_su/'
        else:
            sub_path = '/best_tm_for_each_su/'

        if not (input_settings.fixed_spatial_unrolling is False
                and input_settings.su_search_result_saving is False):
            if not (input_settings.fixed_spatial_unrolling and input_settings.fixed_temporal_mapping):
                pass
                # rf = input_settings.results_path + sub_path + input_settings.results_filename + '_L' + str(
                #     layer_index) + '_M' + str(
                #     mem_scheme_index + 1) + '_SU' + str(
                #     ii_su + 1) + '_min_en'
                # of.print_xml(rf, layer, msc_list[0], best_output_energy, common_settings, tm_count, t2,
                #              input_settings.result_print_mode)

                # rf = input_settings.results_path + sub_path + input_settings.results_filename + '_L' + str(
                #     layer_index) + '_M' + str(
                #     mem_scheme_index + 1) + '_SU' + str(
                #     ii_su + 1) + '_max_ut'
                # of.print_xml(rf, layer, msc_list[0], best_output_utilization, common_settings, tm_count, t2,
                #              input_settings.result_print_mode)
            else:
                pass
                # rf = input_settings.results_path + sub_path + input_settings.results_filename + '_L' + str(
                #     layer_index) + '_M' + str(
                #     mem_scheme_index + 1) + '_SU' + str(
                #     ii_su + 1)
                # of.print_xml(rf, layer, msc_list[0], best_output_energy, common_settings, tm_count, t2,
                #              input_settings.result_print_mode)
    else:
        # total_cost_mem_scheme.value += float('inf')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print()
        print(current_time, ' L', layer_index, ', M ', mem_scheme_index + 1, ': no tl list for layer', layer_index)

    mem_scheme_str = 'M_%d' % (mem_scheme_index + 1)
    layer_str = 'L_%d' % (layer_index)
    mem_scheme_su_str = 'M_%d_SU_%d_%d' % (mem_scheme_index + 1, spatial_unrolling_count, ii_su + 1)

    list_min_energy[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: (min_energy, min_energy_utilization)})
    list_min_en_output[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: best_output_energy})
    list_max_utilization[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: (max_utilization_energy, max_utilization)})
    list_max_ut_output[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: best_output_utilization})
    list_tm_count_en[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: tm_count})
    list_tm_count_ut[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: tm_count})
    list_sim_time[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: t2})

    
def mem_scheme_evaluate(input_settings, layer_index, ii_layer_index, mem_scheme, mem_scheme_index, multi_manager):

    mem_scheme_count = multi_manager.mem_scheme_count
    layer_spec = multi_manager.layer_spec

    t1 = time.time()
    layer = cls.Layer.extract_layer_info(layer_spec.layer_info[layer_index])
    min_energy_utilization = 0
    discard_mem_scheme = False
    if discard_mem_scheme:
        return
    if input_settings.fixed_spatial_unrolling:
        mem_scheme.spatial_unrolling = [input_settings.spatial_unrolling_single]
        mem_scheme.flooring = [input_settings.flooring_single]
        spatial_unrolling = [input_settings.spatial_unrolling_single]
    if not input_settings.fixed_spatial_unrolling:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
              mem_scheme_index + 1, '/', mem_scheme_count, ' SUG started')

        # hint_driven_with_greedy_mapping
        if input_settings.spatial_unrolling_mode == 4:
            layer_rounded = cls.layer_rounding.LayerRound(layer_spec.layer_info[layer_index],
                                                          input_settings.mac_array_info['array_size'],
                                                          input_settings.unrolling_scheme_list)
            layer_spec.layer_info[layer_index] = layer_rounded.round_layer_info
            aux_layer_to_su_hint_table = layer_rounded.aux_layer_to_su_hint_table
            fraction_su = layer_rounded.fraction_su
            ideal_su = layer_rounded.ideal_su
            spatial_unrolling = []
            flooring = []
            fraction_spatial_unrolling = []
            for aux_layer_idx in range(len(layer_spec.layer_info[layer_index])):
                su_hint_idx = aux_layer_to_su_hint_table[aux_layer_idx]
                spatial_unrolling_, flooring_, mem_scheme, not_good = msg.spatial_unrolling_generator_with_hint(
                    mem_scheme, input_settings.mac_array_info['array_size'], layer_spec.layer_info[layer_index][su_hint_idx],
                    [input_settings.unrolling_scheme_list[su_hint_idx]])
                spatial_unrolling_, fraction_spatial_unrolling_ = \
                    msg.su_reformat(spatial_unrolling_, ideal_su[aux_layer_idx], fraction_su[aux_layer_idx])
                spatial_unrolling += spatial_unrolling_
                flooring += flooring_
                fraction_spatial_unrolling += fraction_spatial_unrolling_
            mem_scheme.fraction_spatial_unrolling = fraction_spatial_unrolling

        # hint_driven (prime factor factorization based)
        elif input_settings.spatial_unrolling_mode == 3:
            spatial_unrolling, flooring, mem_scheme, not_good = msg.spatial_unrolling_generator_with_hint(
                mem_scheme, input_settings.mac_array_info['array_size'], layer_spec.layer_info[layer_index],
                input_settings.unrolling_scheme_list)
            mem_scheme.fraction_spatial_unrolling = spatial_unrolling

        # spatial unrolling full search based on user-defined spatial_utilization_threshold
        else:
            spatial_unrolling, flooring, mem_scheme, not_good = msg.spatial_unrolling_generator(
                mem_scheme, input_settings.mac_array_info['array_size'], layer_spec.layer_info[layer_index],
                input_settings.precision, input_settings.spatial_utilization_threshold,
                input_settings.spatial_unrolling_mode)
            mem_scheme.fraction_spatial_unrolling = spatial_unrolling

        mem_scheme.spatial_unrolling = spatial_unrolling
        mem_scheme.flooring = flooring

        print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M', mem_scheme_index + 1, '/', mem_scheme_count, ' SUG finished',
              '| Valid SU found:', len(spatial_unrolling))
    
    if not mem_scheme.spatial_unrolling:
        discard_mem_scheme = True
        print('Layer', layer_index, ': no valid spatial unrolling found')
        return

    ''' input_settings.su_parallel_processing SU parallel '''
    TIMEOUT = 36000
    start = time.time()
    su_number = list(range(0, len(spatial_unrolling), 1))
    su_chunk_list = [su_number[i:i + input_settings.su_parallel_processing] for i in
                     range(0, len(spatial_unrolling), input_settings.su_parallel_processing)]
    for su_chunk in su_chunk_list:
        procs = [Process(target=mem_scheme_su_evaluate,
                         args=(input_settings, layer, layer_index, ii_layer_index, mem_scheme, mem_scheme_index, 
                         ii_su, spatial_unrolling[ii_su], len(spatial_unrolling), multi_manager))
                 for ii_su, su_x in zip(su_chunk, spatial_unrolling[su_chunk[0]:su_chunk[-1] + 1])]
        for p in procs: p.start()
        while time.time() - start <= TIMEOUT:  # and all([p.is_alive() for p in procs]):
            if any(p.is_alive() for p in procs):
                break
                time.sleep(1)
            else:
                print('MEM SCHEME EVALUATE : TIMED OUT - KILLING ALL PROCESSES')
                for p in procs:
                    p.terminate()
                    p.join()
        for p in procs: p.join()

    # Get the results from mem_scheme_su_evaluate
    list_min_energy = multi_manager.list_min_energy
    list_min_en_output = multi_manager.list_min_en_output
    list_max_utilization = multi_manager.list_max_utilization
    list_max_ut_output = multi_manager.list_max_ut_output
    list_tm_count_en = multi_manager.list_tm_count_en
    list_tm_count_ut = multi_manager.list_tm_count_ut
    list_sim_time = multi_manager.list_sim_time
    list_su_count = multi_manager.list_su_count

    mem_scheme_str = 'M_%d' % (mem_scheme_index + 1)
    layer_str = 'L_%d' % (layer_index)
    if len(list_min_energy[mem_scheme_str][layer_str]['best_tm_each_su']) != 0:
        t2 = time.time() - t1

        best_en = sys.float_info.max
        best_en_ut = 0
        for mem_su_str, (en, ut) in list_min_energy[mem_scheme_str][layer_str]['best_tm_each_su'].items():
            if (en < best_en) or (en == best_en and ut > best_en_ut):
                best_en = en
                best_en_ut = ut
                best_en_mem_su_str = mem_su_str

        best_en_output = list_min_en_output[mem_scheme_str][layer_str]['best_tm_each_su'].get(best_en_mem_su_str)
        tm_count_en = list_tm_count_en[mem_scheme_str][layer_str]['best_tm_each_su'].get(best_en_mem_su_str)

        list_tm_count_en[mem_scheme_str][layer_str]['best_su_each_mem'].update({best_en_mem_su_str: tm_count_en})
        list_min_energy[mem_scheme_str][layer_str]['best_su_each_mem'].update({best_en_mem_su_str: (best_en, best_en_ut)})
        list_min_en_output[mem_scheme_str][layer_str]['best_su_each_mem'].update({best_en_mem_su_str: best_en_output})
        list_min_en_output[mem_scheme_str][layer_str]['su_count'] = len(spatial_unrolling)
        list_sim_time[mem_scheme_str][layer_str]['best_su_each_mem'].update({best_en_mem_su_str: t2})
        list_su_count[mem_scheme_str][layer_str] = len(spatial_unrolling)

        best_ut = 0
        best_ut_en = sys.float_info.max
        for mem_su_str, (en, ut) in list_max_utilization[mem_scheme_str][layer_str]['best_tm_each_su'].items():
            if ut > best_ut or (ut == best_ut and en < best_ut_en):
                best_ut = ut
                best_ut_en = en
                best_ut_mem_su_str = mem_su_str

        best_ut_output = list_max_ut_output[mem_scheme_str][layer_str]['best_tm_each_su'].get(best_ut_mem_su_str)
        tm_count_ut = list_tm_count_ut[mem_scheme_str][layer_str]['best_tm_each_su'].get(best_ut_mem_su_str)

        list_tm_count_ut[mem_scheme_str][layer_str]['best_su_each_mem'].update({best_ut_mem_su_str: tm_count_ut})

        list_max_utilization[mem_scheme_str][layer_str]['best_su_each_mem'].update({best_ut_mem_su_str: (best_ut_en, best_ut)})

        dd = list_max_ut_output[mem_scheme_str]
        list_max_ut_output[mem_scheme_str][layer_str]['best_su_each_mem'].update({best_ut_mem_su_str: best_ut_output})

        list_sim_time[mem_scheme_str][layer_str]['best_su_each_mem'].update({best_ut_mem_su_str: t2})

        if not input_settings.fixed_spatial_unrolling:

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            print(
                '{0:s} {1:s} L {2:d},  M {3:d},  SU {4:s}  Min En: ({5:d}, {6:.2f}, {7:d}) in all SUs and TMs'.format(
                    current_time, str(input_settings.layer_filename.split('/')[-1]), layer_index, mem_scheme_index + 1,
                    best_en_mem_su_str.split('_')[-1], int(best_en), best_en_ut, int(best_en_output.area)))
            print(
                '{0:s} {1:s} L {2:d},  M {3:d},  SU {4:s}  Max Ut: ({5:d}, {6:.2f}, {7:d}) in all SUs and TMs'.format(
                    current_time, str(input_settings.layer_filename.split('/')[-1]), layer_index, mem_scheme_index + 1,
                    best_ut_mem_su_str.split('_')[-1], int(best_ut_en), best_ut, int(best_ut_output.area)))


def mem_scheme_list_evaluate(mem_scheme, input_settings, mem_scheme_index, multi_manager):
  
    mem_scheme_count = multi_manager.mem_scheme_count

    print('MEM HIERARCHY ', mem_scheme_index + 1, '/', mem_scheme_count)
    print(mem_scheme.mem_size)
    print(mem_scheme.mem_unroll)

    # total_cost_mem_scheme = Value('d', 0)
    discard_mem_scheme = False

    layer_chunk_list = [input_settings.layer_number[i:i + input_settings.layer_parallel_processing] for i in
                        range(0, len(input_settings.layer_number), input_settings.layer_parallel_processing)]

    for ii_layer_chunk, layer_chunk in enumerate(layer_chunk_list):
        procs = []
        for ii_layer_index, layer_index in enumerate(layer_chunk):
            procs.append(Process(target=mem_scheme_evaluate,
                            args=(input_settings, layer_index, ii_layer_index, mem_scheme, mem_scheme_index, multi_manager)))
        for p in procs: p.start()
        for p in procs: p.join()


def optimal_su_evaluate(input_settings, multi_manager):
    ''' Collect the optimum spatial unrolling results for all memory schemes '''

    # Get the results from multi_manager
    mem_scheme_sim = multi_manager.mem_scheme_sim
    list_tm_count_en = multi_manager.list_tm_count_en
    list_min_energy = multi_manager.list_min_energy
    list_min_en_output = multi_manager.list_min_en_output
    list_tm_count_ut = multi_manager.list_tm_count_ut
    list_max_utilization = multi_manager.list_max_utilization
    list_max_ut_output = multi_manager.list_max_ut_output
    list_sim_time = multi_manager.list_sim_time
    list_sim_time_en = multi_manager.list_sim_time_en
    list_sim_time_ut = multi_manager.list_sim_time_ut

    # Iterate through the processed layers
    # Saving for each layer the best mem + su + tm combination
    for layer_index in input_settings.layer_number:

        layer_str = 'L_%d' % layer_index

        en_best_su_each_mem = []
        en_output_best_su_each_mem = []
        ut_best_su_each_mem = []
        ut_output_best_su_each_mem = []

        # Aggregate results for every memory scheme to a list
        for mem_str in ['M_%d' % (i + 1) for i in range(len(mem_scheme_sim))]:
            en_best_su_each_mem.append(list_min_energy[mem_str][layer_str]['best_su_each_mem'])
            en_output_best_su_each_mem.append(list_min_en_output[mem_str][layer_str]['best_su_each_mem'])
            ut_best_su_each_mem.append(list_max_utilization[mem_str][layer_str]['best_su_each_mem'])
            ut_output_best_su_each_mem.append(list_max_ut_output[mem_str][layer_str]['best_su_each_mem'])

        if not en_best_su_each_mem:
            raise ValueError('No valid design point found. Please consider changing search method in the setting file.')

        best_en = sys.float_info.max
        best_en_ut = 0
        for idd, su_dict in enumerate(en_best_su_each_mem):
            m_su_idx, (en, ut) = list(su_dict.items())[0]
            if (en < best_en) or (en == best_en and ut > best_en_ut):
                best_en = en
                best_en_ut = ut
                best_en_mem_su_str = m_su_idx
                best_ie = idd

        best_en_output_dict = en_output_best_su_each_mem[best_ie]
        mem_scheme_str_en = 'M_%s' % (best_en_mem_su_str.split('_')[1])
        tm_count_en = list_tm_count_en[mem_scheme_str_en][layer_str]['best_su_each_mem'][best_en_mem_su_str]

        list_min_energy['best_mem_each_layer'][layer_str] = {best_en_mem_su_str: (best_en, best_en_ut)}
        list_min_en_output['best_mem_each_layer'][layer_str] = best_en_output_dict
        list_tm_count_en['best_mem_each_layer'][layer_str] = {best_en_mem_su_str: tm_count_en}
        list_sim_time_en['best_mem_each_layer'][layer_str] = list_sim_time[mem_scheme_str_en][layer_str]['best_su_each_mem'][best_en_mem_su_str]

        best_ut = 0
        best_ut_en = sys.float_info.max
        for idd, su_dict in enumerate(ut_best_su_each_mem):
            m_su_idx, (en, ut) = list(su_dict.items())[0]
            if ut > best_ut or (ut == best_ut and en < best_ut_en):
                best_ut = ut
                best_ut_en = en
                best_ut_mem_su_str = m_su_idx
                best_iu = idd

        best_ut_output_dict = ut_output_best_su_each_mem[best_iu]
        mem_scheme_str_ut = 'M_%s' % (best_ut_mem_su_str.split('_')[1])
        tm_count_ut = list_tm_count_ut[mem_scheme_str_ut][layer_str]['best_su_each_mem'][best_ut_mem_su_str]

        list_max_utilization['best_mem_each_layer'][layer_str] = {best_ut_mem_su_str: (best_ut_en, best_ut)}
        list_max_ut_output['best_mem_each_layer'][layer_str] = best_ut_output_dict
        list_tm_count_ut['best_mem_each_layer'][layer_str] = {best_ut_mem_su_str: tm_count_ut}
        list_sim_time_ut['best_mem_each_layer'][layer_str] = list_sim_time[mem_scheme_str_ut][layer_str]['best_su_each_mem'][best_ut_mem_su_str]

    # If multiple layers, iterate through all the memory hierarchies
    # to find the best hierarchy for the network of layers
    if len(input_settings.layer_number) > 1:

        best_network_energy = sys.float_info.max # energy of minimal energy network
        best_network_energy_latency = sys.float_info.max # latency of minimal energy network
        best_network_latency_energy = sys.float_info.max # energy of minimal latency network
        best_network_latency = sys.float_info.max # latency of minimal latency network
        best_mem_scheme_index_en = None
        best_mem_scheme_index_ut = None

        for mem_scheme_index in range(len(mem_scheme_sim)):

            mem_scheme_str = 'M_%d' % (mem_scheme_index + 1)
          
            tot_min_en_energy = 0
            tot_min_en_latency = 0

            tot_max_ut_energy = 0
            tot_max_ut_latency = 0

            for layer_index in input_settings.layer_number:
                layer_str = 'L_%d' % layer_index

                # Energy part
                su_dict_en = list_min_energy[mem_scheme_str][layer_str]['best_su_each_mem']
                mem_scheme_su_str_en = list(su_dict_en.keys())[0] # casts dict_keys type to list
                (min_en_en, min_en_ut) = list(su_dict_en.values())[0] # casts dict_values type to list
                min_en_output = list_min_en_output[mem_scheme_str][layer_str]['best_su_each_mem'][mem_scheme_su_str_en]
                min_en_latency = min_en_output.utilization.latency_tot

                tot_min_en_energy += min_en_en
                tot_min_en_latency += min_en_latency

                # Utilization (latency) part
                su_dict_ut = list_max_utilization[mem_scheme_str][layer_str]['best_su_each_mem']
                mem_scheme_su_str_ut = list(su_dict_ut.keys())[0] # casts dict_keys type to list
                (max_ut_en, max_ut_ut) = list(su_dict_ut.values())[0] # cast dict_values type to list
                max_ut_output = list_max_ut_output[mem_scheme_str][layer_str]['best_su_each_mem'][mem_scheme_su_str_ut]
                max_ut_latency = max_ut_output.utilization.latency_tot

                tot_max_ut_energy += max_ut_en
                tot_max_ut_latency += max_ut_latency

            # Check if total energy for this memory scheme is best so far
            if ((tot_min_en_energy < best_network_energy) or 
            (tot_min_en_energy == best_network_energy and tot_min_en_latency < best_network_energy_latency)):
                best_network_energy = tot_min_en_energy
                best_network_energy_latency = tot_min_en_latency
                best_mem_scheme_index_en = mem_scheme_index

            # Check if total latency for this memory scheme is best so far
            if ((tot_max_ut_latency < best_network_latency) or 
            (tot_max_ut_latency == best_network_latency and tot_max_ut_energy < best_network_latency_energy)):
                best_network_latency = tot_max_ut_latency
                best_network_latency_energy = tot_max_ut_energy
                best_mem_scheme_index_ut = mem_scheme_index

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            network_name = str(input_settings.layer_filename.split('/')[-1])

            print('{0:s} {1:s} M {2:d}: Minimal energy for all layers:      (energy, latency) = ({3:d}, {4:d})'.format(
                current_time, network_name, mem_scheme_index + 1, int(tot_min_en_energy), int(tot_min_en_latency)))
            print('{0:s} {1:s} M {2:d}: Maximal utilization for all layers: (energy, latency) = ({3:d}, {4:d})'.format(
                current_time, network_name, mem_scheme_index + 1, int(tot_max_ut_energy), int(tot_max_ut_latency)))                

        # Set the multi_manager's parameter with the correct mem_scheme_index
        multi_manager.best_mem_scheme_index_en = best_mem_scheme_index_en
        multi_manager.best_mem_scheme_index_ut = best_mem_scheme_index_ut
