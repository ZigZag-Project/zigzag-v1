import output_funcs as of
import cost_model_funcs as cmf
import bsg_ite
import classes as cls
import sys
import numpy as np
from copy import deepcopy
from bsgutils import utilization_rate_optimizer
import msg
import pickle
import copy
import bsg_exh
import time
from multiprocessing import Pool, Process, cpu_count
from datetime import datetime
from pathlib import Path
from itertools import repeat
from classes.layer_rounding import mem_access_count_correct


def tl_worker(tl_list, input_settings, mem_scheme, layer, spatial_loop, spatial_loop_fractional, spatial_loop_comb,
              ii_su, active_mac_cost, idle_mac_cost, im2col_need_correct, pool):

    [layer_origin, layer_rounded] = layer
    pickle_enable = input_settings.tm_search_result_saving
    energy_collect = None
    utilization_collect = None
    latency_collect = None
    if pickle_enable:
        group_count = layer_origin.G
        energy_collect = []
        utilization_collect = []
        latency_collect = []
    
    min_energy = float('inf')
    min_energy_utilization = 0
    max_utilization = 0
    max_utilization_energy = float('inf')

    try:
        greedy_mapping_flag = mem_scheme.greedy_mapping_flag[ii_su]
        footer_info = mem_scheme.footer_info[ii_su]
    except:
        greedy_mapping_flag = False
        footer_info = None

    for idx, tl in enumerate(tl_list):
        temporal_loop = cls.TemporalLoop.extract_loop_info(layer_rounded, tl, spatial_loop)
        loop = cls.Loop.extract_loop_info(layer_rounded, temporal_loop, spatial_loop, input_settings.precision,
                                          input_settings.fixed_temporal_mapping, pool)
        
        # TODO
        # if im2col_need_correct is True:
        #     tl_col2im = tl_col2im_transfer(tl, layer_origin_7D)

        if input_settings.spatial_unrolling_mode in [4, 5]:
            ############# Advanced User Configuration #############
            mem_energy_saving_when_BW_under_utilized = False
            #######################################################
            temporal_loop_fractional = cls.TemporalLoop.extract_loop_info(layer_origin, tl, spatial_loop_fractional)
            loop_fractional = cls.Loop.extract_loop_info(layer_origin, temporal_loop_fractional, spatial_loop_fractional,
                                                         input_settings.precision,
                                                         input_settings.fixed_temporal_mapping, pool)
            if mem_energy_saving_when_BW_under_utilized is False:
                loop_fractional = mem_access_count_correct(loop_fractional, loop)

        else:
            loop_fractional = loop

        msc = copy.deepcopy(mem_scheme)
        ii = 0
        utilization = cls.Utilization.get_utilization(layer_rounded, temporal_loop, spatial_loop_comb, loop,
                                                      input_settings.mac_array_info, msc.mem_size,
                                                      msc.mem_share, msc.mem_type,
                                                      input_settings.mac_array_stall,
                                                      input_settings.precision, msc.mem_bw)

        occupied_area = msg.get_mem_scheme_area(msc, spatial_loop.unit_count)
        
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
                # TODO
                # loop.array_wire_distance[operand].append(
                #     cmf.get_operand_level_wire_distance(operand, level,
                #                                         schedule_info,
                #                                         input_settings.mac_array_info, loop,
                #                                         msc.mem_fifo))
            total_cost_layer += np.sum(operand_cost[operand])
        if input_settings.computing_core == 'digital':
            total_cost_layer += active_mac_cost + idle_mac_cost
        else:
            total_cost_layer += active_mac_cost[0] + idle_mac_cost
        ''' for pickle file (collecting all temporal mappings' energy and array utilization)'''
        if pickle_enable:
            print("Index: %d    Energy: %d      Utilization: %.3f" % (
                idx, int(group_count * total_cost_layer), utilization.mac_utilize_no_load))
            energy_collect.append(int(group_count * total_cost_layer))
            utilization_collect.append(utilization.mac_utilize_no_load)
            latency_collect.append(utilization.latency_no_load)

        if (total_cost_layer < min_energy) or (
                total_cost_layer == min_energy and utilization.mac_utilize_no_load > min_energy_utilization):
            min_energy_utilization = utilization.mac_utilize_no_load
            min_energy = total_cost_layer
            output_result = of.CostModelOutput(total_cost_layer, deepcopy(operand_cost),
                                               (active_mac_cost, idle_mac_cost),
                                               deepcopy(temporal_loop.temporal_loop),
                                               deepcopy(mem_scheme.spatial_unrolling[ii_su]),
                                               deepcopy(mem_scheme.flooring[ii_su]),
                                               deepcopy(loop_fractional), deepcopy(spatial_loop),
                                               greedy_mapping_flag, footer_info,
                                               deepcopy(temporal_loop), occupied_area,
                                               utilization, ii)

            best_output_energy = output_result

        if (utilization.mac_utilize_no_load > max_utilization) or (
                utilization.mac_utilize_no_load == max_utilization and total_cost_layer < max_utilization_energy):
            max_utilization = utilization.mac_utilize_no_load
            max_utilization_energy = total_cost_layer
            output_result = of.CostModelOutput(total_cost_layer, deepcopy(operand_cost),
                                               (active_mac_cost, idle_mac_cost),
                                               deepcopy(temporal_loop.temporal_loop),
                                               deepcopy(mem_scheme.spatial_unrolling[ii_su]),
                                               deepcopy(mem_scheme.flooring[ii_su]),
                                               deepcopy(loop_fractional), deepcopy(spatial_loop),
                                               greedy_mapping_flag, footer_info,
                                               deepcopy(temporal_loop), occupied_area,
                                               utilization, ii)
            best_output_utilization = output_result

    return (min_energy, min_energy_utilization, best_output_energy, 
        max_utilization_energy, max_utilization, best_output_utilization,
        energy_collect, utilization_collect, latency_collect)


def mem_scheme_su_evaluate(input_settings, layer, im2col_layer, layer_index, layer_info, mem_scheme, mem_scheme_index,
                           ii_su, spatial_unrolling, spatial_unrolling_count, im2col_need_correct, multi_manager):
    mem_scheme_count = multi_manager.mem_scheme_count
    list_min_energy = multi_manager.list_min_energy
    list_min_en_output = multi_manager.list_min_en_output
    list_max_utilization = multi_manager.list_max_utilization
    list_max_ut_output = multi_manager.list_max_ut_output
    list_tm_count_en = multi_manager.list_tm_count_en
    list_tm_count_ut = multi_manager.list_tm_count_ut
    list_sim_time = multi_manager.list_sim_time

    '''
    Distinguish different layer parameters
    layer_7D_origin -- multi_manager.layer_spec.layer_info[layer_index], layer
    layer_3D_origin -- multi_manager.layer_info_im2col[layer_index], im2col_layer
    layer_3D/7D_rounded -- layer_info[layer_index][ii_su] (w/wo im2col)
    '''
    layer_filename = 'VGG16'
    if layer_filename == 'VGG16':
        layers = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        pool_list = [[1,1],[2,2],[1,1],[2,2],[1,1],[1,1],[2,2],[1,1],[1,1],[2,2],[1,1],[1,1],[2,2],[1,1],[1,1],[1,1]]
    if layer_filename == 'AlexNet':
        layers = [1,2,3,4,5]
        pool_list = [[2,2],[2,2],[1,1],[1,1],[2,2]]
    if layer_filename == 'ResNet20':
        layers = list(range(1,30))
        pool_list = [[1,1]]*layers.__len__()

    pooling_layer = pool_list[layer_index - 1]

    t1 = time.time()

    # for pickle file
    pickle_enable = input_settings.tm_search_result_saving

    # make sure the # of level in spatial unrolling and in memory scheme match.
    if input_settings.fixed_spatial_unrolling is True and input_settings.mem_hierarchy_single_simulation is False:
        mem_scheme = cmf.su_correction(mem_scheme)

    if input_settings.spatial_unrolling_mode in [4, 5]:
        layer_post = layer_info[layer_index][ii_su]
        spatial_loop = cls.SpatialLoop.extract_loop_info(mem_scheme.spatial_unrolling[ii_su], layer_post)
        spatial_loop_fractional = cls.SpatialLoop.extract_loop_info(mem_scheme.fraction_spatial_unrolling[ii_su],
                                                                    layer_post)
        spatial_loop_comb = [spatial_loop, spatial_loop_fractional]
    else:
        layer_post = layer_info[layer_index]
        spatial_loop = cls.SpatialLoop.extract_loop_info(mem_scheme.spatial_unrolling[ii_su], layer_post)
        spatial_loop_fractional = None
        spatial_loop_comb = [spatial_loop, spatial_loop]

    if input_settings.computing_core == 'digital':
        active_mac_cost = cmf.get_active_mac_cost(layer, input_settings.mac_array_info['single_mac_energy'])
    elif input_settings.computing_core == 'analog':
        au = mem_scheme.spatial_unrolling[ii_su]['W'][0]
        imc_array_unroll = [[x for x in au if x[0] in [1,2,5]],[x for x in au if x[0] in [3,6]]]
        active_mac_cost = cmf.get_imc_cost(imc_array_unroll, input_settings.mac_array_info['array_size'], input_settings.imc_act_line_cap, input_settings.imc_sum_line_cap, input_settings.imc_precision, \
                                           input_settings.imc_vdd, input_settings.imc_act_line_v, input_settings.imc_sum_line_v, input_settings.imc_DAC_cost, input_settings.imc_ADC_cost, input_settings.imc_n_short_rows, input_settings.imc_n_act_serial, \
                                           input_settings.imc_write_cost_cell, layer.total_MAC_op, layer)
    layer_rounded = cls.Layer.extract_layer_info(layer_post)
    idle_mac_cost = cmf.get_idle_mac_cost(layer, layer_rounded, input_settings.mac_array_info['array_size'],
                                          input_settings.mac_array_info['idle_mac_energy'],
                                          mem_scheme.spatial_unrolling)
    # print('user-defined mem ut', mem_scheme.mem_utilization_rate)
    mem_scheme.mem_utilization_rate, good_scheme = utilization_rate_optimizer(mem_scheme.mem_size,
                                                                              mem_scheme.spatial_unrolling[ii_su],
                                                                              layer_post,
                                                                              input_settings.precision,
                                                                              mem_scheme.mem_utilization_rate,
                                                                              spatial_loop.unit_unique)
    # print('generated mem ut', mem_scheme.mem_utilization_rate)
    if not input_settings.utilization_optimizer_pruning:
        good_scheme = True
    tl_list = []
    if not good_scheme:
        print('Utilization pruning active. Mem scheme sub-optimal')
        discard_mem_scheme = True
    if good_scheme:
        # print('SU', ii_su + 1, '/', len(mem_scheme.spatial_unrolling), mem_scheme.spatial_unrolling[ii_su],
        #       mem_scheme.su_utilization[ii_su])
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
                # tl_list = bsg_exh.bsg(mem_scheme.mem_size, mem_scheme.mem_share, input_settings.precision,
                #                       mem_scheme.mem_utilization_rate, layer_post,
                #                       layer_index,
                #                       mem_scheme.spatial_unrolling[ii_su], input_settings.drc_enabled,
                #                       input_settings.stationary_optimization_enable)
                # tl_combinations = len(tl_list)

                ####################### Advanced User Configuration #######################
                fixed_order = [1, 2, 3, 4, 5, 6]
                tl_list = bsg_exh.bsg_fixed_order(fixed_order,
                                                  mem_scheme.mem_size,
                                                  mem_scheme.mem_share,
                                                  input_settings.precision,
                                                  mem_scheme.mem_utilization_rate,
                                                  layer_post, layer_index,
                                                  mem_scheme.spatial_unrolling[ii_su])
                tl_combinations = len(tl_list)
                ###########################################################################

        if input_settings.fixed_temporal_mapping:
            tl_list.append(input_settings.temporal_mapping_single)
            tl_combinations = 1

        t2 = time.time()
        t_tmg = int(t2 - t1)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if tl_list:
            if (not input_settings.fixed_temporal_mapping) and (input_settings.tmg_search_method == 0):
                print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
                      mem_scheme_index + 1, '/', mem_scheme_count, ', SU', ii_su + 1,
                      '/', len(mem_scheme.spatial_unrolling), ' TMG finished', '| Elapsed time:', t_tmg, 'sec',
                      '| Valid TMs found: ( partial:', tl_combinations, ', final:', len(tl_list), ')')
            else:
                print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
                      mem_scheme_index + 1, '/', mem_scheme_count, ', SU', ii_su + 1,
                      '/', len(mem_scheme.spatial_unrolling), ' TMG Finished', '| Elapsed time:', t_tmg, 'sec',
                      '| Valid TMs found:', len(tl_list))

        else:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
                  mem_scheme_index + 1, '/', mem_scheme_count, ', SU', ii_su + 1,
                  '/', len(mem_scheme.spatial_unrolling), ' No TM found')
            return

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
              mem_scheme_index + 1, '/', mem_scheme_count, ', SU', ii_su + 1,
              '/', len(mem_scheme.spatial_unrolling), ' CM  started')

        # Convert tl_list to chunked list to pass to parallel cores
        n_processes = min(cpu_count(), input_settings.temporal_mapping_multiprocessing)
        chunk_size = int(tl_combinations / n_processes) + (tl_combinations % n_processes > 0)  # avoids import math
        tl_count = len(tl_list)
        tl_list = [tl_list[i:i + chunk_size] for i in range(0, tl_combinations, chunk_size)]

        # 'layer' is the original 7D layer
        # 'im2col_layer' is the original 3D/7D layer, depending on im2col_enable
        # 'layer_rounded' is the rounded 3D/7D layer, depending on im2col_enable
        if input_settings.im2col_enable:
            layer = [im2col_layer, layer_rounded]
        else:
            layer = [im2col_layer, layer_rounded]

        # Create list of repeated arguments passed to parallel tl_worker functions
        fixed_args = [input_settings, mem_scheme, layer, spatial_loop, spatial_loop_fractional, spatial_loop_comb,
                      ii_su, active_mac_cost, idle_mac_cost[ii_su], im2col_need_correct, pooling_layer]

        # Call the worker function for each chunk
        pool = Pool(processes=n_processes)
        results = pool.starmap(tl_worker, [[tl_chunk] + fixed_args for tl_chunk in tl_list])

        best_output_energy = None
        best_output_utilization = None
        best_energy = float('inf')
        best_energy_utilization = 0
        best_utilization = 0
        best_utilization_energy = float('inf')

        # Create pickle file to append to if pickle_enable
        if pickle_enable:
            parent_folder = "%s/all_tm_results/" % (input_settings.results_path)
            rf = "%s/%s_L_%d_SU_%d" % (parent_folder, input_settings.results_filename, layer_index, ii_su + 1)
            rf_en = rf + '_energy.pickle'
            rf_ut = rf + '_utilization.pickle'
            rf_lat = rf + '_latency.pickle'
            rf_en_ut = rf + '_combined.pickle'
            # Create parent folder if it does not exist
            Path(parent_folder).mkdir(parents=True, exist_ok=True)

        # Loop through the best energy/ut found by the parallel processes to find the overall best one
        for (min_en, min_en_ut, min_en_output, max_ut_en, max_ut, max_ut_output, en_collect, ut_collect, lat_collect) in results:
            if (min_en < best_energy or (min_en == best_energy and min_en_ut > best_energy_utilization)):
                best_energy = min_en
                best_energy_utilization = min_en_ut
                best_output_energy = min_en_output
            if (max_ut > best_utilization or (max_ut == best_utilization and max_ut_en < best_utilization_energy)):
                best_utilization_energy = max_ut_en
                best_utilization = max_ut
                best_output_utilization = max_ut_output

            # Save the collected (energy,ut) from every temporal mapping if required
            if pickle_enable:
                # Save energy
                with open(rf_en, 'ab') as f:
                    pickle.dump(en_collect, f)
                    f.close()
                # Save utilization
                # with open(rf_ut, 'ab') as f:
                #     pickle.dump(ut_collect, f)
                #     f.close()
                # Save latency
                with open(rf_lat, 'ab') as f:
                    pickle.dump(lat_collect, f)
                    f.close()
                # Save combined (en,ut) tuples
                # combined = zip(en_collect, ut_collect)
                # with open(rf_en_ut, 'ab') as f:
                #     for elem in combined:
                #         pickle.dump(elem, f)
                #     f.close()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
          mem_scheme_index + 1, '/', mem_scheme_count, ', SU', ii_su + 1, '/',
          spatial_unrolling_count, ' CM  finished', end='')
    discard_mem_scheme = False
    if tl_list and not discard_mem_scheme:
        t3 = time.time()
        t_cm = int(t3 - t2)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        group_count = layer[0].G
        if input_settings.fixed_temporal_mapping:
            print(
                ' | Elapsed time: {0:d} sec | (energy, mac_utilization, area): ({1:.3E}, {2:.3f}, {3:.3E})'.format(
                    int(t_cm), int(round(group_count*best_energy)), best_energy_utilization, int(best_output_energy.area)))
        else:
            print(
                ' | Elapsed time: {0:d} sec | [min en: ({1:.3E}, {2:.3f}, {3:.3E}) max ut: ({4:.3E}, {5:.3f}, {6:.3E})] in all TMs'.format(
                    int(t_cm), int(round(group_count*best_energy)), best_energy_utilization, int(best_output_energy.area),
                    int(round(group_count*best_utilization_energy)), best_utilization, int(best_output_utilization.area)))

        if input_settings.fixed_temporal_mapping or input_settings.tmg_search_method != 0:
            tm_count = tl_count
        else:
            tm_count = {'partial': tl_combinations, 'final': tl_count}

    else:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print()
        print(current_time, ' L', layer_index, ', M ', mem_scheme_index + 1, ': no tl list for layer', layer_index)

    mem_scheme_str = 'M_%d' % (mem_scheme_index + 1)
    layer_str = 'L_%d' % (layer_index)
    mem_scheme_su_str = 'M_%d_SU_%d_%d' % (mem_scheme_index + 1, spatial_unrolling_count, ii_su + 1)

    list_min_energy[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: (best_energy, best_energy_utilization)})
    list_min_en_output[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: best_output_energy})
    list_max_utilization[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: (best_utilization_energy, best_utilization)})
    list_max_ut_output[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: best_output_utilization})
    list_tm_count_en[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: tm_count})
    list_tm_count_ut[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: tm_count})
    list_sim_time[mem_scheme_str][layer_str]['best_tm_each_su'].update({mem_scheme_su_str: (t_tmg + t_cm)})


def mem_scheme_evaluate(input_settings, layer_index, layer, im2col_layer, mem_scheme, mem_scheme_index, multi_manager):
    mem_scheme_count = multi_manager.mem_scheme_count

    # Check if this is a duplicate layer
    if layer.is_duplicate:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index,
              'is a duplicate of L', layer.parent, '. Skipping exploration.')
        return

    if input_settings.im2col_enable:
        layer_info = deepcopy(multi_manager.layer_info_im2col)
        current_layer = multi_manager.layer_spec.layer_info[layer_index]
        if (current_layer['FX'] == 1 and current_layer['FY'] == 1) or \
                (current_layer['OX'] == 1 and current_layer['OY'] == 1) or \
                len(mem_scheme.mem_unroll['I']) <= input_settings.im2col_top_mem_level + 1:
            im2col_need_correct = False
        else:
            im2col_need_correct = True
    else:
        layer_info = deepcopy(multi_manager.layer_spec.layer_info)
        im2col_need_correct = False
    print('Layer', layer_index, layer_info[layer_index])

    t1 = time.time()
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

        # greedy mapping without hint
        if input_settings.spatial_unrolling_mode == 5:
            layer_rounded = cls.layer_rounding.LayerRound2(layer_info[layer_index],
                                                          input_settings.mac_array_info['array_size'],
                                                          input_settings.spatial_utilization_threshold)

            layer_info[layer_index] = layer_rounded.round_layer_info
            aux_layer_to_su_hint_table = layer_rounded.aux_layer_to_su_hint_table
            fraction_su = layer_rounded.fraction_su
            ideal_su = layer_rounded.ideal_su
            unrolling_scheme_list = layer_rounded.unrolling_scheme_list
            mem_scheme.greedy_mapping_flag = layer_rounded.greedy_mapping_flag
            mem_scheme.footer_info = layer_rounded.footer_info
            mem_scheme.su_utilization = layer_rounded.spatial_utilization_list
            spatial_unrolling = []
            flooring = []
            fraction_spatial_unrolling = []
            for idd, aux_layer_idx in enumerate(range(len(layer_info[layer_index]))):
                su_hint_idx = aux_layer_to_su_hint_table[aux_layer_idx]
                spatial_unrolling_, flooring_, mem_scheme, not_good = msg.spatial_unrolling_generator_with_hint(
                    mem_scheme, input_settings.mac_array_info['array_size'],
                    layer_info[layer_index][aux_layer_idx], [unrolling_scheme_list[su_hint_idx]])
                if not spatial_unrolling_:
                    continue
                if layer_rounded.greedy_mapping_flag[idd]:
                    spatial_unrolling_, fraction_spatial_unrolling_ = \
                        msg.su_reformat(spatial_unrolling_, ideal_su[aux_layer_idx], fraction_su[aux_layer_idx])
                else:
                    fraction_spatial_unrolling_ = spatial_unrolling_
                spatial_unrolling += spatial_unrolling_
                flooring += flooring_
                fraction_spatial_unrolling += fraction_spatial_unrolling_
            mem_scheme.fraction_spatial_unrolling = fraction_spatial_unrolling

        # greedy mapping with hint
        elif input_settings.spatial_unrolling_mode == 4:
            layer_rounded = cls.layer_rounding.LayerRound(layer_info[layer_index],
                                                          input_settings.mac_array_info['array_size'],
                                                          input_settings.unrolling_scheme_list,
                                                          input_settings.unrolling_size_list,
                                                          input_settings.spatial_utilization_threshold)

            layer_info[layer_index] = layer_rounded.round_layer_info
            aux_layer_to_su_hint_table = layer_rounded.aux_layer_to_su_hint_table
            fraction_su = layer_rounded.fraction_su
            ideal_su = layer_rounded.ideal_su
            mem_scheme.greedy_mapping_flag = layer_rounded.greedy_mapping_flag
            mem_scheme.footer_info = layer_rounded.footer_info
            mem_scheme.su_utilization = layer_rounded.spatial_utilization_list
            spatial_unrolling = []
            flooring = []
            fraction_spatial_unrolling = []
            for idd, aux_layer_idx in enumerate(range(len(layer_info[layer_index]))):
                su_hint_idx = aux_layer_to_su_hint_table[aux_layer_idx]
                spatial_unrolling_, flooring_, mem_scheme, not_good = msg.spatial_unrolling_generator_with_hint(
                    mem_scheme, input_settings.mac_array_info['array_size'],
                    layer_info[layer_index][aux_layer_idx],
                    [input_settings.unrolling_scheme_list[su_hint_idx]])
                if layer_rounded.greedy_mapping_flag[idd]:
                    spatial_unrolling_, fraction_spatial_unrolling_ = \
                        msg.su_reformat(spatial_unrolling_, ideal_su[aux_layer_idx], fraction_su[aux_layer_idx])
                else:
                    fraction_spatial_unrolling_ = spatial_unrolling_
                spatial_unrolling += spatial_unrolling_
                flooring += flooring_
                fraction_spatial_unrolling += fraction_spatial_unrolling_
            mem_scheme.fraction_spatial_unrolling = fraction_spatial_unrolling

        # hint_driven (prime factor factorization based)
        elif input_settings.spatial_unrolling_mode == 3:
            spatial_unrolling, flooring, mem_scheme, not_good = msg.spatial_unrolling_generator_with_hint(
                mem_scheme, input_settings.mac_array_info['array_size'], layer_info[layer_index],
                input_settings.unrolling_scheme_list)
            mem_scheme.fraction_spatial_unrolling = spatial_unrolling
            mem_scheme.greedy_mapping_flag = [False] * len(spatial_unrolling)
            mem_scheme.footer_info = [0] * len(spatial_unrolling)

        # spatial unrolling full search based on user-defined spatial_utilization_threshold
        else:
            even = (mem_scheme.mem_unroll['W'][0] == mem_scheme.mem_unroll['I'][0] == mem_scheme.mem_unroll['O'][0])
            if even and not input_settings.memory_unroll_fully_flexible:
                spatial_unrolling, flooring, mem_scheme, not_good = msg.spatial_unrolling_generator_even(
                    mem_scheme, input_settings.mac_array_info['array_size'], layer_info[layer_index],
                    input_settings.precision, input_settings.spatial_utilization_threshold,
                    input_settings.spatial_unrolling_mode)
            else:
                spatial_unrolling, flooring, mem_scheme, not_good = msg.spatial_unrolling_generator_uneven(
                    mem_scheme, input_settings.mac_array_info['array_size'], layer_info[layer_index],
                    input_settings.precision, input_settings.spatial_utilization_threshold,
                    input_settings.spatial_unrolling_mode, input_settings.memory_unroll_fully_flexible)
            mem_scheme.fraction_spatial_unrolling = spatial_unrolling
            mem_scheme.greedy_mapping_flag = [False] * len(spatial_unrolling)
            mem_scheme.footer_info = [0] * len(spatial_unrolling)

        mem_scheme.spatial_unrolling = spatial_unrolling
        mem_scheme.flooring = flooring
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
              mem_scheme_index + 1, '/', mem_scheme_count, ' SUG finished',
              '| Valid SU found:', len(spatial_unrolling))

    if not mem_scheme.spatial_unrolling:
        discard_mem_scheme = True
        print('Layer', layer_index, ': no valid spatial unrolling found')

        # Set the spatial unrolling count to 0 for this mem scheme + layer combination
        mem_scheme_str = 'M_%d' % (mem_scheme_index + 1)
        layer_str = 'L_%d' % (layer_index)
        multi_manager.list_su_count[mem_scheme_str][layer_str] = len(spatial_unrolling)

        return

    for su_idx, su_ in enumerate(spatial_unrolling):
        print('-SU', su_idx + 1, '/', len(mem_scheme.spatial_unrolling), mem_scheme.spatial_unrolling[su_idx])

    ''' input_settings.su_parallel_processing SU parallel '''
    TIMEOUT = 36000
    start = time.time()

    su_count = len(spatial_unrolling)
    su_number = list(range(0, su_count, 1))
    su_chunk_list = [su_number[i:i + input_settings.su_parallel_processing] for i in
                     range(0, su_count, input_settings.su_parallel_processing)]
    for su_chunk in su_chunk_list:
        procs = []
        su_zipped = [(su_idx, spatial_unrolling[su_idx]) for su_idx in su_chunk]
        for ii_su, su in su_zipped:
            p = Process(target=mem_scheme_su_evaluate,
                        args=(input_settings, layer, im2col_layer, layer_index, layer_info, mem_scheme,
                              mem_scheme_index, ii_su, su, su_count, im2col_need_correct, multi_manager))
            procs.append(p)

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
    if len(list_min_energy[mem_scheme_str][layer_str]['best_tm_each_su']) == 0:
        list_su_count[mem_scheme_str][layer_str] = 0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, str(input_settings.layer_filename.split('/')[-1]), 'L', layer_index, ', M',
                mem_scheme_index + 1, '/', mem_scheme_count, ' No TM found')
        return
    else:
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
        list_min_energy[mem_scheme_str][layer_str]['best_su_each_mem'].update(
            {best_en_mem_su_str: (best_en, best_en_ut)})
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
        list_max_ut_output[mem_scheme_str][layer_str]['best_su_each_mem'].update({best_ut_mem_su_str: best_ut_output})
        list_sim_time[mem_scheme_str][layer_str]['best_su_each_mem'].update({best_ut_mem_su_str: t2})

        # Delete all results in 'best_tm_each_su' if save_all_spatial_unrolling_result is false,
        # As this information is not required anymore
        if not input_settings.su_search_result_saving:
            del list_tm_count_en[mem_scheme_str][layer_str]['best_tm_each_su']
            del list_min_en_output[mem_scheme_str][layer_str]['best_tm_each_su']
            del list_min_energy[mem_scheme_str][layer_str]['best_tm_each_su']

            del list_tm_count_ut[mem_scheme_str][layer_str]['best_tm_each_su']
            del list_max_ut_output[mem_scheme_str][layer_str]['best_tm_each_su']
            del list_max_utilization[mem_scheme_str][layer_str]['best_tm_each_su']

            del list_sim_time[mem_scheme_str][layer_str]['best_tm_each_su']

        if not input_settings.fixed_spatial_unrolling:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            group_count = layer.G
            print(
                '{0:s} {1:s} L {2:d},  M {3:d},  SU {4:s}  Min En: ({5:.3E}, {6:.3f}, {7:.3E}) in all SUs and TMs'.format(
                    current_time, str(input_settings.layer_filename.split('/')[-1]), layer_index, mem_scheme_index + 1,
                    best_en_mem_su_str.split('_')[-1], int(group_count*best_en), best_en_ut, int(best_en_output.area)))
            print(
                '{0:s} {1:s} L {2:d},  M {3:d},  SU {4:s}  Max Ut: ({5:.3E}, {6:.3f}, {7:.3E}) in all SUs and TMs'.format(
                    current_time, str(input_settings.layer_filename.split('/')[-1]), layer_index, mem_scheme_index + 1,
                    best_ut_mem_su_str.split('_')[-1], int(group_count*best_ut_en), best_ut, int(best_ut_output.area)))


def mem_scheme_list_evaluate(input_settings, mem_scheme, mem_scheme_index, layers, multi_manager):

    mem_scheme_count = multi_manager.mem_scheme_count

    print('MEM HIERARCHY ', mem_scheme_index + 1, '/', mem_scheme_count)
    print('memory size:', mem_scheme.mem_size)
    if input_settings.memory_unroll_fully_flexible:
        print('memory unroll: unfixed')
    else:
        print('memory unroll:', mem_scheme.mem_unroll)
    print('memory share:', mem_scheme.mem_share)

    layer_chunk_list = [layers[i:i + input_settings.layer_parallel_processing] for i in
                        range(0, len(layers), input_settings.layer_parallel_processing)]
    if input_settings.im2col_enable:
        im2col_layer_chunk_list = [multi_manager.layers_im2col[i:i + input_settings.layer_parallel_processing] for i in
                                   range(0, len(multi_manager.layers_im2col), input_settings.layer_parallel_processing)]
    else:
        im2col_layer_chunk_list = layer_chunk_list

    for ii_layer_chunk, layer_chunk in enumerate(layer_chunk_list):
        procs = []
        for ii_layer_index, layer in enumerate(layer_chunk):
            layer_idx = ii_layer_chunk * input_settings.layer_parallel_processing + ii_layer_index
            layer_number = input_settings.layer_number[layer_idx]
            im2col_layer = im2col_layer_chunk_list[ii_layer_chunk][ii_layer_index]
            procs.append(Process(target=mem_scheme_evaluate,
                                 args=(input_settings, layer_number, layer, im2col_layer,
                                       mem_scheme, mem_scheme_index, multi_manager)))

        for p in procs: p.start()
        for p in procs: p.join()


def optimal_su_evaluate(input_settings, layers, multi_manager):
    """ Collect the optimum spatial unrolling results for all memory schemes """

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
        list_sim_time_en['best_mem_each_layer'][layer_str] = \
            list_sim_time[mem_scheme_str_en][layer_str]['best_su_each_mem'][best_en_mem_su_str]

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
        list_sim_time_ut['best_mem_each_layer'][layer_str] = \
            list_sim_time[mem_scheme_str_ut][layer_str]['best_su_each_mem'][best_ut_mem_su_str]

    # If multiple layers, iterate through all the memory hierarchies
    # to find the best hierarchy for the network of layers
    if len(input_settings.layer_number) > 1:

        best_network_energy = sys.float_info.max  # energy of minimal energy network
        best_network_energy_latency = sys.float_info.max  # latency of minimal energy network
        best_network_latency_energy = sys.float_info.max  # energy of minimal latency network
        best_network_latency = sys.float_info.max  # latency of minimal latency network
        best_mem_scheme_index_en = None
        best_mem_scheme_index_ut = None

        for mem_scheme_index in range(len(mem_scheme_sim)):

            mem_scheme_str = 'M_%d' % (mem_scheme_index + 1)

            tot_min_en_energy = 0
            tot_min_en_latency = 0

            tot_max_ut_energy = 0
            tot_max_ut_latency = 0

            for i, layer_index in enumerate(input_settings.layer_number):
                layer_str = 'L_%d' % layer_index

                group_count = layers[i].G

                # Energy part
                su_dict_en = list_min_energy[mem_scheme_str][layer_str]['best_su_each_mem']
                mem_scheme_su_str_en = list(su_dict_en.keys())[0]  # casts dict_keys type to list
                (min_en_en, min_en_ut) = list(su_dict_en.values())[0]  # casts dict_values type to list
                min_en_output = list_min_en_output[mem_scheme_str][layer_str]['best_su_each_mem'][mem_scheme_su_str_en]
                min_en_latency = min_en_output.utilization.latency_tot

                tot_min_en_energy += group_count * min_en_en
                tot_min_en_latency += min_en_latency

                # Utilization (latency) part
                su_dict_ut = list_max_utilization[mem_scheme_str][layer_str]['best_su_each_mem']
                mem_scheme_su_str_ut = list(su_dict_ut.keys())[0]  # casts dict_keys type to list
                (max_ut_en, max_ut_ut) = list(su_dict_ut.values())[0]  # cast dict_values type to list
                max_ut_output = list_max_ut_output[mem_scheme_str][layer_str]['best_su_each_mem'][mem_scheme_su_str_ut]
                max_ut_latency = max_ut_output.utilization.latency_tot

                tot_max_ut_energy += group_count * max_ut_en
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
            memory_area = int(min_en_output.area)

            print(
                '{0:s} {1:s} M {2:d}: Minimal energy for all layers:      (energy, latency, area) = ({3:.3E}, {4:.3E}, {5:.3E})'.format(
                    current_time, network_name, mem_scheme_index + 1, int(tot_min_en_energy), int(tot_min_en_latency), memory_area))
            print(
                '{0:s} {1:s} M {2:d}: Maximal utilization for all layers: (energy, latency, area) = ({3:.3E}, {4:.3E}, {5:.3E})'.format(
                    current_time, network_name, mem_scheme_index + 1, int(tot_max_ut_energy), int(tot_max_ut_latency), memory_area))

            # Set the multi_manager's parameter with the correct mem_scheme_index
        multi_manager.best_mem_scheme_index_en = best_mem_scheme_index_en
        multi_manager.best_mem_scheme_index_ut = best_mem_scheme_index_ut
