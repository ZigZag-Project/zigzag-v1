import math
import numpy as np
from itertools import combinations
from itertools import permutations
import time
import copy
import bsgutils as su
import classes as cls
from msg import update_mem_scheme_bw
from cost_model_funcs import get_operand_level_dynamic_mem_cost
from cost_model_funcs import get_active_mac_cost as get_mac_cost
import cost_model_funcs as cmf
import output_funcs as of
from copy import deepcopy

'''
The blocking scheme generator generates the optimal blocking schemes given the memory hierarchy

The sequence of operations that are done are 
    1. blocking scheme generation, 
    2. data reuse cleanup
    3. loop order combinations

The first function to be called is bsg, followed by data_reuse_cleanup and finally loop_order_combinations

The output of the loop_order_combinations function is a list of temporal loops ready to be used by the cost model
'''


def data_reuse_cleanup(layer_loop_info, list_blocking_schemes, spatial_unrolling):
    '''
    @param layer_loop_info: The size of the layer loops
    @type layer_loop_info: dict
    @param list_blocking_schemes: A list of blocking schemes
    @param spatial_unrolling: The spatial unrolling given by the memory scheme
    @return: a list_blocking_schemes without all the schemes that exhibited data reuse equal to 1 in intermediate levels

    Function steps
    ==============
        1. define the data reuse of WEIGHTS AND OUTPUTS. INPUTs are not considered as of now, will be implemented later
        2. look for intermediate levels (threfore excluding level 0 and last level) that have data reuse == 1. If it is the case, the scheme is discarded
        3. if no data reuse == 1 in intermediate levels, append the scheme to the good_list list
    '''

    good_list = []

    spatial_loop = cls.SpatialLoop.extract_loop_info(spatial_unrolling)
    layer = cls.Layer.extract_layer_info(layer_loop_info)

    for bs in list_blocking_schemes:
        temporal_loop = cls.TemporalLoop.extract_loop_info(layer, bs, spatial_loop)
        data_reuse = {'W': [spatial_loop.unit_duplicate['W'][0]],
                      'I_base': [spatial_loop.unit_duplicate['I'][0]],
                      'I': [spatial_loop.unit_duplicate['I'][0]],
                      'I_zigzag': [spatial_loop.unit_duplicate['I'][0]],
                      'O': [spatial_loop.unit_duplicate['O'][0]]}
        for operand in ['W', 'I_base', 'I', 'O']:
            data_reuse[operand].extend(temporal_loop.irrelevant_loop[operand])

        # DATA REUSE CLEANING IS DONE ONLY FOR WEIGHT AND OUTPUT
        discard = False

        for operand in ['W', 'O']:
            for level, level_reuse in enumerate(data_reuse[operand][1:-1]):
                if level_reuse == 1:
                    discard = True
                    break
                # TODO DECREASING DATA REUSE CONDITION
                # if level + 1 >= 2 and level_reuse > data_reuse[operand][level]:
                #     discard = True
                #     break
            if discard:
                break
        if not discard:
            good_list.append(bs)
    # print('\n', len(good_list), len(list_blocking_schemes))
    return good_list


def comb_order_combinations(virtual_level):
    tmp_virtual_level = []
    operand_irrelevant = {'W': [3, 4, 7], 'I': [6], 'O': [1, 2, 5]}
    for pf in virtual_level:
        if pf[0] not in [x[0] for x in tmp_virtual_level]:
            c = np.prod([x[1] for x in virtual_level if x[0] == pf[0]])
            tmp_virtual_level.append(tuple([pf[0], c]))
    # virtual_level_comb = list(permutations(tmp_virtual_level))

    virtual_level_comb = []

    for operand in ['W', 'I', 'O']:
        tmp_virtual_level_aux = copy.deepcopy(tmp_virtual_level)
        tmp_virtual_level_irrel = [lpf for lpf in tmp_virtual_level_aux if lpf[0] in operand_irrelevant[operand]]
        tmp_virtual_level_aux = [lpf for lpf in tmp_virtual_level_aux if lpf[0] not in operand_irrelevant[operand]]
        virtual_level_irrel_perm = list(permutations(tmp_virtual_level_irrel))
        for vl in virtual_level_irrel_perm:
            # for op in ['W', 'I', 'O']:
            virtual_level_comb.append(list(vl) + tmp_virtual_level_aux)

    return virtual_level_comb


def virtual_scheme_access_count(blocking_scheme, spatial_unrolling, layer):
    spatial_loop = cls.SpatialLoop.extract_loop_info(spatial_unrolling)
    temporal_loop = cls.TemporalLoop.extract_loop_info(layer, blocking_scheme, spatial_loop)
    mem_level = {}
    operand_irrelevant = {'W': [3, 4, 7], 'I': [6], 'O': [1, 2, 6]}

    for op in ['W', 'I', 'O']:
        mem_level[op] = [[]] * blocking_scheme[op].__len__()

    mem_access_total_element = {'W': [],
                                'I': [],
                                'O': [],
                                'O_final': [],
                                'O_partial': []}
    mem_read = {'W': [],
                'I': []}

    for operand in ['W', 'I']:
        for level in range(mem_level[operand]):
            mem_read[operand].append(np.prod(
                temporal_loop.irrelevant_loop[operand][level:mem_level[operand]] +
                spatial_loop.unit_duplicate[operand][level + 1:mem_level[operand] + 1]
            ).item())

        for level in range(mem_level[operand]):
            if level == mem_level[operand] - 1:
                ''' 
                For now, we don't consider the writing access from outside peripherals to the top-level memory. 
                '''
                mem_write = 0
            else:
                mem_write = mem_read[operand][level + 1]

            mem_access_total_element[operand].append(
                [int(mem_read[operand][level] * layer.total_data_size[operand]),
                 int(mem_write * layer.total_data_size[operand])])

    mem_read_L = []
    mem_write_L = []
    mem_read_H = []
    mem_write_H = []

    total_num_of_output = layer.total_data_size['O']

    for level in range(mem_level['O']):
        mem_write_L.append(np.prod(
            temporal_loop.irrelevant_loop['O'][level:mem_level['O']] +
            spatial_loop.unit_duplicate['O'][level + 1:mem_level['O'] + 1]
        ).item())

        mem_read_L.append(mem_write_L[level] - 1)

    for level in range(mem_level['O']):
        if level == mem_level['O'] - 1:
            ''' 
            For now, we don't consider the writing access from outside peripherals to the top-level memory. 
            '''
            mem_read_H.append(0)
            mem_write_H.append(0)
        else:
            mem_read_H.append(mem_write_L[level + 1])
            mem_write_H.append(mem_read_L[level + 1])

        mem_access_total_element['O'].append(
            [(mem_read_L[level] * total_num_of_output,
              mem_write_L[level] * total_num_of_output),
             (mem_read_H[level] * total_num_of_output,
              mem_write_H[level] * total_num_of_output)])

    '''
    Distinguish partial output and final output.
    '''

    for level in range(mem_level['O']):
        if (mem_read_L[level], mem_write_L[level]) == (0, 1):
            '''
            Final outputs are only written once from lower level of memory to current level of memory.
            '''
            mem_access_total_element['O_final'].append(
                [(mem_read_L[level] * total_num_of_output, mem_write_L[level] * total_num_of_output),
                 (mem_read_H[level] * total_num_of_output, mem_write_H[level] * total_num_of_output)])

            mem_access_total_element['O_partial'].append([(0, 0), (0, 0)])

        elif (mem_read_H[level], mem_write_H[level]) == (1, 0):
            '''
            Partial outputs are written multiple times from lower level of memory to current level of memory.
            Final outputs are only read once from current level of memory to higher level of memory.
            '''
            mem_access_total_element['O_final'].append(
                [(0, 0), (mem_read_H[level] * total_num_of_output, mem_write_H[level] * total_num_of_output)])

            mem_access_total_element['O_partial'].append(
                [(mem_read_L[level] * total_num_of_output, mem_write_L[level] * total_num_of_output), (0, 0)])

        else:
            '''
            Partial outputs are written multiple times from lower level of memory to current level of memory.
            Partial outputs are read multiple times from current level of memory to higher level of memory.
            '''
            mem_access_total_element['O_final'].append([(0, 0), (0, 0)])

            mem_access_total_element['O_partial'].append(
                [(mem_read_L[level] * total_num_of_output, mem_write_L[level] * total_num_of_output),
                 (mem_read_H[level] * total_num_of_output, mem_write_H[level] * total_num_of_output)])

    for operand in ['W', 'I']:
        mem_access_total_element[operand][0][0] = \
            mem_access_total_element[operand][0][0] // temporal_loop.MAC_level_stationary[operand]

    return mem_access_total_element


def loop_order_combinations(blocking_scheme):
    '''

    @param blocking_scheme: A single blocking scheme
    @return: A list of possible loop orders derived from the input blocking scheme

    The function does not generate all possible loop orderings but finds all of those that maximize stationarity for the lowest levels
    In order to maximize stationarity, for a given operand and memory level, all the irrelevant loops should be the innermost loops.
    Given the irregular nature of the blocking schemes, loop ordering is done only by successive I{virtual levels}
    In order to choose which operand and which memory level prioritize for the stationarity in the virtual level, 2 main criteria are considered:
        1. If there's an irrelevant loop chain already present from the virtual level below, the operand wrt which maximize stationarity is that one
        2. If there's a memory level jump between virtual level, that operand is chosen
        3. If none of the above happens, the smallest memory of the virtual level is prioritized.
    The criterias are in order of priority.
    If in the prioritized virtual level there are no irrelevant loops, all possible permutations of the loops in the virtual level are considered
    If in the prioritized virutal level there is a set of irrelevant loops, all possible permutations of the loops in the set are considered.
    If multiple virtual levels meet the same highest criteria, then they are all considered.

    '''
    # bs_next = []
    # finished = False
    # tmp_blocking_scheme = copy.deepcopy(blocking_scheme)
    # lo_ok = True
    #
    # length_bs = {
    #     'W': len(blocking_scheme['W']),
    #     'I': len(blocking_scheme['I']),
    #     'O': len(blocking_scheme['O']),
    # }
    # bsx = {
    #     'W': [[]],
    #     'I': [[]],
    #     'O': [[]]
    # }
    #
    # bs_next.append(bsx)
    #
    # while not finished:
    #     if any(tmp_blocking_scheme[op] == [[]] for op in tmp_blocking_scheme):
    #         min_virtual_roof = 0
    #     else:
    #         try:
    #             min_virtual_roof = min([len(tmp_blocking_scheme[op][0]) for op in tmp_blocking_scheme])
    #         except IndexError:
    #             print(tmp_blocking_scheme)
    #             min_virtual_roof = 0
    #     min_virtual_roof_list = []
    #     # Find lowest memory level to be virtualized
    #
    #     for operand in tmp_blocking_scheme:
    #         if tmp_blocking_scheme[operand]:
    #             if len(tmp_blocking_scheme[operand][0]) == min_virtual_roof:
    #                 min_virtual_roof_list.append([operand, len(tmp_blocking_scheme[operand][0])])
    #     virtual_level = copy.deepcopy(tmp_blocking_scheme[min_virtual_roof_list[0][0]][0])
    #
    #     for operand in tmp_blocking_scheme:
    #         for pf in virtual_level:
    #             try:
    #                 tmp_blocking_scheme[operand][0].remove(pf)
    #             except IndexError:
    #                 # print(blocking_scheme)
    #                 lo_ok = False
    #                 return lo_ok, bs_next
    #             except ValueError:
    #                 # print(blocking_scheme)
    #                 lo_ok = False
    #                 return lo_ok, bs_next
    #
    #     tmp_virtual_level = []
    #     for pf in virtual_level:
    #         if pf[0] not in [x[0] for x in tmp_virtual_level]:
    #             c = np.prod([x[1] for x in virtual_level if x[0] == pf[0]])
    #             tmp_virtual_level.append(tuple([pf[0], c]))
    #
    #     virtual_level_comb = list(permutations(tmp_virtual_level))
    #
    #     bs_old = copy.deepcopy(bs_next)
    #     bs_next = []
    #     for bs in bs_old:
    #         for vl in virtual_level_comb:
    #             tmp_bs = copy.deepcopy(bs)
    #             for operand in ['W', 'I', 'O']:
    #                 tmp_bs[operand][len(bs[operand]) - 1] += vl
    #                 if operand in [opx[0] for opx in min_virtual_roof_list]:
    #                     if len(tmp_bs[operand]) != length_bs[operand]:
    #                         tmp_bs[operand].append([])
    #             bs_next.append(tmp_bs)
    #
    #     for operand in tmp_blocking_scheme:
    #         if tmp_blocking_scheme[operand]:
    #             if not tmp_blocking_scheme[operand][0]:
    #                 tmp_blocking_scheme[operand].remove(tmp_blocking_scheme[operand][0])
    #
    #     if all(not tmp_blocking_scheme[op] for op in tmp_blocking_scheme):
    #         finished = True
    #
    # return lo_ok, bs_next
    bs_next = []
    finished = False
    tmp_blocking_scheme = copy.deepcopy(blocking_scheme)
    lo_ok = True

    operand_irrelevant = {'W': [3, 4, 7], 'I': [6], 'O': [1, 2, 5]}

    length_bs = {
        'W': len(blocking_scheme['W']),
        'I': len(blocking_scheme['I']),
        'O': len(blocking_scheme['O']),
    }
    bsx = {
        'W': [[]],
        'I': [[]],
        'O': [[]]
    }

    bs_next.append(bsx)
    tbs_removed_level = []
    old_irrelevant_chain = {'W': False, 'I': False, 'O': False}
    irrelevant_chain = {'W': False, 'I': False, 'O': False}

    '''
    The while loop considers successive virtual loops until there are no more
    '''
    while not finished:
        if any(tmp_blocking_scheme[op] == [[]] for op in tmp_blocking_scheme):
            min_virtual_roof = 0
        else:
            try:
                min_virtual_roof = min([len(tmp_blocking_scheme[op][0]) for op in tmp_blocking_scheme])
            except IndexError:
                min_virtual_roof = 0
        min_virtual_roof_list = []

        # Find lowest memory level to be virtualized
        for operand in tmp_blocking_scheme:
            if tmp_blocking_scheme[operand]:
                if len(tmp_blocking_scheme[operand][0]) == min_virtual_roof:
                    min_virtual_roof_list.append([operand, len(tmp_blocking_scheme[operand][0])])

        virtual_level = copy.deepcopy(tmp_blocking_scheme[min_virtual_roof_list[0][0]][0])
        for operand in tmp_blocking_scheme:
            if all([x[0] in operand_irrelevant[operand] for x in virtual_level]):
                old_irrelevant_chain[operand] = True
            else:
                old_irrelevant_chain[operand] = False
            for pf in virtual_level:
                try:
                    tmp_blocking_scheme[operand][0].remove(pf)
                except IndexError:
                    lo_ok = False
                    return lo_ok, bs_next
                except ValueError:
                    lo_ok = False
                    return lo_ok, bs_next

        bs_old = copy.deepcopy(bs_next)
        bs_next = []

        for vr in min_virtual_roof_list:

            tmp_virtual_level = []
            '''
            If multiple prime factors of the same loop are present, they are multiplied.
            For example (K,2) and (K,2) will become (K,4)
            '''
            for pf in virtual_level:
                if pf[0] not in [x[0] for x in tmp_virtual_level]:
                    c = np.prod([x[1] for x in virtual_level if x[0] == pf[0]])
                    tmp_virtual_level.append(tuple([pf[0], c]))
            '''
            The priority operand is found following the previously mentioned criteria.
            Since there might be multiple operands that respect the highest criteria, a list of priority operands is defined
            '''
            priority_op_list = []

            # Irrelevant chain criteria
            for operand in ['W', 'I', 'O']:
                if irrelevant_chain[operand]:
                    priority_op_list.append(operand)

            irrelevant_chain = copy.deepcopy(old_irrelevant_chain)

            # Memory level jump criteria
            if not priority_op_list:
                priority_op_list += tbs_removed_level

            # Smallest v.m. criteria
            if not priority_op_list:
                priority_op_list += [vr[0]]

            '''
            For each priority operand in the list, the possible permutations of irrelevant loops are considered
            '''
            for priority_operand in priority_op_list:

                min_vr_irr_subset = []

                for pf in tmp_virtual_level:
                    if pf[0] in operand_irrelevant[priority_operand]:
                        min_vr_irr_subset.append(pf)

                virtual_level_comb = []
                min_vr_irr_comb_list = list(permutations(min_vr_irr_subset))
                for vr_irr_comb in min_vr_irr_comb_list:
                    tvl = copy.deepcopy(tmp_virtual_level)
                    for pf in vr_irr_comb:
                        tvl.remove(pf)
                        tvl.insert(0, pf)
                    virtual_level_comb.append(tvl)
                if not virtual_level_comb:
                    virtual_level_comb = list(permutations(tmp_virtual_level))

                for bs in bs_old:
                    for vl in virtual_level_comb:
                        tmp_bs = copy.deepcopy(bs)
                        for operand in ['W', 'I', 'O']:
                            tmp_bs[operand][len(bs[operand]) - 1] += vl
                            if operand in [opx[0] for opx in min_virtual_roof_list]:
                                if len(tmp_bs[operand]) != length_bs[operand]:
                                    tmp_bs[operand].append([])
                        if tmp_bs not in bs_next:
                            bs_next.append(tmp_bs)

                # Update tmp_blocking_scheme for next iteration
                for operand in tmp_blocking_scheme:
                    if tmp_blocking_scheme[operand]:
                        if not tmp_blocking_scheme[operand][0]:
                            tmp_blocking_scheme[operand].remove(tmp_blocking_scheme[operand][0])
                            tbs_removed_level.append(operand)
                # Check if finished
        if all(not tmp_blocking_scheme[op] for op in tmp_blocking_scheme):
            finished = True

    return lo_ok, bs_next


def bsg(mem_size, mem_share, precision, utilization_rate, layer_loop_info, spatial_unrolling, layer_info, mem_scheme,
        hw_spec):
    t1 = time.time()
    total_number_of_schemes = 0
    operand_irrelevant = {
        'W': [3, 4, 7],
        'O': [1, 2, 5],
        'I': [6]
    }

    '''
    loops_pf contains prime factors for each loop type.
    After each loop assignment the relative list is updated
    '''
    loops_pf = {
        7: [],
        6: [],
        5: [],
        4: [],
        3: [],
        2: [],
        1: []
    }
    '''
    For each operand and memory level defines the effective size of blockings that it can contain
    '''
    mem_block_size = {
        'W': [],
        'O': [],
        'I': []
    }
    '''
    Auxiliary term that stores the temporary roof values.
    For each operand it is defined to what memory level the roof belongs and how much space is still left to be assigned
    'operand' : [memory_level, roof_value]
    '''
    roof = {
        'O': [0, 0],
        'I': [0, 0],
        'W': [0, 0]
    }

    '''
    Init blocking scheme
    '''
    bs = {
        'W': [[] for i in range(len(mem_size['W']))],
        'O': [[] for i in range(len(mem_size['O']))],
        'I': [[] for i in range(len(mem_size['I']))],
    }
    for op in spatial_unrolling:
        for level in range(0, len(spatial_unrolling[op])):
            if spatial_unrolling[op][level]:
                for unroll in range(0, len(spatial_unrolling[op][level])):
                    sp = su.prime_factors(spatial_unrolling[op][level][unroll][1])
                    bs[op][level] += [tuple([spatial_unrolling[op][level][unroll][0], j]) for j in sp]
    '''
    Assign prime factors for each loop type
    '''
    ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}

    for loop_type in loops_pf:
        loops_pf[loop_type] = su.prime_factors(layer_loop_info[ll[loop_type]])

    loops_pf_irrelevant_unrolled = {'W': [], 'I': [], 'O': []}

    for level in range(0, len(spatial_unrolling['W'])):
        for unroll in range(0, len(spatial_unrolling['W'][level])):
            # if spatial_unrolling[operand][level][unroll][0] in operand_irrelevant[operand]:
            sp = su.prime_factors(spatial_unrolling['W'][level][unroll][1])
            for j in sp:
                # loops_pf_irrelevant_unrolled[operand].append(tuple([spatial_unrolling[operand][level][unroll][0], j]))
                try:
                    loops_pf[spatial_unrolling['W'][level][unroll][0]].remove(j)
                except:
                    raise ValueError("The spatial unrolling given has unroll dimensions that can not be found "
                                     "in the layer. Please check the spatial unrolling setting and correct it.")

    '''
    Assign relative memory sizes for each operand at each memory level
    '''
    for operand in mem_size:
        for level in range(0, len(mem_size[operand])):
            mem_block_size[operand].append(math.floor(mem_size[operand][level] / precision[operand]))
    '''
    Initialize roof values
    '''
    for operand in mem_block_size:
        roof[operand][1] = mem_block_size[operand][0]
    roof = su.update_roof(bs, spatial_unrolling, [], roof, mem_share, mem_size, precision, operand_irrelevant, loops_pf,
                          layer_loop_info)
    r_op, r_lev = roof.keys(), roof.values()
    r_lev, r_size = zip(*r_lev)

    min_roof_aux = ['', 0, max(r_size)]
    next_min_roof = ['', 0, max(r_size)]
    last_roof = ['', 0, max(r_size)]

    root_node = su.SchedulerNode(bs, roof, loops_pf)
    layer = []
    nextLayer = [root_node]
    finalLayer = []
    finished = False
    rep = 0
    old_cleanLayer = []
    no_fitting = [True]
    old_no_fitting = []
    iteration_layers = 0
    while not finished:
        # Check if all the nodes in nextLayer are leaf nodes
        cleanLayer = su.cleaner(nextLayer)
        if any(nodes.leaf_over == True for nodes in cleanLayer):
            if all(any(old_node == node for node in cleanLayer) for old_node in
                   old_cleanLayer):
                finalLayer = [bn for bn in cleanLayer if bn.leaf_over == True]
                finished = True
                continue
        if cleanLayer:
            if all(any(old_node == node for old_node in old_cleanLayer) for node in
                   cleanLayer):
                rep += 1
                if rep == 5:
                    # print('no fitting found!')
                    finished = True
                    continue
        old_cleanLayer = copy.deepcopy(cleanLayer)
        if all(nodes.leaf_over == True for nodes in cleanLayer):
            finalLayer = [bn for bn in cleanLayer if bn.leaf_over == True]
            finished = True
            continue

        old_no_fitting = copy.deepcopy(no_fitting)
        no_fitting = []
        layer = copy.deepcopy(cleanLayer)
        nextLayer = []
        iteration_layers += 1
        for z in range(0, len(layer)):

            if layer[z].leaf_over:
                leaf_node = su.SchedulerNode(layer[z].LPF_scheme, layer[z].roof, layer[z].loops_pf)
                leaf_node.set_leaf_over()
                nextLayer.append(leaf_node)
                continue
            roof = layer[z].roof
            blocking_scheme = layer[z].LPF_scheme
            loops_pf = layer[z].loops_pf

            r_op, r_lev = roof.keys(), roof.values()
            r_lev, r_size = zip(*r_lev)
            m_size = max([len(mem_size['I']), len(mem_size['W']), len(mem_size['O'])])

            min_roof_aux = ['', m_size, max(r_size)]
            tmp_min_roof = ['', m_size, max(r_size)]

            '''
            Find smallest roof
            '''

            # Find all different minroofs
            min_roof_list = []
            for operand in roof:
                if any((roof[roofval][0] < roof[operand][0]) for roofval in
                       roof):
                    if roof[operand][0] == (len(mem_size[operand]) - 1):
                        continue
                if (roof[operand][0] == (len(mem_size[operand]) - 1) and (
                        roof[operand][1] == 1 or roof[operand][1] == 0)):
                    continue
                else:
                    if roof[operand][1] <= min_roof_aux[2]:
                        min_roof_aux[0] = operand
                        min_roof_aux[1] = roof[operand][0]
                        min_roof_aux[2] = roof[operand][1]
            for operand in roof:
                if roof[operand][0] == min_roof_aux[1] and roof[operand][1] == min_roof_aux[2]:
                    tmp_min_roof = ['', m_size, r_size]
                    tmp_min_roof[0] = operand
                    tmp_min_roof[1] = roof[operand][0]
                    tmp_min_roof[2] = roof[operand][1]
                    min_roof_list.append(tmp_min_roof)

            mr_list_operand = [mr[0] for mr in min_roof_list]

            '''
            Find different combinations of relevant loop blockings wrt to the operand of the smallest roof that fit in the smallest
            roof AND are above the utilization rate
            '''
            for min_roof in min_roof_list:
                # Check if min roof belongs to a shared level of memory. If so, create a list with all shared memory levels related to min roof
                shared_min_roof_levels = [tuple([min_roof[0], min_roof[1]])]
                for shared_set in mem_share:
                    if tuple([min_roof[0], min_roof[1]]) in mem_share[shared_set]:
                        shared_min_roof_levels = mem_share[shared_set]

                fitting_combination = []
                min_list_fitting_combinations = []
                loop_blocks = []

                # List all relevant loop blockings of min roof operand
                for loop_type in loops_pf:
                    for i in range(0, len(loops_pf[loop_type])):
                        loop_blocks.append(tuple([loop_type, loops_pf[loop_type][i]]))
                # List fitting combination of relevant blockings wrt each min roof in the min roof list
                roof_list = [[rf, roof[rf][0], roof[rf][1]] for rf in roof]
                if all([roof[rf][0] == len(mem_size[rf]) - 1 for rf in roof]):
                    k_range = range(len(loop_blocks), len(loop_blocks) + 1)
                else:
                    k_range = range(0, len(loop_blocks) + 1)
                # min_list_fitting_combinations = []
                for k in k_range:
                    # print('\r  |-> layer[z] : ', z,'/', len(layer),' subseq : ', k, '/',len(loop_blocks)+1,end='')
                    fitting_combination = []
                    min_list_fitting_combinations = []
                    tmp_comb = combinations(loop_blocks, k)
                    comb = []
                    for x in tmp_comb:
                        if x not in comb:
                            comb.append(x)

                    for j in range(0, len(comb)):
                        is_fit = True
                        for r in roof_list:
                            is_min_roof = False
                            if r[0] in mr_list_operand:
                                is_min_roof = True
                            is_fit = su.check_comb_fit(blocking_scheme, spatial_unrolling, comb[j], r, mem_size,
                                                       mem_share,
                                                       utilization_rate, precision, operand_irrelevant, is_min_roof,
                                                       layer_loop_info)
                            if not is_fit:
                                break
                        if not is_fit:
                            continue
                        fitting_combination.append(comb[j])
                    min_list_fitting_combinations = fitting_combination

                    if not min_list_fitting_combinations:
                        no_fitting.append(True)
                        # No need to update loops_pf
                        new_loops_pf = copy.deepcopy(loops_pf)
                        tt = sum([len(x) for x in new_loops_pf.values()])
                        # Update roof
                        # try:
                        # Level up is usually one. It's value is zero when the memory level of min roof is the last memory level for its operand

                        new_tmp_roof = copy.deepcopy(roof)
                        for p in range(0, len(shared_min_roof_levels)):
                            level_up = 1
                            if shared_min_roof_levels[p][1] == len(
                                    mem_block_size[shared_min_roof_levels[p][0]]) - 1:
                                level_up = 0
                            new_tmp_roof[shared_min_roof_levels[p][0]][0] = shared_min_roof_levels[p][1] + level_up
                        new_roof = su.update_roof(blocking_scheme, spatial_unrolling, [], new_tmp_roof, mem_share,
                                                  mem_size, precision,
                                                  operand_irrelevant, new_loops_pf, layer_loop_info)

                        # No need to update blocking scheme
                        new_blocking_scheme = copy.deepcopy(blocking_scheme)

                        # Generate new node in the blocking scheme tree, add it to the list
                        blocking_node = su.SchedulerNode(new_blocking_scheme, new_roof, new_loops_pf)

                        nextLayer.append(blocking_node)
                    else:
                        no_fitting.append(False)
                        for h in range(0, len(min_list_fitting_combinations)):

                            shared_roof = [smr[0] for smr in shared_min_roof_levels]
                            non_shared_roof_list = [[rf, roof[rf][0]] for rf in roof if rf not in shared_roof]
                            min_comb = float("inf")
                            best_comb = []
                            good_scheme_list = []
                            comb_p = comb_order_combinations(min_list_fitting_combinations[h])
                            comb_permutations = []
                            for combx in comb_p:
                                if combx not in comb_permutations: comb_permutations.append(combx)
                            total_number_of_schemes += len(comb_permutations)
                            for comb_level0 in comb_permutations:
                                new_blocking_scheme = copy.deepcopy(blocking_scheme)
                                for op in spatial_unrolling:
                                    for level in range(0, len(spatial_unrolling[op])):
                                        if spatial_unrolling[op][level]:
                                            for unroll in range(0, len(spatial_unrolling[op][level])):
                                                sp = su.prime_factors(spatial_unrolling[op][level][unroll][1])
                                                for i in range(0, len(sp)):
                                                    try:
                                                        new_blocking_scheme[op][level].remove(
                                                            tuple([spatial_unrolling[op][level][unroll][0], sp[i]]))
                                                    except IndexError:
                                                        continue
                                level0 = comb_level0
                                # level up is usually one, it is assigned to zero only when the min roof is already the last level of memory for the operand
                                for p in range(0, len(shared_min_roof_levels)):
                                    level_up = 1
                                    if shared_min_roof_levels[p][1] == len(
                                            mem_block_size[shared_min_roof_levels[p][0]]) - 1:
                                        level_up = 0

                                    # Concatenate the generated blocking scheme with the blocking scheme up to this point
                                    new_blocking_scheme[shared_min_roof_levels[p][0]][
                                        shared_min_roof_levels[p][1]] = new_blocking_scheme[
                                                                            shared_min_roof_levels[p][0]][
                                                                            shared_min_roof_levels[p][1]] + list(level0)

                                for p in range(0, len(non_shared_roof_list)):
                                    # Concatenate the generated blocking scheme with the blocking scheme up to this point
                                    new_blocking_scheme[non_shared_roof_list[p][0]][non_shared_roof_list[p][1]] = \
                                        new_blocking_scheme[non_shared_roof_list[p][0]][
                                            non_shared_roof_list[p][1]] + list(level0)

                                bs = copy.deepcopy(new_blocking_scheme)

                                spatial_loop = cls.SpatialLoop.extract_loop_info(spatial_unrolling, layer_loop_info)
                                temporal_loop = cls.TemporalLoop.extract_loop_info(layer_info, bs, spatial_loop)
                                loop = cls.Loop.extract_loop_info(layer_info, temporal_loop, spatial_loop,
                                                                  hw_spec.precision, False)
                                mem_bw_bit = None
                                mem_scheme_cost = copy.deepcopy(mem_scheme)
                                utilization = cls.Utilization.get_utilization(layer_info, temporal_loop, spatial_loop,
                                                                              loop, hw_spec.mac_array_info,
                                                                              mem_scheme_cost.mem_size,
                                                                              mem_scheme_cost.mem_share,
                                                                              mem_scheme_cost.mem_type,
                                                                              hw_spec.mac_array_stall,
                                                                              hw_spec.precision, mem_bw_bit)
                                mem_scheme_cost_non_shared, mem_scheme_cost_shared = update_mem_scheme_bw(
                                    mem_scheme_cost, utilization)
                                msc = mem_scheme_cost_non_shared
                                ii = False
                                utilization = cls.Utilization.get_utilization(layer_info, temporal_loop, spatial_loop,
                                                                              loop,
                                                                              hw_spec.mac_array_info, msc.mem_size,
                                                                              msc.mem_share, msc.mem_type,
                                                                              hw_spec.mac_array_stall,
                                                                              hw_spec.precision, msc.mem_bw)
                                total_cost = 0
                                operand_cost = {'W': [], 'I': [], 'O': []}
                                for operand in bs:
                                    for level in range(0, len(bs[operand])):
                                        total_cost += get_operand_level_dynamic_mem_cost(operand, level, loop,
                                                                                         msc.mem_cost, msc,
                                                                                         hw_spec.precision, utilization,
                                                                                         0)
                                total_cost += get_mac_cost(layer_info, hw_spec.mac_array_info['single_mac_energy'])
                                if (total_cost <= min_comb):
                                    if total_cost < min_comb:
                                        best_comb = []
                                        good_scheme_list = []
                                    good_scheme = copy.deepcopy(bs)
                                    for op in spatial_unrolling:
                                        for level in range(0, len(spatial_unrolling[op])):
                                            if spatial_unrolling[op][level]:
                                                for unroll in range(0, len(spatial_unrolling[op][level])):
                                                    sp = su.prime_factors(spatial_unrolling[op][level][unroll][1])
                                                    try:
                                                        for s in sp:
                                                            good_scheme[op][level].insert(
                                                                0, tuple([spatial_unrolling[op][level][unroll][0], s])
                                                            )
                                                    except IndexError:
                                                        # print(bs[op][level])
                                                        continue
                                    min_comb = total_cost
                                    good_scheme_list.append(good_scheme)
                                    best_comb.append(comb_level0)

                        for ii_good_scheme, good_scheme in enumerate(good_scheme_list):
                            # print(best_comb[ii_good_scheme])
                            new_loops_pf = copy.deepcopy(loops_pf)
                            # Given remaining loop prime factors, remove those assigned in this combination to min roof
                            for i in range(0, len(best_comb[ii_good_scheme])):
                                pf_size_list = su.prime_factors(best_comb[ii_good_scheme][i][1])
                                for pf_size in pf_size_list:
                                    new_loops_pf[best_comb[ii_good_scheme][i][0]].remove(
                                        pf_size)
                            tt = sum([len(x) for x in new_loops_pf.values()])
                            # Set temporary roof with selected fitting combination
                            tmp_roof = su.update_roof(blocking_scheme, spatial_unrolling,
                                                      list(best_comb[ii_good_scheme]), roof, mem_share,
                                                      mem_size, precision, operand_irrelevant, new_loops_pf,
                                                      layer_loop_info)

                            # No need to update loops pf

                            over = False
                            if all(new_loops_pf[loop_types] == [] for loop_types in new_loops_pf):
                                over = True

                            new_tmp_roof = copy.deepcopy(tmp_roof)
                            new_blocking_scheme = copy.deepcopy(blocking_scheme)

                            level0 = copy.deepcopy(best_comb[ii_good_scheme])

                            for rf in roof:
                                new_blocking_scheme[rf][roof[rf][0]] += copy.deepcopy(level0)
                            # for op in ['W','I','O']:
                            #     print(new_blocking_scheme[op])
                            for p in range(0, len(shared_min_roof_levels)):
                                level_up = 1
                                if shared_min_roof_levels[p][1] == len(
                                        mem_block_size[shared_min_roof_levels[p][0]]) - 1:
                                    level_up = 0
                                new_tmp_roof[shared_min_roof_levels[p][0]][0] = shared_min_roof_levels[p][1] + level_up

                            new_tmp_roof = copy.deepcopy(tmp_roof)
                            new_tmp_roof[shared_min_roof_levels[p][0]][0] = shared_min_roof_levels[p][1] + level_up
                            new_roof = su.update_roof(good_scheme, spatial_unrolling, [], new_tmp_roof, mem_share,
                                                      mem_size,
                                                      precision,
                                                      operand_irrelevant, new_loops_pf, layer_loop_info)

                            # Generate new node in the blocking scheme tree, add it to the list
                            blocking_node = su.SchedulerNode(good_scheme, new_roof, new_loops_pf)
                            if over:
                                blocking_node.set_leaf_over()
                            nextLayer.append(blocking_node)
                        # print('a',len(nextLayer))
                break

    list_blocking_schemes = []
    list_blocking_nodes = []

    for i, blocking_node in enumerate(finalLayer):
        # print('\r bs cleaning ', i, '/', len(finalLayer), end='', flush=True)
        good = True
        good = su.check_node(blocking_node.LPF_scheme, mem_size, operand_irrelevant, mem_share, precision, layer_loop_info,
                             utilization_rate)
        if good:
            if blocking_node.LPF_scheme not in list_blocking_schemes:
                list_blocking_schemes.append(blocking_node.LPF_scheme)
                list_blocking_nodes.append(blocking_node)

    for bs in list_blocking_schemes:
        for op in spatial_unrolling:
            for level in range(0, len(spatial_unrolling[op])):
                if spatial_unrolling[op][level]:
                    for unroll in range(0, len(spatial_unrolling[op][level])):
                        sp = su.prime_factors(spatial_unrolling[op][level][unroll][1])
                        for i in range(0, len(sp)):
                            try:
                                bs[op][level].remove(tuple([spatial_unrolling[op][level][unroll][0], sp[i]]))
                            except IndexError:
                                continue

    total = []
    total = list_blocking_schemes

    # print('\n  |-> total number of schemes analyzed: ', total_number_of_schemes)
    # print('\n  |-> loop order combinations: ', len(total))
    return (total), total_number_of_schemes


def get_input_data_reuse(pf_list, layer):
    ox_tot = 1
    fx_tot = 1
    oy_tot = 1
    fy_tot = 1
    k_tot = 1
    ox_fx_present = False
    oy_fy_present = False
    if 3 in [pf[0] for pf in pf_list] and 1 in [pf[0] for pf in pf_list]: ox_fx_present = True
    if 4 in [pf[0] for pf in pf_list] and 2 in [pf[0] for pf in pf_list]: oy_fy_present = True

    for pf in pf_list:
        if pf[0] == 6:
            k_tot *= pf[1]
        if pf[0] == 1 and ox_fx_present:
            fx_tot *= pf[1]
        if pf[0] == 3 and ox_fx_present:
            ox_tot *= pf[1]
        if pf[0] == 2 and oy_fy_present:
            fy_tot *= pf[1]
        if pf[0] == 4 and oy_fy_present:
            oy_tot *= pf[1]
    xval = (ox_tot * fx_tot) / (layer['SX'] * (ox_tot - 1) + layer['SFX'] * (fx_tot - 1) + 1)
    yval = (oy_tot * fy_tot) / (layer['SY'] * (oy_tot - 1) + layer['SFY'] * (fy_tot - 1) + 1)

    return xval * yval * k_tot

