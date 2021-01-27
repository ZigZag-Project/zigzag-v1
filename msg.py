import numpy as np
from itertools import combinations
from itertools import product
from itertools import combinations_with_replacement
from copy import deepcopy
import pickle
from classes import Layer
from math import log
import bsgutils as su
import sys


class MemoryNode:

    def __init__(self, memory_level, operand, cluster_level, fixed, unique_name=None):
        self.memory_level = memory_level
        self.operand = operand
        self.cluster_level = cluster_level
        self.fixed = fixed
        self.unique_name = unique_name
        self.read_from_above_cost = {}
        self.write_to_above_cost = {}

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            if self.cluster_level < other.cluster_level:
                return True
            elif self.cluster_level == other.cluster_level:
                if self.memory_level['size_bit'] < other.memory_level['size_bit']:
                    return True
            else:
                return False

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            if self.cluster_level > other.cluster_level:
                return True
            elif self.cluster_level == other.cluster_level:
                if self.memory_level['size_bit'] > other.memory_level['size_bit']:
                    return True
            else:
                return False

    def set_read_from_above_cost(self, op, cost):
        '''
        Method to set the cost of reading from the node that is directly above this one in the hierarchy.
        As this node might hold mulitple operands, we use a dictionary.
        '''
        self.read_from_above_cost[op] = cost

    def set_write_to_above_cost(self, op, cost):
        '''
        Method to set the cost of writing to the node that is directly above this one in the hierarchy.
        As this node might hold mulitple operands, we use a dictionary.
        '''
        self.write_to_above_cost[op] = cost


class MemorySchemeNode:

    def __init__(self, temp_comb):
        self.memory_scheme = set()
        self.temp_comb = temp_comb

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.memory_scheme == other.memory_scheme
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class MemoryScheme:
    spatial_unrolling = []
    flooring = []
    fraction_spatial_unrolling = []
    greedy_mapping_flag = []
    footer_info = []
    su_utilization = []

    col2im_spatial_unrolling = []
    col2im_flooring = []
    col2im_fraction_spatial_unrolling = []

    # the complete memory unrolling count
    # this parameter is used for total area estimation (<-> active area)
    mem_unroll_complete = []

    def __init__(self, mem_name, mem_size, mem_cost, mem_utilization_rate, mem_utilization_rate_fixed, mem_share,
                 mem_unroll, mem_fifo, mem_bw, mem_type, mem_area, mem_nbanks, nodes):
        self.mem_name = mem_name
        self.mem_size = mem_size
        self.mem_cost = mem_cost
        self.mem_utilization_rate = mem_utilization_rate
        self.mem_utilization_rate_fixed = mem_utilization_rate_fixed
        self.mem_share = mem_share
        self.mem_unroll = mem_unroll
        self.mem_fifo = mem_fifo
        self.mem_bw = mem_bw
        self.mem_type = mem_type
        self.mem_area = mem_area
        self.mem_nbanks = mem_nbanks
        self.nodes = nodes

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def set_spatial_unrolling_flooring(self, spatial_unrolling, flooring):
        self.spatial_unrolling = spatial_unrolling
        self.flooring = flooring

    def set_fraction_spatial_unrolling(self, fraction_spatial_unrolling):
        self.frac_spatial_unrolling = fraction_spatial_unrolling

    def set_im2col_parameters(self, col2im_spatial_unrolling, col2im_flooring, col2im_fraction_spatial_unrolling):
        self.col2im_spatial_unrolling = col2im_spatial_unrolling
        self.col2im_flooring = col2im_flooring
        self.col2im_fraction_spatial_unrolling = col2im_fraction_spatial_unrolling


def fix_best_scheme(old_best_scheme, new_best_scheme, mem_pool):
    # !! Only used for iterative search of architecture !!

    fixed_scheme = MemorySchemeNode([])
    not_fixed_scheme = MemorySchemeNode([])
    if old_best_scheme != None:
        for new_mem_level in new_best_scheme.memory_scheme:
            fixed = 0
            for old_mem_level in old_best_scheme.memory_scheme:
                new_mem_level_op_list = list(new_mem_level.operand)
                new_mem_level_op_list.sort()
                old_mem_level_op_list = list(old_mem_level.operand)
                old_mem_level_op_list.sort()
                if new_mem_level.memory_level['size_bit'] == old_mem_level.memory_level['size_bit'] and \
                        new_mem_level.memory_level['unroll'] == old_mem_level.memory_level['unroll'] and \
                        new_mem_level.memory_level['mem_type'] == old_mem_level.memory_level['mem_type'] and \
                        new_mem_level_op_list == old_mem_level_op_list:
                    tmp_mem_level = None
                    for mp in mem_pool:
                        if mp['size_bit'] == new_mem_level.memory_level['size_bit'] and new_mem_level.memory_level[
                            'mem_type'] == mp['mem_type']:
                            tmp_mem_level = mp
                            tmp_mem_level['unroll'] = new_mem_level.memory_level['unroll']
                            break
                    fixed = 1
                    fixed_scheme.memory_scheme.add(
                        MemoryNode(tmp_mem_level, new_mem_level.operand, new_mem_level.cluster_level, fixed))
                    break
            if fixed == 0:
                for mp in mem_pool:
                    if mp['size_bit'] == new_mem_level.memory_level['size_bit'] and new_mem_level.memory_level[
                        'mem_type'] == mp['mem_type']:
                        tmp_mem_level = mp
                        tmp_mem_level['unroll'] = new_mem_level.memory_level['unroll']
                        break
                fixed = 0
                not_fixed_scheme.memory_scheme.add(
                    MemoryNode(tmp_mem_level, new_mem_level.operand, new_mem_level.cluster_level, fixed))

    return fixed_scheme, not_fixed_scheme


def update_mem_pool(partial_pool, memory_pool, not_fixed_scheme, memory_hierarchy_ratio):
    # !! Only used for iterative search of architecture !!

    if partial_pool:
        max_mem = 0
        max_mem_level = None
        min_mem = float('inf')
        min_mem_level = None
        tmp_mem_node_list = []
        for mem in partial_pool:
            if mem['size_bit'] >= max_mem:
                max_mem = mem['size_bit']
                max_mem_level = deepcopy(mem)
            if mem['size_bit'] <= min_mem:
                min_mem = mem['size_bit']
                min_mem_level = deepcopy(mem)
        partial_pool.remove(max_mem_level)
        for memory in not_fixed_scheme.memory_scheme:
            if memory.memory_level['size_bit'] == max_mem and memory.fixed == 0:
                tmp_max_level = deepcopy(max_mem_level)
                tmp_max_level['unroll'] = memory.memory_level['unroll']
                mn = MemoryNode(tmp_max_level, memory.operand, memory.cluster_level, memory.fixed)
                tmp_mem_node_list.append(mn)
        min_mem_mhr = min_mem  # / memory_hierarchy_ratio
        max_mem = 0
        max_mem_level = None
        for mem in memory_pool:
            if mem['size_bit'] < min_mem_mhr and mem['size_bit'] >= max_mem:
                max_mem = mem['size_bit']
                max_mem_level = deepcopy(mem)
        if max_mem_level != None:
            partial_pool.append(max_mem_level)
    else:
        max_mem = 0
        max_mem_level = None
        tmp_mem_node_list = []
        for mem in memory_pool:
            if mem['size_bit'] >= max_mem:
                max_mem = mem['size_bit']
                max_mem_level = deepcopy(mem)
        partial_pool.append(max_mem_level)
        max_mem_mhr = max_mem  # / memory_hierarchy_ratio
        max_mem = 0
        max_mem_level = None
        for mem in memory_pool:
            if mem['size_bit'] < max_mem_mhr and mem['size_bit'] >= max_mem:
                max_mem = mem['size_bit']
                max_mem_level = deepcopy(mem)
        partial_pool.append(max_mem_level)

    return partial_pool, tmp_mem_node_list


def get_available_area(max_area, memory_scheme_node):
    occupied_area = 0
    for mem_node in memory_scheme_node.memory_scheme:
        occupied_area += min(mem_node.memory_level['area']) * mem_node.memory_level['unroll']
    available_area = (float(max_area) - occupied_area)
    return available_area


def array_mem_pool(mem_pool, array_size, area, PE_RF_size_threshold, tmp_mem_node_list, banking, L1_size, L2_size):
    # Generate pool of memories that include the unrolled versions across the PE array
    # if their bit size is lower than PE_RF_size_threshold
    # Only 2D array for now
    array_pool = []
    update_mem_pool = []  # deepcopy(mem_pool)
    for l1 in L1_size:
        for mem in mem_pool:
            if mem['size_bit'] > PE_RF_size_threshold:
                nbanks = l1 / mem['size_bit']
                if nbanks in banking and mem['area'][0] * nbanks < area:
                    tmp_mem = {'size_bit': l1, 'area': [mem['area'][0] * nbanks],
                               'unroll': mem['unroll'], 'nbanks': nbanks}
                    update_mem_pool.append(tmp_mem)
                    # print(l1)
                    # break
    for l2 in L2_size:
        for mem in mem_pool:
            if mem['size_bit'] > PE_RF_size_threshold:
                nbanks = l2 / mem['size_bit']
                if nbanks in banking:  # and nbanks %4 == 0:
                    tmp_mem = {'size_bit': l2, 'area': [mem['area'][0] * nbanks],
                               'unroll': mem['unroll'], 'nbanks': nbanks}
                    update_mem_pool.append(tmp_mem)
                    break
    for mem in mem_pool:
        if mem['size_bit'] <= PE_RF_size_threshold:
            tmp_mem = {'size_bit': mem['size_bit'], 'area': [mem['area'][0]],
                       'unroll': np.prod(array_size), 'nbanks': 1}
            size_list = [x['size_bit'] for x in update_mem_pool]
            if mem['size_bit'] not in size_list:
                update_mem_pool.append(tmp_mem)
    # for mem in update_mem_pool:
    #    if mem['size_bit'] > PE_RF_size_threshold:
    #        array_pool.append(deepcopy(mem))
    #

    return update_mem_pool


def fitting_memories(array_mem_pool, area, max_area, utilization_rate, L1_size, L2_size, occupied_area=0):
    # Create a list of combinations of memories drawn from array_mem_pool
    # that fit in max_area and are above the utilization_rate set
    # The memories DO NOT have operand(s) assigned to them yet

    fitting_comb = []
    if area == 0:
        return fitting_comb
    array_mem_pool_index = list(range(0, len(array_mem_pool)))
    for k in range(0, len(array_mem_pool) ** 2):
        print('\r memory combination ', k + 1, '/', len(array_mem_pool) + 1, ' fitting list:', len(fitting_comb),
                end="")        
        # Create combination with repetition of memories from array_mem_pool
        # The repetition is due to the fact that the same memory can hold a different operand
        mem_combinations = combinations_with_replacement(array_mem_pool_index, k)
        oldfclen = fitting_comb.__len__()
        for comb in mem_combinations:
            comb = list(comb)
            size_bit_list = [array_mem_pool[x]['size_bit'] for x in comb]
            unique_mems = [size_bit_list.count(x) for x in size_bit_list]
            L1_count = 0
            L2_count = 0
            sblx = []
            for ii_sbl, sbl in enumerate(size_bit_list):
                if sbl not in sblx:
                    if sbl in L1_size:
                        L1_count += unique_mems[ii_sbl]
                    if sbl in L2_size:
                        L2_count += unique_mems[ii_sbl]
                    sblx.append(sbl)
            # The following if prunes away those combinations of memories where
            # an equal memory is present more than three times.
            # This is due to the fact that there can't be more than 3 operand
            if unique_mems:
                if max(unique_mems) > 3:
                    continue
            if L1_count > 3 or L2_count > 3:
                continue
            comb_mem_area = [min(array_mem_pool[x]['area']) * array_mem_pool[x]['unroll'] for x in comb]
            # Check if the combination fits area and is above the utilization rate
            if np.sum(comb_mem_area) <= area:
                if (np.sum(comb_mem_area) + occupied_area) / float(max_area) >= utilization_rate:
                    mem_comb = [array_mem_pool[x] for x in comb]
                    if mem_comb not in fitting_comb:
                        fitting_comb.append(mem_comb)
        if oldfclen == fitting_comb.__len__() and oldfclen != 0:
            break
    return fitting_comb


def memory_scheme_generator_cluster(memory_comb, memory_pool, array_dimension, mem_sn, max_area, utilization_rate_area,
                                    tmp_mem_node_list, PE_RF_size_threshold, banking, L1_size, L2_size):
    memory_schemes_list = []
    size_bit_list = [x['size_bit'] for x in memory_comb]
    unroll_list = [x['unroll'] for x in memory_comb]
    total_size_list = list(np.multiply(size_bit_list, unroll_list))
    index_smallest_memory = total_size_list.index(min(total_size_list))
    is_over = False

    init_scheme = MemorySchemeNode(memory_comb)
    if mem_sn.memory_scheme:
        for mem in mem_sn.memory_scheme:
            init_scheme.memory_scheme.add(mem)
    # layer and nextLayer are NOT the neural network layers but a list of MemorySchemeNode objects
    layer = []
    nextLayer = [init_scheme]

    # Assign iteratively the smallest memory still to be assigned to an operand(s)
    while not is_over:
        nextLayer = clean_memory_schemes(nextLayer)

        # The while loop is broken if all the memories in temp_comb for each MemorySchemeNode
        # have been assigned to an operand
        if all([not x.temp_comb for x in nextLayer]):
            is_over = True
            break

        layer = deepcopy(nextLayer)
        nextLayer.clear()

        for msn in layer:  # msn is a single memory scheme node(with mem_scheme and temp_comb), incompleted

            # Find the smallest memory still to be assigned to an operand
            # contained in temp_comb
            size_bit_list = [x['size_bit'] for x in msn.temp_comb]
            unroll_list = [x['unroll'] for x in msn.temp_comb]
            total_size_list = [[], []]
            total_size_list[0] = list(np.multiply(size_bit_list, unroll_list))
            total_size_list[1] = unroll_list
            index_smallest_memory = total_size_list[0].index(min(total_size_list[0]))
            if total_size_list[0].count(min(total_size_list[0])) > 1:
                max_unroll = 0
                for i_sl, sl in enumerate(total_size_list[0]):
                    if sl == min(total_size_list[0]):
                        if total_size_list[1][i_sl] >= max_unroll:
                            max_unroll = total_size_list[1][i_sl]
                for i_sl, sl in enumerate(total_size_list[0]):
                    if total_size_list[0][i_sl] == min(total_size_list[0]) and total_size_list[1][i_sl] == max_unroll:
                        index_smallest_memory = i_sl
                        break
            operand_irrelevant = {'W': [7, 3, 4], 'I': [6], 'O': [1, 2, 5]}

            # The k_list defines what length of combinations of operands can be defined
            # EG k = 1 -> ('I'), ('O'), ('W') assigned separately to the smallest memory, no shared cases
            # EG k = 2 -> ('I','O'), ('I','W'), ('W','O'), ... shared case with 2 operands
            # Moreover, the k can be defined separately for memory within the PE array and outside

            if size_bit_list[index_smallest_memory] in L1_size:  # PE_RF_size_threshold:
                k_list = [1, 2, 3]  # for group 1
            elif size_bit_list[index_smallest_memory] in L2_size:  # PE_RF_size_threshold:
                k_list = [3]
            else:
                k_list = [1]
            for k in k_list:
                operand_comb = combinations(['I', 'O', 'W'], k)
                for oc in operand_comb:
                    if any([msn.temp_comb[index_smallest_memory] == m.memory_level for m in tmp_mem_node_list]):
                        oc_list = [m.operand for m in tmp_mem_node_list if
                                   m.memory_level == msn.temp_comb[index_smallest_memory]]
                        if oc not in oc_list:
                            continue
                    if msn.temp_comb[index_smallest_memory]['size_bit'] in L1_size:
                        if any([x.memory_level['size_bit'] in L1_size for x in msn.memory_scheme \
                                if any([opx in oc for opx in x.operand])]):
                            continue
                    if msn.temp_comb[index_smallest_memory]['size_bit'] in L2_size:
                        if any([x.memory_level['size_bit'] in L2_size for x in msn.memory_scheme \
                                if any([opx in oc for opx in x.operand])]):
                            continue
                    memory_node = MemoryNode(msn.temp_comb[index_smallest_memory], oc, 0, 0)
                    memory_operand_list = [x.memory_level for x in msn.memory_scheme if
                                           set(x.operand).intersection(set(oc))]
                    memory_operand_list_unroll = [x['unroll'] for x in memory_operand_list]
                    if memory_operand_list:
                        if any([memory_node.memory_level['size_bit'] == x['size_bit'] for x in
                                memory_operand_list]):
                            continue

                    new_memory_comb = deepcopy(msn.temp_comb)
                    new_memory_comb.remove(msn.temp_comb[index_smallest_memory])
                    new_msn = deepcopy(msn)
                    new_msn.temp_comb = new_memory_comb
                    new_msn.memory_scheme.add(memory_node)
                    nextLayer.append(new_msn)

    architecture_list = []
    mem_sn_list = [[x.memory_level['size_bit'], list(x.operand)] for x in mem_sn.memory_scheme]
    for s in nextLayer:
        # print('NEW SCHEME')
        m_list = []
        for ii_m, m in enumerate(s.memory_scheme):
            m_list.append([])

            if [m.memory_level['size_bit'], list(m.operand)] in mem_sn_list:
                tmpx = [x for x in mem_sn.memory_scheme if
                        [x.memory_level['size_bit'], list(x.operand)] == [m.memory_level['size_bit'], list(m.operand)]]
                tmpx = tmpx[0]
                m_list[ii_m].append(tmpx)
            else:
                for mem in memory_pool:
                    nbanks = m.memory_level['size_bit'] / mem['size_bit']
                    if nbanks in banking and mem['area'][0] * nbanks < max_area:
                        if mem['size_bit'] * nbanks <= PE_RF_size_threshold:
                            unroll_factor = np.prod(array_dimension)
                        else:
                            unroll_factor = 1
                        tmp_mem = {'name': mem['name'], 'size_bit': mem['size_bit'] * nbanks,
                                   'area': [mem['area'][0] * nbanks],
                                   'cost': [[mem['cost'][0][0] * nbanks, mem['cost'][0][1] * nbanks]],
                                   'mem_bw': [[mem['mem_bw'][0][0] * nbanks, mem['mem_bw'][0][1] * nbanks]],
                                   'mem_type': mem['mem_type'], 'unroll': unroll_factor,
                                   'utilization_rate': mem['utilization_rate'],
                                   'mem_fifo': mem['mem_fifo'], 'nbanks': nbanks}
                        t_m = MemoryNode(tmp_mem, m.operand, m.cluster_level, m.fixed)
                        m_list[ii_m].append(t_m)
                        break
        s_comb = product(*m_list)
        s_combx = [s for s in s_comb]
        for s_c in s_combx:
            msc_sc = MemorySchemeNode([])
            for m_node in s_c:
                msc_sc.memory_scheme.add(m_node)
            architecture_list.append(msc_sc)

    memory_schemes_list += architecture_list  # nextLayer
    memory_schemes_list = check_memory_schemes(memory_schemes_list, array_dimension, float(max_area),
                                               utilization_rate_area)
    return memory_schemes_list


def check_memory_schemes(memory_scheme_list, array_dimension, max_area, utilization_rate_area):
    # Check whether each memory scheme contained in memory scheme list respect the hardware constraints
    # of max area and utilization rate
    good_list = []
    for ms in memory_scheme_list:
        total_area = 0
        if all([any([x in mn.operand for mn in ms.memory_scheme]) for x in ['W', 'I', 'O']]):
            for memory_node in ms.memory_scheme:
                area_memory = min(memory_node.memory_level['area']) * memory_node.memory_level['unroll']
                total_area += area_memory
            if utilization_rate_area <= total_area / max_area <= 1:
                good_list.append(ms)

    return good_list


def clean_memory_schemes(memory_scheme_list):
    # Remove duplicates from memory_scheme_list
    good_list = []
    if memory_scheme_list:
        good_list = [memory_scheme_list[0]]
        for i, msn in enumerate(memory_scheme_list):
            ms_good_list = [x.memory_scheme for x in good_list]
            ms_check = msn.memory_scheme
            node_present = True
            node_equal = []
            for ms in ms_good_list:
                for x in ms_check:
                    if x not in list(ms):
                        node_present = False
                        node_equal.append(node_present)
                        break
            if len(node_equal) == len(ms_good_list):
                good_list.append(msn)

    return good_list


def memory_scheme_generator(mem_pool, array_dimension, max_area, utilization_rate_area,
                            memory_hierarchy_ratio, prune_PE_RF,
                            PE_RF_size_threshold, PE_RF_depth, CHIP_depth, banking, L1_size, L2_size,
                            memory_scheme_hint=MemorySchemeNode([]), tmp_mem_node_list=[], single_sim=0):
    memory_scheme_list = []
    tmp_msn = memory_scheme_hint
    tmp_memory_scheme_list = [tmp_msn]

    memory_scheme_list = deepcopy(tmp_memory_scheme_list)
    memory_scheme_list = clean_memory_schemes(memory_scheme_list)
    tmp_memory_scheme_list.clear()
    schemes = None
    if not single_sim:
        for i, memory_scheme in enumerate(memory_scheme_list):
            area = get_available_area(max_area, memory_scheme)
            array_pool = array_mem_pool(mem_pool, array_dimension, area, PE_RF_size_threshold, tmp_mem_node_list,
                                        banking, L1_size, L2_size)
            memory_comb_list = fitting_memories(array_pool, area, float(max_area), utilization_rate_area, L1_size,
                                                L2_size, float(max_area) - area)
            for ii_mc, memory_comb in enumerate(memory_comb_list):
                if not memory_comb:
                    continue
                ms_list = memory_scheme_generator_cluster(memory_comb, mem_pool, array_dimension, memory_scheme,
                                                          float(max_area), utilization_rate_area, tmp_mem_node_list,
                                                          PE_RF_size_threshold, banking, L1_size, L2_size)
                if not ms_list:
                    ms_list.append(memory_scheme)
                tmp_memory_scheme_list += mem_scheme_not_ordered_check(ms_list, memory_hierarchy_ratio, array_dimension,
                                                                       prune_PE_RF, PE_RF_size_threshold, PE_RF_depth,
                                                                       CHIP_depth)
                print('\r Area-fitting memory combination: ', ii_mc + 1, '/', len(memory_comb_list),
                      '| Valid hierarchy found: ', len(tmp_memory_scheme_list), end='')
            print()

        memory_scheme_list = clean_memory_schemes(tmp_memory_scheme_list)
        tmp_memory_scheme_list.clear()
        schemes = check_memory_schemes(memory_scheme_list, array_dimension, float(max_area), utilization_rate_area)
    else:
        schemes = memory_scheme_list
    return schemes


def msg(mem_pool, array_dimension, max_area, utilization_rate_area, memory_hierarchy_ratio, prune_PE_RF,
        PE_RF_size_threshold, PE_RF_depth, CHIP_depth, tmp_msn, mh_name, tmp_node_list, single_sim, banking, L1_size, L2_size):
    memory_scheme_list = memory_scheme_generator(mem_pool, array_dimension, max_area, utilization_rate_area,
                                                 memory_hierarchy_ratio, prune_PE_RF, PE_RF_size_threshold, PE_RF_depth,
                                                 CHIP_depth, banking, L1_size, L2_size, tmp_msn, tmp_node_list,
                                                 single_sim)
    ms_list = []
    # Conversion from memory_pool format to framework input format
    for memory_scheme_node in memory_scheme_list:
        mem_name = {'W': [], 'I': [], 'O': []}
        mem_size = {'W': [], 'I': [], 'O': []}
        mem_area = {'W': [], 'I': [], 'O': []}
        mem_share = {}
        mem_word_cost = {'W': [], 'I': [], 'O': []}
        mem_type = {'W': [], 'I': [], 'O': []}
        mem_utilization_rate = {'W': [], 'I': [], 'O': []}
        mem_utilization_rate_fix = {'W': [], 'I': [], 'O': []}
        mem_unroll = {'W': [], 'I': [], 'O': []}
        mem_fifo = {'W': [], 'I': [], 'O': []}
        mem_bw = {'W': [], 'I': [], 'O': []}
        mem_nbanks = {'W': [], 'I': [], 'O': []}
        mem_ops = {'W': [], 'I': [], 'O': []}
        if not single_sim: # reset the mh_name
            mh_name = {'W':[], 'I': [], 'O': []}

        mem_list = [x for x in memory_scheme_node.memory_scheme]
        for op in ['W', 'I', 'O']:
            mem_list_op = [x for x in mem_list if op in x.operand]
            mem_list_op.sort()
            for mn in mem_list_op:
                mem_name[op].append(mn.memory_level['name'])
                mem_size[op].append(mn.memory_level['size_bit'])
                mem_area[op].append(mn.memory_level['area'])
                mem_utilization_rate[op].append(mn.memory_level['utilization_rate'])
                mem_utilization_rate_fix[op].append(mn.memory_level['utilization_rate'])
                mem_word_cost[op].append(mn.memory_level['cost'])
                mem_bw[op].append(mn.memory_level['mem_bw'])
                mem_type[op].append(mn.memory_level['mem_type'])
                mem_unroll[op].append(mn.memory_level['unroll'])
                mem_fifo[op].append(mn.memory_level['mem_fifo'])
                mem_nbanks[op].append(mn.memory_level['nbanks'])
                mem_ops[op].append(mn.operand)
                if not single_sim: # if doing hierarchy search mh_name will be empty
                    unique_name = mn.memory_level['name'] + '_' + "".join(mn.operand)
                    mn.unique_name = unique_name
                    mh_name[op].append(unique_name)

        for mem in mem_list:
            if len(mem.operand) > 1:
                shared_list = []
                for i in range(len(mem.operand)):
                    shared_mem = (mem.operand[i], mem_size[mem.operand[i]].index(mem.memory_level['size_bit']))
                    shared_list.append(shared_mem)
                mem_share[len(mem_share)] = shared_list
        
        # LOMA: Construct structured list of memory nodes present in architecture,
        # going from left (closest to PE) to right (closest to off-chip DRAM)
        # inserted with None(s) if an operand has less levels than the max # levels for any operand
        max_levels = max([len(mh_name[op]) for op in ['W','I','O']])
        mh_name_with_none = deepcopy(mh_name)
        for op in ['W','I','O']:
            last_shared_level = -1
            mem_levels_op = len(mh_name[op])
            if mem_levels_op < max_levels:
                for mem_level in range(mem_levels_op):
                    if len(mem_ops[op][mem_level]) > 1:
                        last_shared_level = mem_level
                        for shared_op in mem_ops[op][mem_level]:
                        # shared_op = mem_ops[op][mem_level][1]
                            index_shared_op = mh_name[shared_op].index(mh_name[op][mem_level])
                            if index_shared_op > mem_level:
                                mh_name_with_none[op].insert(mem_level, None)
                                break
            # Check if there are still less levels for this operand (possible if all non-shared)
            # If so, insert None(s) at position after last shared level, as it will give better OMA search time.
            # This is because last level is discarded for TM search using OMA.
            mem_levels_op_updated = len(mh_name_with_none[op])
            if mem_levels_op_updated < max_levels:
                difference = int(max_levels - mem_levels_op_updated)
                for _ in range(difference):
                    mh_name_with_none[op].insert(last_shared_level + 1, None)

        mh_name_left_to_right = zip(mh_name_with_none['W'], mh_name_with_none['I'], mh_name_with_none['O'])
        # Remove duplicate elements at each level
        seen = set()
        mh_name_left_to_right = [[x for x in seq if x not in seen and not seen.add(x)] 
                                for seq in mh_name_left_to_right] 
        # Construct the node list for each memory level
        nodes_set = memory_scheme_node.memory_scheme
        nodes = []
        for mem_level, seq in enumerate(mh_name_left_to_right):
            nodes_level = []
            for unique_name in seq:
                # find corresponding node in the set by iterating through it
                for node in nodes_set:
                    if node.unique_name == unique_name:
                        nodes_level.append(deepcopy(node))
                        if mem_level == 0:
                            break # we have found corresponding node
                        else:
                            # find the node(s) in the below levels connected to this one
                            for op in node.operand:
                                found = False
                                for level_below in range(mem_level-1, -1, -1): # [mem_level-1, mem_level-2, ..., 0]
                                    for node_below in nodes[level_below]:
                                        if op in node_below.operand:
                                            found = True
                                            node_below.set_read_from_above_cost(op, node.memory_level["cost"][0][0])
                                            node_below.set_write_to_above_cost(op, node.memory_level["cost"][0][0])
                                            break
                                    if found:
                                        break

            nodes.append(nodes_level)

        ms = MemoryScheme(mem_name, mem_size, mem_word_cost, mem_utilization_rate, mem_utilization_rate_fix, mem_share,
                          mem_unroll, mem_fifo, mem_bw, mem_type, mem_area, mem_nbanks, nodes)

        ms_list.append(ms)
    return ms_list, memory_scheme_list


def mem_scheme_not_ordered_check(mem_scheme_set_list, memory_hierarchy_ratio, array_dimension, prune_PE_RF,
                                 PE_RF_size_threshold, PE_RF_depth, CHIP_depth):
    good_list = []
    operand_irrelevant = {'W': [7, 3, 4], 'I': [6], 'O': [1, 2, 5]}
    for ii_memscheme, mem_scheme in enumerate(mem_scheme_set_list):

        ii_memscheme += 1
        # print('\r mem check: ', ii_memscheme, '/', len(mem_scheme_set_list), ' good schemes: ', len(good_list), end="")
        mem_scheme_fit = True

        # Check if last level for each operand meets memory requirements of the operand it holds
        for operand in ['W', 'I', 'O']:
            memory_operand_list = [x for x in mem_scheme.memory_scheme if
                                   set(x.operand).intersection(set(operand))]
            memory_operand_list_size = [x.memory_level['size_bit'] for x in memory_operand_list]
            PE_mo_size = 0
            for mol in memory_operand_list_size:
                if mol <= PE_RF_size_threshold:
                    PE_mo_size += 1
            # if memory_operand_list_size:
            #     if max(memory_operand_list_size) != 416777216:
            #         mem_scheme_fit = False
            # break
            # Check if the depth parameters set are respected
            if PE_mo_size > PE_RF_depth or PE_mo_size == 0 or len(
                    memory_operand_list) - PE_mo_size > CHIP_depth:  # or memory_operand_list_size.__len__() - PE_RF_depth < CHIP_depth:
                mem_scheme_fit = False
                # print(PE_mo_size, PE_RF_depth,'Not all operands are present in this scheme')
                break
            # Check if two consecutive levels in the hierarchy for an operand have increasing unrolling
            # If so, prune away
            mo_unroll_list = [mo.memory_level['unroll'] for mo in memory_operand_list]
            for mo_u in mo_unroll_list:
                if mo_u != 1:
                    for mo_u2 in mo_unroll_list:
                        if mo_u2 < mo_u and mo_u % mo_u2 != 0:
                            mem_scheme_fit = False
                            break
            # Check if the size of memories within the PE array is below PE_RF_size_threshold (redundant)
            if prune_PE_RF:
                for mo in memory_operand_list:
                    if mo.memory_level['unroll'] == array_dimension[0] * array_dimension[1]:
                        if mo.memory_level['size_bit'] > PE_RF_size_threshold:
                            mem_scheme_fit = False
                            break
                    if mo.memory_level['size_bit'] < PE_RF_size_threshold:
                        if mo.memory_level['unroll'] == 1:
                            mem_scheme_fit = False
                            break
            memory_ratios = []
            # Check if the memory scheme respects the memory hierarchy ratio
            for m in memory_operand_list:
                memory_ratios = [x.memory_level['size_bit'] / m.memory_level['size_bit'] for x in memory_operand_list if
                                 x != m]
                if any([x > 1 / memory_hierarchy_ratio and x < memory_hierarchy_ratio for x in memory_ratios]):
                    mem_scheme_fit = False
                    break

        if mem_scheme_fit:
            good_list.append(mem_scheme)

    return good_list


# def mem_scheme_check(mem_scheme, spatial_unrolling, precision, layer):
#
#     good_list = []
#     operand_irrelevant = {'W': [7, 3, 4], 'I': [6], 'O': [1, 2, 5]}
#
#     mem_scheme_fit = True
#
#
#     operand_size = {}
#     for operand in ['W', 'I', 'O']:
#         if operand == 'W':
#             operand_size['W'] = layer['FX'] * layer['FY'] * layer['C'] * layer['K'] * precision[operand]
#         elif operand == 'O':
#             operand_size['O'] = layer['OX'] * layer['OY'] * layer['K'] * layer['B'] * precision[operand]
#         elif operand == 'I':
#             operand_size['I'] = (layer['FX'] + layer['OX'] - 1) * (layer['FY'] + layer['OY'] - 1) * layer['C'] * \
#                                 layer['B'] * precision[operand]
#
#     # Check if last level for each operand meets memory requirements of the operand it holds
#     for operand in ['W', 'I', 'O']:
#         total_size = 0
#         for i, mem_share_set in enumerate(mem_scheme.mem_share):
#             if tuple([operand, len(mem_scheme.mem_size[operand]) - 1]) in mem_scheme.mem_share[mem_share_set]:
#                 total_size = np.sum([operand_size[op] for op in operand_size if op in [x[0] for x in mem_scheme.mem_share[mem_share_set]]])
#                 if mem_scheme.mem_size[operand][-1] < total_size:
#                     mem_scheme_fit = False
#                     break
#         if total_size == 0:
#             if mem_scheme.mem_size[operand][-1] < operand_size[operand]:
#                 mem_scheme_fit = False
#                 break
#
#     # Check if the memory scheme can effectively contain the spatial unrollings of levels below
#     if mem_scheme_fit:
#         for operand in ['W', 'I', 'O']:
#             shared = False
#             for level, mem_level in enumerate(mem_scheme.mem_size[operand]):
#                 for i, mem_share_set in enumerate(mem_scheme.mem_share):
#                     if tuple([operand, level]) in mem_scheme.mem_share[mem_share_set]:
#                         tot_size = 0
#                         shared = True
#                         for mshared in mem_scheme.mem_share[mem_share_set]:
#                             block_size = precision[mshared[0]]
#                             for spatial_unrolling_level in range(0, mshared[1] + 1):
#                                 for ur in spatial_unrolling[mshared[0]][spatial_unrolling_level]:
#                                     if ur[0] not in operand_irrelevant[mshared[0]]:
#                                         block_size *= ur[1]
#                             tot_size += block_size
#                         if tot_size > mem_level:
#                             print(tot_size, mem_level)
#                             mem_scheme_fit = False
#                 if not shared:
#                         block_size = precision[operand]
#                         for spatial_unrolling_level in range(0, level + 1):
#                             for ur in spatial_unrolling[operand][spatial_unrolling_level]:
#                                 if ur[0] not in operand_irrelevant[operand]:
#                                     block_size *= ur[1]
#                         if block_size > mem_level:
#                             mem_scheme_fit = False
#
#
#     return mem_scheme_fit

def mem_scheme_fit_check(mem_idx, mem_scheme, precision, layer, layer_number):
    mem_scheme_fit = True

    for layer_idx, each_layer in layer.items():
        if layer_idx in layer_number:
            operand_size = {
                'W': each_layer['FX'] * each_layer['FY'] * each_layer['C'] * each_layer['K'] * precision['W'],
                'O': each_layer['OX'] * each_layer['OY'] * each_layer['K'] * each_layer['B'] * precision['O_final'],
                'I': (each_layer['SX'] * (each_layer['OX'] - 1) +
                      each_layer['SFX'] * (each_layer['FX'] - 1) + 1) * \
                     (each_layer['SY'] * (each_layer['OY'] - 1) +
                      each_layer['SFY'] * (each_layer['FY'] - 1) + 1) * \
                     each_layer['C'] * each_layer['B'] * precision['I']}

            # Check if the all the data can fit in the top level memory.
            for operand in ['W', 'I', 'O']:
                total_size = 0
                for mem_share_set in mem_scheme.mem_share:
                    if tuple([operand, len(mem_scheme.mem_size[operand]) - 1]) in mem_scheme.mem_share[mem_share_set]:
                        shared_mem_list = [x[0] for x in mem_scheme.mem_share[mem_share_set]]
                        total_size = np.sum([operand_size[op] for op in operand_size if op in shared_mem_list])
                        if mem_scheme.mem_size[operand][-1] < total_size:
                            mem_scheme_fit = False
                            print('Memory Scheme %d cannot hold all the data in NN Layer %d.' % (mem_idx, layer_idx),
                                  end=' | ')
                            print('Required memory size:', operand_size[operand], '<-> Available memory size:',
                                  mem_scheme.mem_size[operand][-1], '(unit: bit)')
                            return mem_scheme_fit
                if total_size == 0:
                    if mem_scheme.mem_size[operand][-1] < operand_size[operand]:
                        mem_scheme_fit = False
                        print('Memory Scheme %d cannot hold all the data in NN Layer %d.' % (mem_idx, layer_idx),
                              end=' | ')
                        print('Required memory size:', operand_size[operand], '<-> Available memory size:',
                              mem_scheme.mem_size[operand][-1], 'Operand:', operand, '(unit: bit)')
                        return mem_scheme_fit

    return mem_scheme_fit


def loop_same_term_merge1(unmerged):
    """
    This function merges same type of loops' dimension size at X/Y directions,
    assuming same type of loops are close to each other.
    """
    merged = []

    # change data format from tuple to list
    for level_list in unmerged:
        merged.append([])
        for loop_elem in level_list:
            merged[-1].append(list(loop_elem))

    # merge same type loops within X, Y unrolling list
    for level, level_list in enumerate(unmerged):
        if len(level_list) in [1, 0]:
            continue
        else:
            va_clean_idx = 0
            for va_idx in range(1, len(level_list)):
                if level_list[va_idx - 1][0] == level_list[va_idx][0]:
                    merged[level][va_clean_idx][1] *= level_list[va_idx][1]
                    merged[level].remove(list(level_list[va_idx]))
                    va_clean_idx -= 1
                va_clean_idx += 1

    return merged


def loop_same_term_merge2(unmerged):
    """
    This function merges same type of loops' dimension size at each level,
    without assuming same type of loops are close to each other.
    """
    merged = {'W': [], 'I': [], 'O': []}
    merged_loop_type = {'W': [], 'I': [], 'O': []}
    for operand in ['W', 'I', 'O']:
        for level, level_list in enumerate(unmerged[operand]):
            merged[operand].append([])
            merged_loop_type[operand].append([])
            if len(level_list) > 1:
                for level_elem in level_list:
                    if level_elem[0] not in merged_loop_type[operand][-1]:
                        merged[operand][-1].append(deepcopy(level_elem))
                        merged_loop_type[operand][-1].append(level_elem[0])
                    else:
                        idx = merged_loop_type[operand][-1].index(level_elem[0])
                        merged[operand][-1][idx][1] *= level_elem[1]

    return merged, merged_loop_type


def memory_unroll_candidate_gen(unrolling_scheme, layer):
    """ This function generate EVEN/UNEVEN memory unroll candidate according to unrolling_scheme. """
    B = 1
    K = 1
    C = 1
    OY = 1
    OX = 1
    FY = 1
    FX = 1

    for unroll in unrolling_scheme:
        for loop in unroll:
            if loop[0] == 7:
                B *= loop[1]
            elif loop[0] == 6:
                K *= loop[1]
            elif loop[0] == 5:
                C *= loop[1]
            elif loop[0] == 4:
                OY *= loop[1]
            elif loop[0] == 3:
                OX *= loop[1]
            elif loop[0] == 2:
                FY *= loop[1]
            elif loop[0] == 1:
                FX *= loop[1]

    IX = layer['SX'] * (OX - 1) + layer['SFX'] * (FX - 1) + 1
    IY = layer['SY'] * (OY - 1) + layer['SFY'] * (FY - 1) + 1
    W_unroll = K * C * FY * FX
    I_unroll = B * C * IX * IY
    O_unroll = B * K * OY * OX
    memory_unroll_candidate = {'W': W_unroll, 'I': I_unroll, 'O': O_unroll}
    return memory_unroll_candidate


def spatial_unrolling_generator_uneven(mem_scheme, array_dimension, layer, precision, SU_threshold, SU_mode,
                                       memory_unroll_fully_flexible):
    spatial_loop_list = []
    flooring_list = []

    unrolling_scheme_list = unroll_scheme_list_generator(mem_scheme, array_dimension, layer, precision, SU_threshold,
                                                         SU_mode)

    mem_unroll_candidates = []
    unrolling_scheme_candidates = []
    if memory_unroll_fully_flexible:
        for i, unroll_i in enumerate(unrolling_scheme_list):
            unrolling_scheme_list[i] = loop_same_term_merge1(unroll_i)
            mem_unroll_candidate = memory_unroll_candidate_gen(unrolling_scheme_list[i], layer)
            mem_unroll_candidates.append(mem_unroll_candidate)
            unrolling_scheme_candidates.append(unrolling_scheme_list[i])
    else:
        mem_unroll = {'W': mem_scheme.mem_unroll['W'][0],
                      'I': mem_scheme.mem_unroll['I'][0],
                      'O': mem_scheme.mem_unroll['O'][0]}
        for i, unroll_i in enumerate(unrolling_scheme_list):
            unrolling_scheme_list[i] = loop_same_term_merge1(unroll_i)
            mem_unroll_candidate = memory_unroll_candidate_gen(unrolling_scheme_list[i], layer)
            if mem_unroll_candidate == mem_unroll:
                mem_unroll_candidates.append(mem_unroll_candidate)
                unrolling_scheme_candidates.append(unrolling_scheme_list[i])
                # TODO remove the below "break" when later take interconnection cost into account
                break

    for i, unroll_i in enumerate(unrolling_scheme_candidates):
        spatial_loop = {'W': [], 'I': [], 'O': []}
        flooring = {'W': [], 'I': [], 'O': []}
        for operand in ['W', 'I', 'O']:
            spatial_loop[operand] = [[] for _ in range(len(mem_scheme.mem_size[operand]) + 1)]
            flooring[operand] = [[[], []] for _ in range(len(mem_scheme.mem_size[operand]) + 1)]

        for XY_dim, XY_list in enumerate(unroll_i):
            for XY_elem in XY_list:
                # Weight
                if XY_elem[0] in [7, 4, 3]:
                    spatial_loop['W'][0].append(XY_elem)
                    flooring['W'][0][XY_dim].append(XY_elem[0])
                else:
                    spatial_loop['W'][1].append(XY_elem)
                    flooring['W'][1][XY_dim].append(XY_elem[0])
                # Input
                if XY_elem[0] in [6]:
                    spatial_loop['I'][0].append(XY_elem)
                    flooring['I'][0][XY_dim].append(XY_elem[0])
                else:
                    spatial_loop['I'][1].append(XY_elem)
                    flooring['I'][1][XY_dim].append(XY_elem[0])
                # Output
                if XY_elem[0] in [5, 2, 1]:
                    spatial_loop['O'][0].append(XY_elem)
                    flooring['O'][0][XY_dim].append(XY_elem[0])
                else:
                    spatial_loop['O'][1].append(XY_elem)
                    flooring['O'][1][XY_dim].append(XY_elem[0])
        # spatial_loop, flooring = loop_same_term_merge2(spatial_loop)
        for op in ['W', 'I', 'O']:
            for level, level_list in enumerate(flooring[op]):
                if level_list == [[], []]:
                    flooring[op][level] = []
        spatial_loop_list.append(spatial_loop)
        flooring_list.append(flooring)
        print('mem_unroll', i, mem_unroll_candidates[i])
        print('spatial_loop', i, spatial_loop)
        print('flooring', i, flooring)
        print()

    return spatial_loop_list, flooring_list, mem_scheme, False


def spatial_unrolling_generator_even(mem_scheme, array_dimension, layer, precision, SU_threshold, SU_mode):
    spatial_loop_list = []
    flooring_list = []

    ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
    loops_pf = {
        7: [],
        6: [],
        5: [],
        4: [],
        3: [],
        2: [],
        1: []
    }
    for loop_type in loops_pf:
        loops_pf[loop_type] = su.prime_factors(layer[ll[loop_type]])
    unrolling_scheme_list = unroll_scheme_list_generator(mem_scheme, array_dimension, layer, precision, SU_threshold,
                                                         SU_mode)

    for cluster_scheme in unrolling_scheme_list:
        unrolling_scheme = [[x[0] for x in cluster_scheme[0]], [x[0] for x in cluster_scheme[1]]]
        operand_irrelevant = {'W': [7, 3, 4], 'I': [6], 'O': [1, 2, 5]}
        spatial_loop = {'W': [[]], 'I': [[]], 'O': [[]]}
        flooring = {'W': [[]], 'I': [[]], 'O': [[]]}
        for operand in ['W', 'I', 'O']:
            if len(spatial_loop[operand]) < len(mem_scheme.mem_size[operand]) + 1:
                spatial_loop[operand] += [[]] * ((len(mem_scheme.mem_size[operand]) + 1) - len(spatial_loop[operand]))
            if len(flooring[operand]) < len(mem_scheme.mem_size[operand]) + 1:
                flooring[operand] += [[]] * ((len(mem_scheme.mem_size[operand]) + 1) - len(flooring[operand]))

        not_good = False
        for operand in ['W', 'I', 'O']:
            spatial_loop[operand][0] = deepcopy(cluster_scheme[0] + cluster_scheme[1])
            aux = [[x[0]] for x in cluster_scheme]
            flooring[operand][0] = deepcopy(unrolling_scheme)
        for operand in ['W', 'I', 'O']:

            if mem_scheme.mem_unroll[operand][-1] != 1:
                print('issue: Last level of hierarchy is unrolled >1')
                not_good = True
                break
            if len(mem_scheme.mem_unroll[operand]) > 1:
                for ii_mu, mu in enumerate(mem_scheme.mem_unroll[operand][1:]):
                    if mem_scheme.mem_unroll[operand][ii_mu] < mem_scheme.mem_unroll[operand][ii_mu + 1]:
                        print('issue: Unfeasible unrolling')
                        # return [], [], [], True
                        not_good = True
                        break
            if not_good:
                break
            for ii_level, unroll in enumerate(mem_scheme.mem_unroll[operand]):
                if array_dimension[0] < unroll <= array_dimension[0] * array_dimension[1] and \
                   array_dimension[1] < unroll <= array_dimension[0] * array_dimension[1]:
                    unroll = np.prod([x[1] for x in cluster_scheme[0] + cluster_scheme[1]])
                elif unroll != 1:
                    unroll = np.prod([x[1] for x in cluster_scheme[0] + cluster_scheme[1] if
                                      x[0] not in operand_irrelevant[operand]])
                shared_set = [tuple([operand, ii_level])]
                unroll_level = 1
                unroll_level_below = 1
                for level in spatial_loop[operand][ii_level + 1:]:
                    unroll_level *= np.prod([l[1] for l in level])
                try:
                    unroll_level_below *= np.prod([l[1] for l in spatial_loop[operand][ii_level]])
                except:
                    print('Unfeasible unrolling scheme')
                    # return [], [], [], True
                    not_good = True
                    break

                if unroll_level == unroll:
                    continue
                else:
                    if unroll_level_below == unroll:
                        unrolling = deepcopy(spatial_loop[operand][ii_level])
                        for shared_level in shared_set:
                            spatial_loop[shared_level[0]][shared_level[1] + 1] = deepcopy(unrolling)
                            flooring[shared_level[0]][shared_level[1] + 1] = deepcopy(unrolling_scheme)
                            for ii_level_shared, level_shared in enumerate(
                                    spatial_loop[shared_level[0]][shared_level[1]:shared_level[1] + 1]):
                                for ur in unrolling:
                                    try:
                                        spatial_loop[shared_level[0]][
                                            ii_level_shared + shared_level[1]] = []  # .remove(ur)
                                        flooring[shared_level[0]][ii_level_shared + shared_level[1]] = []
                                    except:
                                        continue
                    else:
                        unrolling_list = []
                        for ii in range(0, ii_level + 1):
                            unrolling_list += [uf for uf in spatial_loop[operand][ii] if uf[1] * unroll_level == unroll]
                        if len(unrolling_list) > 1:
                            unrolling_list = [uf for uf in unrolling_list if uf[0] not in operand_irrelevant[operand]]
                        try:
                            unrolling = unrolling_list[0]
                        except:
                            not_good = True
                            break
                        for shared_level in shared_set:
                            spatial_loop[shared_level[0]][shared_level[1] + 1] = [unrolling]
                            flooring[shared_level[0]][shared_level[1] + 1] = [[unrolling[0]]]
                            for ii_level_shared, level_shared in enumerate(
                                    spatial_loop[shared_level[0]][:shared_level[1] + 1]):
                                try:
                                    spatial_loop[shared_level[0]][ii_level_shared].remove(unrolling)
                                    flooring[shared_level[0]][ii_level_shared].remove([unrolling[0]])
                                except ValueError:
                                    continue
            if not_good:
                break
        if not not_good:
            spatial_loop_list.append(spatial_loop)
            flooring_list.append(flooring)

    # best_spatial_loop_list = []
    # best_flooring_list = []
    # mincost = float('inf')
    # mem_cost = {'W': [], 'I': [], 'O': []}
    # for op in ['W', 'I', 'O']:
    #     readc = []
    #     writec = []
    #     try:
    #         for ii_rc, rc in enumerate(mem_scheme.mem_cost[op]):
    #             readc.append(
    #                 (precision[op] / mem_scheme.mem_bw[op][ii_rc][0][0]) * mem_scheme.mem_cost[op][ii_rc][0][0])
    #             writec.append(
    #                 (precision[op] / mem_scheme.mem_bw[op][ii_rc][0][0]) * mem_scheme.mem_cost[op][ii_rc][0][1])
    #     except:
    #         for ii_rc, rc in enumerate(mem_scheme.mem_cost[op]):
    #             if ii_rc == 0:
    #                 for lv, per_word_rd_cost in enumerate(rc):
    #                     readc.append((precision[op] / mem_scheme.mem_bw[op][lv][0]) * per_word_rd_cost)
    #             elif ii_rc == 1:
    #                 for lv, per_word_wr_cost in enumerate(rc):
    #                     writec.append((precision[op] / mem_scheme.mem_bw[op][lv][1]) * per_word_wr_cost)
    #     mem_cost[op] = [readc, writec]
    #
    # totalMACop = 1
    # output_data_size = 1
    # for loop_type in layer:
    #     if loop_type in ['FX', 'FY', 'OX', 'OY', 'C', 'K', 'B']:
    #         totalMACop *= layer[loop_type]
    #     if loop_type in ['OX', 'OY', 'K']:
    #         output_data_size *= layer[loop_type]
    #
    # for ii_su, sl in enumerate(spatial_loop_list):
    #     energy_su = 0
    #     for op in ['W', 'I', 'O']:
    #         for ii_lev, lev in enumerate(sl[op]):
    #             if ii_lev == 0: continue
    #             totalMACop_tmp = totalMACop
    #             unroll_below = []
    #             for levaux in sl[op][:ii_lev]:
    #                 for pf in levaux: unroll_below.append(pf)
    #             data_reuse = 1
    #             if op in ['W', 'O']:
    #                 for pf in unroll_below:
    #                     if pf[0] in operand_irrelevant[op]: data_reuse *= pf[1]
    #             else:
    #                 data_reuse = get_input_data_reuse(unroll_below, layer)
    #             totalMACop_tmp /= data_reuse
    #             if op in ['W', 'I']:
    #                 energy_su += totalMACop_tmp * (mem_cost[op][0][ii_lev - 1] + mem_cost[op][0][ii_lev - 1])
    #             else:
    #                 energy_su += totalMACop_tmp * mem_cost[op][1][ii_lev - 1]
    #                 energy_su += (totalMACop_tmp - output_data_size) * mem_cost[op][0][ii_lev - 1]
    #
    #     if energy_su == mincost:
    #         best_spatial_loop_list.append(sl)
    #         best_flooring_list.append(flooring_list[ii_su])
    #     if energy_su < mincost:
    #         # print(energy_su)
    #         mincost = energy_su
    #         best_spatial_loop_list.clear()
    #         best_flooring_list.clear()
    #         best_spatial_loop_list.append(sl)
    #         best_flooring_list.append(flooring_list[ii_su])

    # ---------------------------------------------------------------------------------------------HERE----------------
    # return best_spatial_loop_list, best_flooring_list, mem_scheme, not_good
    try:
        return spatial_loop_list, flooring_list, mem_scheme, not_good
    except:
        raise ValueError(
            'No spatial unrolling found. Consider lowering down the spatial_utilization_threshold in setting file.')


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


def spatial_unrolling_generator_with_hint(mem_scheme, array_dimension, layer, unrolling_scheme_list,
                                          memory_unroll_fully_flexible):
    spatial_loop_list = []
    flooring_list = []

    ll = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
    loops_pf = {
        7: [],
        6: [],
        5: [],
        4: [],
        3: [],
        2: [],
        1: []
    }
    for loop_type in loops_pf:
        loops_pf[loop_type] = su.prime_factors(layer[ll[loop_type]])
        if not loops_pf[loop_type]:
            # allow unrolling size to be 1
            loops_pf[loop_type] = [1]
    for i, unrolling_scheme in enumerate(unrolling_scheme_list):
        cluster_scheme = []
        good_scheme = True
        # for unroll_dim in unrolling_scheme:
        #     if any([not loops_pf[x] for x in unroll_dim]):
        #         good_scheme = False
        #         break
        # if not good_scheme:
        #     continue

        for ii_unroll_dim, unroll_dim in enumerate(unrolling_scheme):
            lpf_list = []
            for ud in unroll_dim:
                for lpf in loops_pf[ud]:
                    lpf_list.append([ud, lpf])

            best_unroll_size = 1
            best_comb = None
            for k in range(1, len(lpf_list) + 1):
                uf_comb = combinations(lpf_list, k)
                uf_comb = list(uf_comb)
                for uc in uf_comb:
                    uc_size = np.prod([u[1] for u in uc])
                    if uc_size <= array_dimension[ii_unroll_dim] and uc_size / array_dimension[
                        ii_unroll_dim] > best_unroll_size / array_dimension[ii_unroll_dim]:
                        uc_type, uc_s = zip(*uc)
                        if all([u_dim in uc_type for u_dim in unroll_dim]):
                            best_unroll_size = uc_size
                            best_comb = uc
            if best_unroll_size == 1:
                # good_scheme = False
                # break
                best_comb = ([unroll_dim[0], 1],)
            for u in best_comb:
                if cluster_scheme:
                    u_cs_type, u_cs_size = zip(*cluster_scheme)
                    if u[0] in u_cs_type:
                        cluster_scheme[u_cs_type.index(u[0])][1] *= u[1]
                    else:
                        cluster_scheme.append(list(u))
                else:
                    cluster_scheme.append(list(u))
            if not good_scheme:
                break
        if not good_scheme:
            continue

        if not memory_unroll_fully_flexible:
            operand_irrelevant = {'W': [7, 3, 4], 'I': [6], 'O': [1, 2, 5]}
            spatial_loop = {'W': [[]], 'I': [[]], 'O': [[]]}
            flooring = {'W': [[]], 'I': [[]], 'O': [[]]}
            for operand in ['W', 'I', 'O']:
                if len(spatial_loop[operand]) < len(mem_scheme.mem_size[operand]) + 1:
                    spatial_loop[operand] += [[]] * ((len(mem_scheme.mem_size[operand]) + 1) - len(spatial_loop[operand]))
                if len(flooring[operand]) < len(mem_scheme.mem_size[operand]) + 1:
                    flooring[operand] += [[]] * ((len(mem_scheme.mem_size[operand]) + 1) - len(flooring[operand]))

            not_good = False
            for operand in ['W', 'I', 'O']:
                spatial_loop[operand][0] = deepcopy(cluster_scheme)
                aux = [[x[0]] for x in cluster_scheme]
                flooring[operand][0] = deepcopy(unrolling_scheme)
            for operand in ['W', 'I', 'O']:
                if mem_scheme.mem_unroll[operand][-1] != 1:
                    print('issue: Last level of hierarchy is unrolled >1')
                    not_good = True
                    break
                if len(mem_scheme.mem_unroll[operand]) > 1:
                    for ii_mu, mu in enumerate(mem_scheme.mem_unroll[operand][1:]):
                        if mem_scheme.mem_unroll[operand][ii_mu] < mem_scheme.mem_unroll[operand][ii_mu + 1]:
                            # print('issue: Unfeasible unrolling')
                            # return [], [], [], True
                            not_good = True
                            break
                if not_good:
                    break
                for ii_level, unroll in enumerate(mem_scheme.mem_unroll[operand]):
                    if array_dimension[0] < unroll <= array_dimension[0] * array_dimension[1] and \
                       array_dimension[1] < unroll <= array_dimension[0] * array_dimension[1]:
                        unroll = np.prod([x[1] for x in cluster_scheme])
                    elif unroll != 1:
                        unroll = np.prod([x[1] for x in cluster_scheme if x[0] not in operand_irrelevant[operand]])
                    shared_set = [tuple([operand, ii_level])]
                    unroll_level = 1
                    unroll_level_below = 1
                    for level in spatial_loop[operand][ii_level + 1:]:
                        unroll_level *= np.prod([l[1] for l in level])
                    try:
                        unroll_level_below *= np.prod([l[1] for l in spatial_loop[operand][ii_level]])
                    except:
                        # print('Unfeasible unrolling scheme')
                        # return [], [], [], True
                        not_good = True
                        break

                    if unroll_level == unroll:
                        continue
                    else:
                        if unroll_level_below == unroll:
                            unrolling = deepcopy(spatial_loop[operand][ii_level])
                            for shared_level in shared_set:
                                spatial_loop[shared_level[0]][shared_level[1] + 1] = deepcopy(unrolling)
                                flooring[shared_level[0]][shared_level[1] + 1] = deepcopy(unrolling_scheme)
                                for ii_level_shared, level_shared in enumerate(
                                        spatial_loop[shared_level[0]][shared_level[1]:shared_level[1] + 1]):
                                    for ur in unrolling:
                                        try:
                                            spatial_loop[shared_level[0]][
                                                ii_level_shared + shared_level[1]] = []  # .remove(ur)
                                            flooring[shared_level[0]][ii_level_shared + shared_level[1]] = []
                                        except:
                                            continue
                        else:
                            unrolling_list = []
                            for ii in range(0, ii_level + 1):
                                unrolling_list += [uf for uf in spatial_loop[operand][ii] if uf[1] * unroll_level == unroll]
                            if len(unrolling_list) > 1:
                                unrolling_list = [uf for uf in unrolling_list if uf[0] not in operand_irrelevant[operand]]
                            try:
                                unrolling = unrolling_list[0]
                            except:
                                not_good = True
                                break
                            for shared_level in shared_set:
                                spatial_loop[shared_level[0]][shared_level[1] + 1] = [unrolling]
                                flooring[shared_level[0]][shared_level[1] + 1] = [[unrolling[0]]]
                                for ii_level_shared, level_shared in enumerate(
                                        spatial_loop[shared_level[0]][:shared_level[1] + 1]):
                                    try:
                                        spatial_loop[shared_level[0]][ii_level_shared].remove(unrolling)
                                        flooring[shared_level[0]][ii_level_shared].remove([unrolling[0]])
                                    except ValueError:
                                        continue
                if not_good:
                    break
            if not not_good:
                spatial_loop_list.append(spatial_loop)
                flooring_list.append(flooring)
        else:
            unroll_i = distinguish_XY(cluster_scheme, unrolling_scheme)
            mem_unroll_candidate = memory_unroll_candidate_gen([cluster_scheme], layer)

            spatial_loop = {'W': [], 'I': [], 'O': []}
            flooring = {'W': [], 'I': [], 'O': []}
            for operand in ['W', 'I', 'O']:
                spatial_loop[operand] = [[] for _ in range(len(mem_scheme.mem_size[operand]) + 1)]
                flooring[operand] = [[[], []] for _ in range(len(mem_scheme.mem_size[operand]) + 1)]

            for XY_dim, XY_list in enumerate(unroll_i):
                for XY_elem in XY_list:
                    # Weight
                    if XY_elem[0] in [7, 4, 3]:
                        spatial_loop['W'][0].append(XY_elem)
                        flooring['W'][0][XY_dim].append(XY_elem[0])
                    else:
                        spatial_loop['W'][1].append(XY_elem)
                        flooring['W'][1][XY_dim].append(XY_elem[0])
                    # Input
                    if XY_elem[0] in [6]:
                        spatial_loop['I'][0].append(XY_elem)
                        flooring['I'][0][XY_dim].append(XY_elem[0])
                    else:
                        spatial_loop['I'][1].append(XY_elem)
                        flooring['I'][1][XY_dim].append(XY_elem[0])
                    # Output
                    if XY_elem[0] in [5, 2, 1]:
                        spatial_loop['O'][0].append(XY_elem)
                        flooring['O'][0][XY_dim].append(XY_elem[0])
                    else:
                        spatial_loop['O'][1].append(XY_elem)
                        flooring['O'][1][XY_dim].append(XY_elem[0])

            for op in ['W', 'I', 'O']:
                for level, level_list in enumerate(flooring[op]):
                    if level_list == [[], []]:
                        flooring[op][level] = []
            spatial_loop_list.append(spatial_loop)
            flooring_list.append(flooring)
            print('mem_unroll', i+1, mem_unroll_candidate)
            print('spatial_loop', i+1, spatial_loop)
            print('flooring', i+1, flooring)
            print()

    if not 'not_good' in locals():
        not_good = not (good_scheme)
    return spatial_loop_list, flooring_list, mem_scheme, not_good


def distinguish_XY(cluster_scheme, floor_scheme):
    unroll_i = []
    i = 0
    for floors in floor_scheme:
        unroll_i.append([])
        for floor in floors:
            if floor == cluster_scheme[i][0]:
                unroll_i[-1].append(cluster_scheme[i])
                i += 1
            else:
                raise ValueError("For debug: unrolling and flooring do not follow the same order.")
    return unroll_i


def pareto_clean_3D(dr_list):
    del_idx = []
    for order in [('W', 'I', 'O'), ('W', 'O', 'I'), ('I', 'W', 'O'), ('I', 'O', 'W'), ('O', 'W', 'I'), ('O', 'I', 'W')]:
        dr_prepare_list = []
        for i, dr in enumerate(dr_list):
            dr_prepare_list.append({i: [dr[order[0]], dr[order[1]], dr[order[2]]]})
        dr_prepare_list = sorted(dr_prepare_list,
                                 key=lambda x: (list(x.values())[0][0], list(x.values())[0][1], list(x.values())[0][2]))

        for i in range(1, len(dr_prepare_list)):
            idx_pre = list(dr_prepare_list[i - 1].keys())[0]
            idx_post = list(dr_prepare_list[i].keys())[0]
            li_pre = list(dr_prepare_list[i - 1].values())[0]
            li_post = list(dr_prepare_list[i].values())[0]
            if (li_pre[0] <= li_post[0] and li_pre[1] <= li_post[1] and li_pre[2] <= li_post[2]):
                del_idx.append(idx_pre)
            elif (li_pre[0] >= li_post[0] and li_pre[1] >= li_post[1] and li_pre[2] >= li_post[2]):
                del_idx.append(idx_post)

    del_idx = list(set(del_idx))
    return del_idx


def spatial_loop_same_term_merge(unrolling, flooring):
    spatial_list = {'W': [], 'I': [], 'O': []}
    for operand in ['W', 'I', 'O']:
        for level, level_list in enumerate(flooring[operand]):
            spatial_list[operand].append([])
            if not level_list:
                continue
            else:
                for XY_idx, XY_list in enumerate(level_list):
                    spatial_list[operand][-1].append([])
                    for va in XY_list:
                        spatial_list[operand][-1][-1].append(list(unrolling[operand][level].pop(0)))

    spatial_list_clean = deepcopy(spatial_list)
    for operand in ['W', 'I', 'O']:
        for level, level_list in enumerate(spatial_list[operand]):
            if not level_list:
                continue
            else:
                for XY_idx, XY_list in enumerate(level_list):
                    if len(XY_list) in [1, 0]:
                        continue
                    else:
                        va_clean_idx = 0
                        for va_idx in range(1, len(XY_list)):
                            if XY_list[va_idx - 1][0] == XY_list[va_idx][0]:
                                spatial_list_clean[operand][level][XY_idx][va_clean_idx][1] *= XY_list[va_idx][1]
                                spatial_list_clean[operand][level][XY_idx].remove(XY_list[va_idx])
                                va_clean_idx -= 1
                            va_clean_idx += 1
    return spatial_list_clean


def unroll_scheme_list_generator(mem_scheme, array_dimension, layer, precision, SU_threshold, mode):
    ll2 = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}
    loops_pf = {
        7: [],
        6: [],
        5: [],
        4: [],
        3: [],
        2: [],
        1: []
    }
    mem_cost = {'W': [], 'I': [], 'O': []}
    for op in ['W', 'I', 'O']:
        readc = []
        writec = []
        try:
            for ii_rc, rc in enumerate(mem_scheme.mem_cost[op]):
                readc.append(
                    (precision[op] / mem_scheme.mem_bw[op][ii_rc][0][0]) * mem_scheme.mem_cost[op][ii_rc][0][0])
                writec.append(
                    (precision[op] / mem_scheme.mem_bw[op][ii_rc][0][0]) * mem_scheme.mem_cost[op][ii_rc][0][1])
        except:
            for ii_rc, rc in enumerate(mem_scheme.mem_cost[op]):
                if ii_rc == 0:
                    for lv, per_word_rd_cost in enumerate(rc):
                        readc.append((precision[op] / mem_scheme.mem_bw[op][lv][0]) * per_word_rd_cost)
                elif ii_rc == 1:
                    for lv, per_word_wr_cost in enumerate(rc):
                        writec.append((precision[op] / mem_scheme.mem_bw[op][lv][1]) * per_word_wr_cost)
        mem_cost[op] = [readc, writec]

    '''operand size'''
    opsize = {'W': layer['K'] * layer['C'] * layer['FY'] * layer['FX'],
              'I': (layer['SX'] * (layer['OX'] - 1) + layer['SFX'] * (layer['FX'] - 1) + 1) * \
                   (layer['SY'] * (layer['OY'] - 1) + layer['SFY'] * (layer['FY'] - 1) + 1) * layer['C'] * layer['B'],
              'O': layer['B'] * layer['K'] * layer['OY'] * layer['OX']}
    totalMACop = layer['B'] * layer['K'] * layer['C'] * layer['OY'] * layer['OX'] * layer['FY'] * layer['FX']
    '''maximum data reuse'''
    mdr = {'W': totalMACop / opsize['W'], 'I': totalMACop / opsize['I'], 'O': totalMACop / opsize['O']}

    lpf_list = []
    for loop_type in loops_pf:
        loops_pf[loop_type] = su.prime_factors(layer[ll2[loop_type]])
        for lp in loops_pf[loop_type]:
            lpf_list.append(tuple([loop_type, lp]))
    unrolling_scheme_list = []
    for k in range(1, len(lpf_list)):
        uf_comb = combinations(lpf_list, k)
        uf_comb = list(uf_comb)
        uf_comb = list(dict.fromkeys(uf_comb))
        for comb in uf_comb:
            comb = list(comb)
            comb.sort()
            if array_dimension[0] >= np.prod([x[1] for x in comb]) > array_dimension[0] * SU_threshold:
                lpf2_list = deepcopy(lpf_list)
                for pf in comb:
                    lpf2_list.remove(pf)
                for k2 in range(1, len(lpf2_list)):
                    uf2_comb = combinations(lpf2_list, k2)
                    uf2_comb = list(uf2_comb)
                    uf2_comb = list(dict.fromkeys(uf2_comb))
                    for comb2 in uf2_comb:
                        comb2 = list(comb2)
                        comb2.sort()
                        if array_dimension[1] >= np.prod([x[1] for x in comb2]) > array_dimension[1] * SU_threshold:
                            combaux = [[x for x in comb2], [x2 for x2 in comb]]
                            if combaux not in unrolling_scheme_list:
                                unrolling_scheme_list.append([[x for x in comb], [x2 for x2 in comb2]])

    '''spatial data reuse list'''
    sdr_list = []
    '''temporal data reuse list'''
    tdr_list = []
    for ii_us, us in enumerate(unrolling_scheme_list):
        us_aux = us[0] + us[1]
        layer_sp = {'B': 1, 'K': 1, 'C': 1, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1,
                    'SY': layer['SY'], 'SX': layer['SX'], 'SFY': layer['SFY'], 'SFX': layer['SFX']}

        for va in us_aux:
            layer_sp[ll2[va[0]]] *= va[1]
        '''_sp means at spatial level'''
        totalMACop_sp = layer_sp['B'] * layer_sp['K'] * layer_sp['C'] * layer_sp['OY'] * layer_sp['OX'] * \
                        layer_sp['FY'] * layer_sp['FX']
        opsize_sp = {'W': layer_sp['K'] * layer_sp['C'] * layer_sp['FY'] * layer_sp['FX'],
                     'I': (layer_sp['SX'] * (layer_sp['OX'] - 1) + layer_sp['SFX'] * (layer_sp['FX'] - 1) + 1) * \
                          (layer_sp['SY'] * (layer_sp['OY'] - 1) + layer_sp['SFY'] * (layer_sp['FY'] - 1) + 1) * \
                          layer_sp['C'] * layer_sp['B'],
                     'O': layer_sp['B'] * layer_sp['K'] * layer_sp['OY'] * layer_sp['OX']}
        '''spatial data reuse'''
        sdr = {'W': totalMACop_sp / opsize_sp['W'], 'I': totalMACop_sp / opsize_sp['I'],
               'O': totalMACop_sp / opsize_sp['O']}
        tdr = {'W': mdr['W'] / sdr['W'], 'I': mdr['I'] / sdr['I'], 'O': mdr['O'] / sdr['O']}
        sdr_list.append(sdr)
        tdr_list.append(tdr)

    ''' exhaustive search '''
    if mode == 0:
        # for ii, m in enumerate(unrolling_scheme_list):
        #     print(ii + 1, m)
        #     print(opsize)
        #     print('MAX DATA REUSE: ', mdr)
        #     print('SPATIAL DATA REUSE: ', sdr_list[ii])
        #     print('TEMPORAL DATA REUSE: ', tdr_list[ii])
        #     print()
        return (unrolling_scheme_list)

    ''' heuristic search v1: with unrolling_scheme_list same spatial data reuse clean up '''
    if mode == 1:
        sdr_list_clean = []
        tdr_list_clean = []
        unrolling_scheme_list_clean = []
        for i, dr in enumerate(sdr_list):
            if dr not in sdr_list_clean:
                sdr_list_clean.append(dr)
                tdr_list_clean.append(tdr_list[i])
                unrolling_scheme_list_clean.append(unrolling_scheme_list[i])

        # for ii, m in enumerate(unrolling_scheme_list_clean):
        # print(ii + 1, m)
        # print(opsize)
        # print('MAX DATA REUSE: ', mdr)
        # print('SPATIAL DATA REUSE: ', sdr_list_clean[ii])
        # print('TEMPORAL DATA REUSE: ', tdr_list_clean[ii])
        # print()
        return (unrolling_scheme_list_clean)

    ''' heuristic search v2: with unrolling_scheme_list same spatial data reuse clean up + Pareto optimum clean up'''
    if mode == 2:
        sdr_list_clean = []
        tdr_list_clean = []
        unrolling_scheme_list_clean = []
        for i, dr in enumerate(sdr_list):
            if dr not in sdr_list_clean:
                sdr_list_clean.append(dr)
                tdr_list_clean.append(tdr_list[i])
                unrolling_scheme_list_clean.append(unrolling_scheme_list[i])

        sdr_list_clean2 = deepcopy(sdr_list_clean)
        tdr_list_clean2 = deepcopy(tdr_list_clean)
        unrolling_scheme_list_clean2 = deepcopy(unrolling_scheme_list_clean)

        del_idx = pareto_clean_3D(sdr_list_clean)

        if del_idx:
            for idx in del_idx:
                sdr_list_clean2.remove(sdr_list_clean[idx])
                tdr_list_clean2.remove(tdr_list_clean[idx])
                unrolling_scheme_list_clean2.remove(unrolling_scheme_list_clean[idx])

        # for ii, m in enumerate(unrolling_scheme_list_clean2):
        # print(ii + 1, m)
        # print(opsize)
        # print('MAX DATA REUSE: ', mdr)
        # print('SPATIAL DATA REUSE: ', sdr_list_clean2[ii])
        # print('TEMPORAL DATA REUSE: ', tdr_list_clean2[ii])
        # print()
        return (unrolling_scheme_list_clean2)


def su_reformat(spatial_unrolling, ideal_su_old, fraction_su_old):
    ideal_su = deepcopy(spatial_unrolling)
    fraction_su = deepcopy(spatial_unrolling)
    for op in ['W', 'I', 'O']:
        for level, outer_list in enumerate(spatial_unrolling[0][op]):
            if outer_list:
                for idx, inner_list in enumerate(outer_list):
                    locate_corresponding_id_su = [x for x in ideal_su_old if x[0] == inner_list[0]]
                    locate_corresponding_fr_su = [x for x in fraction_su_old if x[0] == inner_list[0]]
                    ideal_su[0][op][level][idx] = locate_corresponding_id_su[0]
                    fraction_su[0][op][level][idx] = locate_corresponding_fr_su[0]
    return ideal_su, fraction_su


def iterative_data_format_clean(original_dict):
    new_dict = {'W': [], 'I': [], 'O': []}
    for operand in ['W', 'I', 'O']:
        for li in original_dict[operand]:
            new_dict[operand].append(li[0])
    return new_dict


def get_mem_scheme_area(mem_scheme, ii_su):
    """
    This function computes total memory occupied area.
    It distinguishes active area and total area.
    total area = active area + dark silicon area
    """

    total_area = 0
    active_area = 0

    if type(mem_scheme.mem_area['W'][0]) in [list, tuple]:
        mem_scheme.mem_area = iterative_data_format_clean(mem_scheme.mem_area)
    for op in ['W', 'I', 'O']:
        for level, mem_area in enumerate(mem_scheme.mem_area[op]):

            try:
                mem_unroll = mem_scheme.mem_unroll_complete['mem_unroll_total'][ii_su][op][level]
            except:
                mem_unroll = mem_scheme.mem_unroll[op][level]

            index_unroll_shared = [tuple([op, level]) in mem_scheme.mem_share[x] for x in
                                   mem_scheme.mem_share]
            if any(index_unroll_shared):
                level_area_active = mem_area * mem_unroll \
                                    / len(mem_scheme.mem_share[index_unroll_shared.index(True)])
                level_area_total = mem_area * mem_unroll \
                                   / len(mem_scheme.mem_share[index_unroll_shared.index(True)])
            else:
                level_area_active = mem_area * mem_unroll
                level_area_total = mem_area * mem_unroll
            active_area += level_area_active
            total_area += level_area_total

    return total_area, active_area

def get_mem_scheme_area2(mem_scheme, unit_count, spatial_utilization):
    """
    This function computes total memory occupied area.
    It distinguishes active area and total area.
    total area = active area + dark silicon area
    """

    total_area = 0
    active_area = 0
    if type(mem_scheme.mem_area['W'][0]) in [list, tuple]:
        mem_scheme.mem_area = iterative_data_format_clean(mem_scheme.mem_area)
    for op in ['W', 'I', 'O']:
        for level, mem_area in enumerate(mem_scheme.mem_area[op]):
            index_unroll_shared = [tuple([op, level]) in mem_scheme.mem_share[x] for x in
                                   mem_scheme.mem_share]
            if any(index_unroll_shared):
                level_area_active = mem_area * unit_count[op][level + 1] / len(
                    mem_scheme.mem_share[index_unroll_shared.index(True)])
                if unit_count[op][level + 1] > 1:
                    level_area_total = level_area_active / spatial_utilization
                else:
                    level_area_total = level_area_active
            else:
                level_area_active = mem_area * unit_count[op][level + 1]
                if unit_count[op][level + 1] > 1:
                    level_area_total = level_area_active / spatial_utilization
                else:
                    level_area_total = level_area_active
            active_area += level_area_active
            total_area += level_area_total

    return total_area, active_area


def update_mem_scheme_bw(mem_scheme, utilization):
    # Set correct area, bandwidth
    mem_bw_non_shared_final = {'W': [], 'I': [], 'O': []}
    mem_bw_shared_final = {'W': [], 'I': [], 'O': []}

    mem_scheme_shared_bw = deepcopy(mem_scheme)
    mem_scheme_non_shared_bw = deepcopy(mem_scheme)

    for operand in ['W', 'I', 'O']:
        for level, mem in enumerate(mem_scheme.mem_size[operand]):
            shared_level_list = [tuple([operand, level])]
            index_shared = [tuple([operand, level]) in mem_scheme.mem_share[x] for x in mem_scheme.mem_share]
            if any(index_shared):
                shared_level_list = mem_scheme.mem_share[index_shared.index(True)]
            shared_bw = []
            shared_bw_sum = []
            for shared_level in shared_level_list:
                shared_bw.append(max(utilization.req_mem_bw_bit[shared_level[0]][shared_level[1]]))
                shared_bw_sum.append(max(utilization.req_sh_mem_bw_bit[shared_level[0]][shared_level[1]]))
            req_bw_non_shared = max(shared_bw)
            req_bw_shared = max(shared_bw_sum)
            actual_bw_non_shared = float('inf')
            actual_bw_shared = float('inf')

            for bw in mem_scheme.mem_bw[operand][level]:
                if req_bw_non_shared <= bw[0] and bw[0] <= actual_bw_non_shared:
                    actual_bw_non_shared = bw[0]
                if req_bw_shared <= bw[0] and bw[0] <= actual_bw_shared:
                    actual_bw_shared = bw[0]

            if actual_bw_non_shared == float('inf'):
                actual_bw_non_shared = max([x[0] for x in mem_scheme.mem_bw[operand][level]])
            if actual_bw_shared == float('inf'):
                actual_bw_shared = max([x[0] for x in mem_scheme.mem_bw[operand][level]])

            index_best_bw_non_shared = mem_scheme.mem_bw[operand][level].index(
                [actual_bw_non_shared, actual_bw_non_shared])
            mem_scheme_non_shared_bw.mem_area[operand][level] = mem_scheme.mem_area[operand][level][
                index_best_bw_non_shared]
            mem_scheme_non_shared_bw.mem_cost[operand][level] = mem_scheme.mem_cost[operand][level][
                index_best_bw_non_shared]
            mem_bw_non_shared_final[operand].append(mem_scheme.mem_bw[operand][level][index_best_bw_non_shared])

            index_best_bw_shared = mem_scheme.mem_bw[operand][level].index([actual_bw_shared, actual_bw_shared])
            mem_scheme_shared_bw.mem_area[operand][level] = mem_scheme.mem_area[operand][level][index_best_bw_shared]
            mem_scheme_shared_bw.mem_cost[operand][level] = mem_scheme.mem_cost[operand][level][index_best_bw_shared]
            mem_bw_shared_final[operand].append(mem_scheme.mem_bw[operand][level][index_best_bw_shared])

        # mem_cost_reformatted = [[], []]
        # for mc in mem_scheme_non_shared_bw.mem_cost[operand]:
        #     read_cost, write_cost = zip(mc)
        #     mem_cost_reformatted[0] += list(read_cost)
        #     mem_cost_reformatted[1] += list(write_cost)
        # mem_scheme_non_shared_bw.mem_cost[operand] = mem_cost_reformatted
        #
        # mem_cost_reformatted = [[], []]
        # for mc in mem_scheme_shared_bw.mem_cost[operand]:
        #     read_cost, write_cost = zip(mc)
        #     mem_cost_reformatted[0] += list(read_cost)
        #     mem_cost_reformatted[1] += list(write_cost)
        # mem_scheme_shared_bw.mem_cost[operand] = mem_cost_reformatted

    mem_scheme_non_shared_bw.mem_bw = mem_bw_non_shared_final
    mem_scheme_shared_bw.mem_bw = mem_bw_shared_final

    return mem_scheme_non_shared_bw, mem_scheme_shared_bw