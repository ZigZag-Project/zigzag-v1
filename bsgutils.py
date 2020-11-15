from copy import deepcopy
import numpy as np
import math
import time
from itertools import combinations


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def input_relevant_size_below(LPF_scheme, mem_level, layer_loop_info):
    FX_size = 1
    FY_size = 1
    OX_size = 1
    OY_size = 1
    C_size = 1
    B_size = 1
    for y in range(0, mem_level + 1):
        for x in range(0, len(LPF_scheme['I'][y])):
            if LPF_scheme['I'][y][x][0] == 1:
                FX_size *= LPF_scheme['I'][y][x][1]
            if LPF_scheme['I'][y][x][0] == 2:
                FY_size *= LPF_scheme['I'][y][x][1]
            if LPF_scheme['I'][y][x][0] == 3:
                OX_size *= LPF_scheme['I'][y][x][1]
            if LPF_scheme['I'][y][x][0] == 4:
                OY_size *= LPF_scheme['I'][y][x][1]
            if LPF_scheme['I'][y][x][0] == 5:
                C_size *= LPF_scheme['I'][y][x][1]
            if LPF_scheme['I'][y][x][0] == 7:
                B_size *= LPF_scheme['I'][y][x][1]

    IX_size = layer_loop_info['SX'] * (OX_size - 1) + layer_loop_info['SFX'] * (FX_size - 1) + 1
    IY_size = layer_loop_info['SY'] * (OY_size - 1) + layer_loop_info['SFY'] * (FY_size - 1) + 1
    return IX_size * IY_size * C_size * B_size


def check_node(blocking_node, mem_size, operand_irrelevant, mem_share, precision, layer_loop_info, utilization_rate):
    good = True
    # Given a blocking_node (with loops_pf, roof and LPF_scheme) checks whether the LPFs assigned
    # in its LPF_scheme don't exceed the storage parameters per memory level

    for op in blocking_node:

        for lev in range(0, len(blocking_node[op])):
            shared_min_roof_levels = [tuple([op, lev])]
            for shared_set in mem_share:
                if tuple([op, lev]) in mem_share[shared_set]:
                    shared_min_roof_levels = mem_share[shared_set]
            tot_size = 0
            for i in range(0, len(shared_min_roof_levels)):
                if shared_min_roof_levels[i][0] == 'I':
                    rel_loop_size = input_relevant_size_below(blocking_node, shared_min_roof_levels[i][1], layer_loop_info) * precision[
                        shared_min_roof_levels[i][0]]

                else:
                    rel_loop_size = precision[shared_min_roof_levels[i][0]]
                    for lev_below in range(0, shared_min_roof_levels[i][1] + 1):
                        try:
                            if blocking_node[shared_min_roof_levels[i][0]][lev_below]:
                                block_types, block_sizes = zip(
                                    *blocking_node[shared_min_roof_levels[i][0]][lev_below])
                                for bt in range(0, len(block_types)):
                                    if block_types[bt] not in operand_irrelevant[shared_min_roof_levels[i][0]]:
                                        rel_loop_size *= block_sizes[bt]
                        except IndexError:
                            rel_loop_size = precision[shared_min_roof_levels[i][0]]
                tot_size += rel_loop_size
            if tot_size > mem_size[op][lev]:
                # print(blocking_node[op])
                # print(op, lev, utilization_rate[op][lev], tot_size)
                good = False
    return good


def cleaner(LPF_schemes_list):
    # Given a list of SchedulerNodes, delete all the duplicate ones
    clean_nodes = []
    for ii_blocking_node, blocking_node in enumerate(LPF_schemes_list):
        is_present = False
        for bn in clean_nodes:
            if bn.LPF_scheme == blocking_node.LPF_scheme and \
                    bn.roof == blocking_node.roof:
                is_present = True
                break
        if not is_present:
            clean_nodes.append(blocking_node)

    return clean_nodes


def check_comb_fit(LPF_scheme, spatial_unrolling, comb, min_roof, mem_size, mem_share, utilization_rate, precision,
                   operand_irrelevant, is_min_roof, layer_loop_info):
    # Given a LPF_scheme and a combination of LPFs check whether they fit in the roof provided
    total_size = 0
    is_fit = True
    
    # Find set of levels that are shared with the min_roof
    shared_min_roof_levels = [tuple([min_roof[0], min_roof[1]])]
    for shared_set in mem_share:
        if tuple([min_roof[0], min_roof[1]]) in mem_share[shared_set]:
            shared_min_roof_levels = mem_share[shared_set]

    # Find the total size occupied by summing up the LPF size contained in
    # LPF_scheme + comb for all shared operands, check whether it fits in the level of the min roof
    for i in range(0, len(shared_min_roof_levels)):
        tmp_LPF_scheme = deepcopy(LPF_scheme)
        # Append the LPFs contained in comb to the LPF scheme
        tmp_LPF_scheme[shared_min_roof_levels[i][0]][shared_min_roof_levels[i][1]] = deepcopy(
            tmp_LPF_scheme[shared_min_roof_levels[i][0]][shared_min_roof_levels[i][1]] + list(comb))

        # Compute the size in bits of the LPF contained in the tmp LPF scheme if operand is 'I' or otherwise
        # While for the other operands is enough to compute the product of the sizes of the relevant LPFs, for 'I'
        # it is necessary to take into account FX, OX, FY, OY, stride values
        if shared_min_roof_levels[i][0] == 'I':
            block_size = input_relevant_size_below(tmp_LPF_scheme, shared_min_roof_levels[i][1], layer_loop_info) * \
                         precision[shared_min_roof_levels[i][0]]
        else:
            block_size = precision[shared_min_roof_levels[i][0]]
            for z in range(0, shared_min_roof_levels[i][1] + 1):
                if tmp_LPF_scheme[shared_min_roof_levels[i][0]][z]:
                    # print(tmp_LPF_scheme)
                    block_types, block_sizes = zip(*tmp_LPF_scheme[shared_min_roof_levels[i][0]][z])
                    for j in range(0, len(block_types)):
                        if block_types[j] not in operand_irrelevant[shared_min_roof_levels[i][0]]:
                            block_size *= block_sizes[j]
        total_size += block_size
    # The comb is considered to be NOT fit if:
    # - The total size is larger than the mem level size
    # - The total size is lower than the utilization rate if is_min_roof is True
    if is_min_roof and not min_roof[1] == len(mem_size[min_roof[0]]) - 1:
        if total_size > mem_size[shared_min_roof_levels[0][0]][shared_min_roof_levels[0][1]] or total_size / \
                mem_size[shared_min_roof_levels[0][0]][shared_min_roof_levels[0][1]] < \
                utilization_rate[min_roof[0]][min_roof[1]]:
            is_fit = False
    else:
        if total_size > mem_size[shared_min_roof_levels[0][0]][shared_min_roof_levels[0][1]]:
            is_fit = False
    # if is_fit == False:
    #      print()
    #      print('COMB',comb)
    #     for opx in ['W','I','O']:
    #         print(opx, tmp_LPF_scheme[opx])
    #     print(min_roof)
    #     print('total size', total_size)
    # print(mem_size)
    # print('ur ', total_size / mem_size[min_roof[0]][min_roof[1]] < utilization_rate[min_roof[0]][min_roof[1]])

    return is_fit


def update_roof(LPF_scheme, spatial_unrolling, fitting_combination, old_roof, mem_share, mem_size, precision, operand_irrelevant,
                loops_pf, layer_loop_info):
    # The function updates the max blocks available for each operand at the specified level in the roof
    # taking into account the partial LPF scheme with the fitting combination stacked on top of it
    tmp_LPF_scheme = deepcopy(LPF_scheme)
    tmp_LPF_scheme2 = deepcopy(LPF_scheme)
    for operand in tmp_LPF_scheme:
        list_fitting_comb = deepcopy(fitting_combination)
        tmp_LPF_scheme[operand][old_roof[operand][0]] = deepcopy(tmp_LPF_scheme[operand][
                                                                          old_roof[operand][0]] + fitting_combination)
        tmp_LPF_scheme2[operand][old_roof[operand][0]] = deepcopy(tmp_LPF_scheme[operand][
                                                                           old_roof[operand][0]] + fitting_combination)
    new_roof = deepcopy(old_roof)
    # For each operand:
    # - If the respective roof memory level is NOT SHARED compute the space available still and divide by the precision of the operand
    # - If the respective roof memory level is SHARED find the max size of combination of relevant LPFs that fit in the shared level
    for operand in old_roof:
        tot_size = 0
        shared_roof = []
        for shared_set in mem_share:
            if tuple([operand, old_roof[operand][0]]) in mem_share[shared_set]:
                shared_roof = mem_share[shared_set]
        if not shared_roof:
            shared_roof = [tuple([operand, old_roof[operand][0]])]
        # NOT SHARED CASE (the shared roof is the roof of a single operand)
        if len(shared_roof) == 1:
            if shared_roof[0][0] == 'I':
                block_size = input_relevant_size_below(tmp_LPF_scheme, shared_roof[0][1], layer_loop_info) * precision[
                    shared_roof[0][0]]
            else:
                block_size = precision[shared_roof[0][0]]
                for j in range(0, shared_roof[0][1] + 1):
                    for k in range(0, len(tmp_LPF_scheme[shared_roof[0][0]][j])):
                        if tmp_LPF_scheme[shared_roof[0][0]][j][k][0] not in operand_irrelevant[shared_roof[0][0]]:
                            block_size *= tmp_LPF_scheme[shared_roof[0][0]][j][k][1]
            tot_size = block_size
            new_roof[operand][1] = math.floor(mem_size[operand][old_roof[operand][0]] / tot_size)
        # SHARED CASE
        else:
            loop_blocks = []
            max_blocks_available = 1
            blocks_available = 1
            for loop_type in loops_pf:
                irrel_loops = list(operand_irrelevant[opit] for opit in operand_irrelevant if
                                   opit in [op[0] for op in shared_roof])
                if any(loop_type not in i_l for i_l in irrel_loops):
                    for i in range(0, len(loops_pf[loop_type])):
                        loop_blocks.append(tuple([loop_type, loops_pf[loop_type][i]]))

            # Check for all the combinations of LPFs still to be assigned what is the max size of the combination of 
            # relevant LPFs that fit in the shared level
            for k in range(1, len(loop_blocks) + 1):
                comb = []
                tmp_comb = combinations(loop_blocks, k)
                for x in tmp_comb:
                    if x not in comb:
                        comb.append(x)
                for j in range(0, len(comb)):
                    if operand == 'I':
                        tmp_bs = {'I': [comb[j]]}
                        blocks_available = input_relevant_size_below(tmp_bs, 0, layer_loop_info)
                    else:
                        blocks_available = np.prod(
                            list(lpf[1] for lpf in comb[j] if lpf[0] not in operand_irrelevant[operand]))
                    if blocks_available < max_blocks_available:
                        continue
                    is_fit = True
                    rf = [operand, old_roof[operand][0], 0]
                    is_fit = check_comb_fit(tmp_LPF_scheme2, spatial_unrolling, comb[j], rf, mem_size, mem_share, [],
                                            precision, operand_irrelevant, False, layer_loop_info)
                    if not is_fit:
                        continue
                    max_blocks_available = blocks_available

            new_roof[operand][1] = max_blocks_available

    return new_roof


def utilization_rate_optimizer(mem_size, spatial_unroll, layer, precision, utilization_rate, unit_unique):
    # Adjusts the utilization rate set by the user in the input file
    # so as to meet the max utilization rate of the memory levels that exceed the memory requirements for the 
    # operands in the layer
    # EG : the last level of memory for O has a size of 2MByte and a utilization rate set by the user of 0.7,
    #   However the max size occupied by the Outputs in the specified layer is 100KByte, so the 0.7 utilization
    #   rate would never be met. To avoid not finding any fitting LPF combination, the utilization rate is 
    #   adjusted accordingly

    operand_size = {}
    layerc2a = {7: 'B', 6: 'K', 5: 'C', 4: 'OY', 3: 'OX', 2: 'FY', 1: 'FX'}
    for operand in ['W', 'I', 'O']:
        layer_cleaned = deepcopy(layer)
        if len(spatial_unroll[operand]) - 1 == len(mem_size[operand]) and spatial_unroll[operand][-1]:
            layer_cleaned[layerc2a[spatial_unroll[operand][-1][0][0]]] /= spatial_unroll[operand][-1][0][1]
        if operand == 'W':
            operand_size['W'] = layer_cleaned['FX'] * layer_cleaned['FY'] * layer_cleaned['C'] * layer_cleaned['K'] * precision[operand]
        elif operand == 'O':
            operand_size['O'] = layer_cleaned['OX'] * layer_cleaned['OY'] * layer_cleaned['K'] * layer_cleaned['B'] * precision[operand]
        elif operand == 'I':
            operand_size['I'] = (layer_cleaned['SX'] * (layer_cleaned['OX'] - 1) + layer_cleaned['SFX'] * (layer_cleaned['FX'] - 1) + 1) * \
                                (layer_cleaned['SY'] * (layer_cleaned['OY'] - 1) + layer_cleaned['SFY'] * (layer_cleaned['FY'] - 1) + 1) * \
                                layer_cleaned['C'] * layer_cleaned['B'] * precision[operand]

    for operand in ['W', 'I', 'O']:
        for level, level_size in enumerate(mem_size[operand]):
            max_utilization = operand_size[operand] / level_size / unit_unique[operand][level+1]
            if max_utilization < utilization_rate[operand][level]:
                utilization_rate[operand][level] = max_utilization * 0.9
    return utilization_rate, True


def check_mem_ut_after_CM(actual_mem_ut1, actual_mem_ut2, req_mem_ut, current_best_en, current_best_ut,
                          previous_best_en, previous_best_ut):
    """
    This function compare current best design points found with the previous ones.
    If the current design points are better, it also checks ig the memory utilization TH is met or not,
    based on which, it decides whether to continue internal memory utilization threshold adjusting while loop.
    """
    redo_flag = False
    req_mem_ut_update = deepcopy(req_mem_ut)
    if not (current_best_en == previous_best_en and current_best_ut == previous_best_ut):
        for op in ['W', 'I', 'O']:
            for level, ut in enumerate(req_mem_ut[op]):
                if actual_mem_ut1[op][level] < ut or actual_mem_ut2[op][level] < ut:
                    req_mem_ut_update[op][level] = min(actual_mem_ut1[op][level], actual_mem_ut2[op][level])*0.95
                    # req_mem_ut_update[op][level] *= 0.8
                    redo_flag = True

    return redo_flag, req_mem_ut_update


class SchedulerNode:
    LPF_scheme = {
        'I': [],
        'O': [],
        'W': []
    }
    roof = {
        'W': [0, 0],
        'I': [0, 0],
        'O': [0, 0]
    }
    loops_pf = {
        7: [],
        6: [],
        5: [],
        4: [],
        3: [],
        2: [],
        1: []
    }

    leaf_over = False

    def __init__(self, LPF_scheme, roof, loops_pf):
        self.LPF_scheme = LPF_scheme
        self.roof = roof
        self.loops_pf = loops_pf
        self.leaf_over = False

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def set_leaf_over(self):
        self.leaf_over = True
